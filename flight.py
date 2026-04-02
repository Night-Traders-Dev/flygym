"""
flight.py — Flight system for Drosophila in NeuroMechFly v2

Applies aerodynamic lift via PD altitude control on xfrc_applied,
with wing beat animation for visual fidelity.

The lift force is computed as:
  F_z = (weight + Kp * altitude_error - Kd * vertical_velocity) * throttle_ramp
This produces stable hover at the target altitude.
"""

import numpy as np
import mujoco as mj


class WingBeatController:
    """Generates wing beat patterns for visual animation.

    Drosophila wing kinematics (simplified):
      Stroke (yaw): 200Hz sweep, ±70°
      Pitch: ~45° AoA, flips at reversal
      Roll: small deviation
    """

    def __init__(self, timestep, freq=200.0):
        self.timestep = timestep
        self.freq = freq
        self.phase = 0.0
        self.stroke_amplitude = np.radians(70)
        self.pitch_amplitude = np.radians(45)
        self.roll_amplitude = np.radians(8)

    def step(self, throttle=1.0, pitch_bias=0.0, roll_bias=0.0, yaw_bias=0.0):
        self.phase += 2 * np.pi * self.freq * self.timestep
        t = self.phase
        amp = self.stroke_amplitude * np.clip(throttle, 0, 1.2)

        l_yaw = amp * (1.0 + yaw_bias * 0.2) * np.sin(t)
        r_yaw = amp * (1.0 - yaw_bias * 0.2) * np.sin(t)

        aoa = self.pitch_amplitude
        l_pitch = aoa * np.sign(np.cos(t)) + pitch_bias * np.radians(10)
        r_pitch = aoa * np.sign(np.cos(t)) + pitch_bias * np.radians(10)

        l_roll = self.roll_amplitude * np.sin(2 * t) + roll_bias * np.radians(15)
        r_roll = self.roll_amplitude * np.sin(2 * t) - roll_bias * np.radians(15)

        return {
            "l_yaw": l_yaw, "l_pitch": l_pitch, "l_roll": l_roll,
            "r_yaw": r_yaw, "r_pitch": r_pitch, "r_roll": r_roll,
        }


class FlightController:
    """PD-based flight controller with wing animation.

    Applies a vertical force via xfrc_applied to achieve stable hover,
    while animating wing joints for visual realism.
    """

    def __init__(self, sim, fly_name, fly, timestep):
        self.sim = sim
        self.fly_name = fly_name
        self.fly = fly
        self.wings = WingBeatController(timestep)
        self.is_flying = False
        self.target_altitude = 3.0  # mm

        # PD gains (tuned empirically — see audit_sim.py tests)
        self.Kp = 3.0
        self.Kd = 0.08

        # Fly weight (mass * |gravity|)
        total_mass = sum(sim.mj_model.body_mass[1:])
        self.weight = total_mass * abs(sim.mj_model.opt.gravity[2])

        # Thorax body ID for xfrc_applied
        root_seg = fly.root_segment
        self._thorax_id = mj.mj_name2id(
            sim.mj_model, mj.mjtObj.mjOBJ_BODY,
            fly.bodyseg_to_mjcfbody[root_seg].full_identifier
        )

        # Wing actuator ctrl indices
        self._wing_ctrl_map = {}
        for side in ["l", "r"]:
            for axis in ["yaw", "pitch", "roll"]:
                for suffix in ["-position", "-motor", "-velocity", ""]:
                    act_name = f"{fly_name}/c_thorax-{side}_wing-{axis}{suffix}"
                    aid = mj.mj_name2id(
                        sim.mj_model, mj.mjtObj.mjOBJ_ACTUATOR, act_name
                    )
                    if aid >= 0:
                        self._wing_ctrl_map[f"{side}_{axis}"] = aid
                        break

        self._flight_time = 0.0
        self._prev_z = 0.0

    def start_flying(self):
        self.is_flying = True
        self._flight_time = 0.0
        bp = self.sim.get_body_positions(self.fly_name)
        self._prev_z = bp[0, 2]

    def stop_flying(self):
        self.is_flying = False
        # Zero wing actuators
        for aid in self._wing_ctrl_map.values():
            self.sim.mj_data.ctrl[aid] = 0
        # Zero applied forces
        self.sim.mj_data.xfrc_applied[self._thorax_id, :] = 0

    def step(self, move_direction=None, turn=0.0):
        if not self.is_flying:
            return

        dt = self.sim.mj_model.opt.timestep
        self._flight_time += dt

        # Current altitude
        bp = self.sim.get_body_positions(self.fly_name)
        z = bp[0, 2]
        vz = (z - self._prev_z) / max(dt, 1e-8)
        self._prev_z = z

        # Smooth takeoff ramp (0.3s)
        ramp = min(1.0, self._flight_time / 0.3)

        # PD altitude control
        err = self.target_altitude - z
        force_z = (self.weight + self.Kp * err - self.Kd * vz) * ramp
        force_z = max(0, min(force_z, self.weight * 2.5))

        # Horizontal forces for movement
        force_x, force_y = 0.0, 0.0
        pitch_bias, roll_bias, yaw_bias = 0.0, 0.0, turn * 0.5

        if move_direction is not None:
            dx, dy = move_direction
            mag = np.sqrt(dx**2 + dy**2)
            if mag > 0.01:
                # Apply gentle horizontal force toward target
                force_x = dx / mag * self.weight * 0.15 * ramp
                force_y = dy / mag * self.weight * 0.15 * ramp
                pitch_bias = -0.3  # tilt forward visually

                # Yaw toward target
                rot = self.sim.get_body_rotations(self.fly_name)
                q = rot[0]
                fwd_x = 1 - 2 * (q[2]**2 + q[3]**2)
                fwd_y = 2 * (q[1]*q[2] + q[0]*q[3])
                cross = fwd_x * dy - fwd_y * dx
                yaw_bias += np.clip(cross * 0.3, -0.5, 0.5)

        # Apply forces
        self.sim.mj_data.xfrc_applied[self._thorax_id, :] = 0
        self.sim.mj_data.xfrc_applied[self._thorax_id, 0] = force_x
        self.sim.mj_data.xfrc_applied[self._thorax_id, 1] = force_y
        self.sim.mj_data.xfrc_applied[self._thorax_id, 2] = force_z

        # Small restoring torque for stability (prevents tumbling)
        # Damp angular velocity
        thorax_xmat = self.sim.mj_data.xmat[self._thorax_id].reshape(3, 3)
        body_up = thorax_xmat[:, 2]
        world_up = np.array([0, 0, 1])
        tilt = np.cross(body_up, world_up)
        self.sim.mj_data.xfrc_applied[self._thorax_id, 3:] = tilt * 0.5

        # Animate wings
        throttle = np.clip(force_z / self.weight, 0, 1.2)
        wing_angles = self.wings.step(
            throttle=throttle,
            pitch_bias=pitch_bias,
            roll_bias=roll_bias,
            yaw_bias=yaw_bias,
        )
        for key, aid in self._wing_ctrl_map.items():
            self.sim.mj_data.ctrl[aid] = wing_angles[key] * 0.3  # scaled for visual
