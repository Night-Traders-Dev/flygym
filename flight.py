"""
flight.py — Quasi-steady aerodynamic flight model for Drosophila

Implements a simplified blade-element aerodynamic model that computes lift
and drag forces from wing kinematics, and a basic wing beat controller
for hovering, forward flight, and turning.

Drosophila flight parameters (from Dickinson et al., 1999; Fry et al., 2003):
  - Wing beat frequency: ~200 Hz
  - Wing length: ~2.5 mm (but model wing span ~1.35 mm per side)
  - Stroke amplitude: ~140-160 degrees
  - Body mass: ~1 mg
  - Lift coefficient: ~1.5-2.0 during translation

Since MuJoCo has no native aerodynamics, we compute forces analytically
and apply them via xfrc_applied on the thorax body each timestep.
"""

import numpy as np
import mujoco as mj


class AerodynamicsModel:
    """Quasi-steady aerodynamic force calculator for a single fly.

    Uses a simplified blade-element model: lift and drag are computed from
    wing angular velocity, angle of attack, and empirical force coefficients.
    Forces are applied to the thorax body via xfrc_applied.
    """

    def __init__(self, sim, fly_name, fly):
        self.sim = sim
        self.fly_name = fly_name
        self.fly = fly

        # Find body IDs
        root_seg = fly.root_segment
        self._thorax_bodyid = mj.mj_name2id(
            sim.mj_model, mj.mjtObj.mjOBJ_BODY,
            fly.bodyseg_to_mjcfbody[root_seg].full_identifier
        )

        # Wing joint IDs (for reading angular velocity)
        self._wing_joint_names = {
            "l": {
                "yaw": f"{fly_name}/c_thorax-l_wing-yaw",
                "pitch": f"{fly_name}/c_thorax-l_wing-pitch",
                "roll": f"{fly_name}/c_thorax-l_wing-roll",
            },
            "r": {
                "yaw": f"{fly_name}/c_thorax-r_wing-yaw",
                "pitch": f"{fly_name}/c_thorax-r_wing-pitch",
                "roll": f"{fly_name}/c_thorax-r_wing-roll",
            },
        }
        self._wing_dofadrs = {}
        for side, joints in self._wing_joint_names.items():
            self._wing_dofadrs[side] = {}
            for axis, jname in joints.items():
                jid = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_JOINT, jname)
                if jid >= 0:
                    self._wing_dofadrs[side][axis] = sim.mj_model.jnt_dofadr[jid]

        # Aerodynamic parameters (tuned for this model's scale)
        # Model units: mm, mg, ms-like time
        self.air_density = 1.225e-9   # mg/mm³ (real: 1.225 kg/m³)
        self.wing_length = 1.35       # mm (from geom size)
        self.wing_chord = 0.5         # mm (mean chord)
        self.wing_area = self.wing_length * self.wing_chord  # mm²

        # Empirical lift/drag coefficients (Dickinson 1999)
        # CL ≈ 0.225 + 1.58*sin(2.13*alpha - 7.2°)
        # CD ≈ 1.92 - 1.55*cos(2.04*alpha - 9.82°)
        # Simplified: at optimal AoA (~45°), CL ≈ 1.8, CD ≈ 1.2

        # Force scaling factor — calibrated so that realistic wing kinematics
        # produce enough lift to hover (~10 force units needed)
        # F = 0.5 * rho * v² * S * C
        # With wing tip velocity v = omega * R, and omega = 2*pi*200 * stroke_amp
        # This needs a large scaling factor because the model wing is smaller
        # than real Drosophila wings and the air density in mm units is tiny
        self.lift_scale = 800.0   # empirical scaling to match weight
        self.drag_scale = 400.0

    def compute_forces(self, wing_commands=None):
        """Compute aerodynamic forces from wing beat commands.

        Instead of measuring actual joint velocities (which lag behind due to
        actuator bandwidth limits), we compute forces from the COMMANDED wing
        kinematics. This is the standard approach for insect flight simulation
        where wing inertia is negligible compared to aero forces.

        Args:
            wing_commands: dict from WingBeatController.step() with commanded
                           wing angles. If None, returns zero forces.

        Returns (force_xyz, torque_xyz) in world frame to apply to thorax.
        """
        if wing_commands is None:
            return np.zeros(3), np.zeros(3)

        data = self.sim.mj_data
        thorax_xmat = data.xmat[self._thorax_bodyid].reshape(3, 3)
        body_up = thorax_xmat[:, 2]
        body_forward = thorax_xmat[:, 0]
        body_left = thorax_xmat[:, 1]

        total_force = np.zeros(3)
        total_torque = np.zeros(3)

        # Weight to hover: total_mass * |gravity_z| ≈ 10
        # Each wing produces ~5 at throttle=1.0, but averaged over the stroke
        # cycle the mean is lower. Calibrated so throttle=0.5 ≈ hover.
        hover_force_per_wing = 5.0

        for side in ["l", "r"]:
            stroke_angle = wing_commands[f"{side}_yaw"]  # current stroke position
            wing_pitch = wing_commands[f"{side}_pitch"]  # angle of attack
            wing_roll = wing_commands[f"{side}_roll"]

            # Instantaneous stroke velocity from commanded sinusoidal
            # d/dt(A*sin(wt)) = A*w*cos(wt), but we approximate from angle
            # At peak velocity (mid-stroke), force is maximum
            # Use |sin(stroke/amplitude)| as a proxy for force variation
            stroke_amp = self._parent_controller.wings.stroke_amplitude
            if stroke_amp > 0.01:
                normalized_pos = stroke_angle / stroke_amp
                # Force peaks at mid-stroke (where velocity is highest)
                velocity_factor = np.sqrt(1 - np.clip(normalized_pos**2, 0, 0.99))
            else:
                velocity_factor = 0

            # Lift coefficient from angle of attack
            alpha = abs(wing_pitch)
            cl = 0.225 + 1.58 * np.sin(2.13 * alpha - np.radians(7.2))
            cl = max(0, cl)

            # Lift force: proportional to velocity and AoA
            lift_mag = hover_force_per_wing * velocity_factor * cl / 1.5  # normalize by typical CL
            lift_force = body_up * lift_mag

            # Small forward thrust from drag asymmetry during upstroke/downstroke
            thrust = body_forward * lift_mag * 0.05 * np.sign(wing_pitch)

            total_force += lift_force + thrust

            # Torque for turning: asymmetric wing amplitude creates yaw torque
            side_sign = 1.0 if side == "l" else -1.0
            yaw_torque = body_up * lift_mag * wing_roll * side_sign * 0.3
            # Pitch torque from forward/back stroke bias
            pitch_torque = body_left * lift_mag * 0.01 * normalized_pos
            total_torque += yaw_torque + pitch_torque

        return total_force, total_torque

    def apply_forces(self, wing_commands=None):
        """Compute and apply aerodynamic forces to thorax."""
        force, torque = self.compute_forces(wing_commands)
        self.sim.mj_data.xfrc_applied[self._thorax_bodyid, :3] += force
        self.sim.mj_data.xfrc_applied[self._thorax_bodyid, 3:] += torque


class WingBeatController:
    """Generates wing beat patterns for flight.

    Drosophila wing kinematics (simplified):
      - Stroke (yaw): sinusoidal sweep at ~200 Hz, amplitude ~70° (140° total)
      - Pitch (rotation): ~45° angle of attack, flips at stroke reversal
      - Roll (deviation): small, ~10° variation

    Control inputs:
      - throttle: 0-1, scales wing beat amplitude (0=folded, 1=full power)
      - pitch_bias: -1 to 1, pitches the fly forward/backward
      - roll_bias: -1 to 1, rolls left/right for turning
      - yaw_bias: -1 to 1, differential stroke amplitude for yaw turning
    """

    def __init__(self, timestep, freq=200.0):
        self.timestep = timestep
        self.freq = freq
        self.phase = 0.0

        # Kinematic amplitudes (radians)
        self.stroke_amplitude = np.radians(70)   # ±70° = 140° total
        self.pitch_amplitude = np.radians(45)    # angle of attack
        self.roll_amplitude = np.radians(8)      # deviation

    def step(self, throttle=1.0, pitch_bias=0.0, roll_bias=0.0, yaw_bias=0.0):
        """Compute wing joint angles for this timestep.

        Returns dict with keys 'l_yaw', 'l_pitch', 'l_roll',
        'r_yaw', 'r_pitch', 'r_roll' (angles in radians).
        """
        self.phase += 2 * np.pi * self.freq * self.timestep
        t = self.phase

        stroke_amp = self.stroke_amplitude * np.clip(throttle, 0, 1.2)

        # Stroke (yaw) — sinusoidal sweep
        l_stroke = stroke_amp * (1.0 + yaw_bias * 0.2) * np.sin(t)
        r_stroke = stroke_amp * (1.0 - yaw_bias * 0.2) * np.sin(t)

        # Wing pitch — flips near stroke reversal (approximated with cos)
        # At mid-stroke: ~45° AoA. At reversal: rapid rotation
        aoa = self.pitch_amplitude
        l_pitch = aoa * np.sign(np.cos(t)) + pitch_bias * np.radians(10)
        r_pitch = aoa * np.sign(np.cos(t)) + pitch_bias * np.radians(10)

        # Roll (deviation) — small oscillation at 2x frequency
        l_roll = self.roll_amplitude * np.sin(2 * t) + roll_bias * np.radians(15)
        r_roll = self.roll_amplitude * np.sin(2 * t) - roll_bias * np.radians(15)

        return {
            "l_yaw": l_stroke, "l_pitch": l_pitch, "l_roll": l_roll,
            "r_yaw": r_stroke, "r_pitch": r_pitch, "r_roll": r_roll,
        }


class FlightController:
    """High-level flight controller that combines aerodynamics and wing beats.

    Manages takeoff, hover, directional flight, and landing.
    """

    def __init__(self, sim, fly_name, fly, timestep):
        self.sim = sim
        self.fly_name = fly_name
        self.fly = fly
        self.aero = AerodynamicsModel(sim, fly_name, fly)
        self.aero._parent_controller = self  # back-reference for wing params
        self.wings = WingBeatController(timestep)
        self.is_flying = False
        self.target_altitude = 3.0  # mm above ground
        self._flight_time = 0.0

        # Find wing actuator indices
        self._wing_act_indices = {}
        all_act_dofs = fly.get_actuated_jointdofs_order(
            mj.mjtObj.mjOBJ_ACTUATOR  # placeholder, we'll map manually
        ) if hasattr(fly, '_actuator_order_cache') else None

        # Map wing DOF names to actuator ctrl indices
        # Actuator names follow pattern: {fly_name}/c_thorax-{side}_wing-{axis}-position
        self._wing_ctrl_map = {}
        for side in ["l", "r"]:
            for axis in ["yaw", "pitch", "roll"]:
                # Try common naming patterns
                for suffix in ["-position", "-motor", "-velocity", ""]:
                    act_name = f"{fly_name}/c_thorax-{side}_wing-{axis}{suffix}"
                    aid = mj.mj_name2id(
                        sim.mj_model, mj.mjtObj.mjOBJ_ACTUATOR, act_name
                    )
                    if aid >= 0:
                        self._wing_ctrl_map[f"{side}_{axis}"] = aid
                        break

    def start_flying(self):
        self.is_flying = True
        self._flight_time = 0.0

    def stop_flying(self):
        self.is_flying = False
        # Zero out wing actuators
        for key, aid in self._wing_ctrl_map.items():
            self.sim.mj_data.ctrl[aid] = 0

    def step(self, move_direction=None, turn=0.0):
        """Step the flight controller.

        Args:
            move_direction: (dx, dy) desired movement direction, or None for hover
            turn: -1 to 1 yaw turning input
        """
        if not self.is_flying:
            return

        self._flight_time += self.sim.mj_model.opt.timestep

        # Get current state
        bp = self.sim.get_body_positions(self.fly_name)
        altitude = bp[0, 2]

        # Altitude PD-controller
        alt_error = self.target_altitude - altitude
        # Estimate vertical velocity from body state
        body_id = self.aero._thorax_bodyid
        vz = self.sim.mj_data.cvel[body_id, 5]  # linear z velocity
        throttle = np.clip(0.5 + alt_error * 0.15 - vz * 0.05, 0.2, 1.0)

        # Direction control
        pitch_bias = 0.0
        roll_bias = 0.0
        yaw_bias = turn * 0.5

        if move_direction is not None:
            dx, dy = move_direction
            mag = np.sqrt(dx**2 + dy**2)
            if mag > 0.01:
                # Get fly heading
                rot = self.sim.get_body_rotations(self.fly_name)
                q = rot[0]
                fwd_x = 1 - 2 * (q[2]**2 + q[3]**2)
                fwd_y = 2 * (q[1]*q[2] + q[0]*q[3])
                # Pitch forward to move
                pitch_bias = -np.clip(mag * 5, 0, 0.5)
                # Compute yaw correction
                cross = fwd_x * dy - fwd_y * dx
                yaw_bias += np.clip(cross * 0.3, -0.5, 0.5)

        # Generate wing kinematics
        wing_angles = self.wings.step(
            throttle=throttle,
            pitch_bias=pitch_bias,
            roll_bias=roll_bias,
            yaw_bias=yaw_bias,
        )

        # Apply wing angles to actuators
        for key, aid in self._wing_ctrl_map.items():
            self.sim.mj_data.ctrl[aid] = wing_angles[key]

        # Apply aerodynamic forces (from commanded kinematics, not measured)
        self.aero.apply_forces(wing_angles)
