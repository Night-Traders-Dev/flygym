#!/usr/bin/env python3
"""
fly_autonomous.py - Multi-fly simulation in a multi-biome world

5 biome types: forest floor, meadow, wetland, sandy arid, fruit garden.
Each has unique temperature, humidity, wind, friction, and food density.
Flies walk using real recorded kinematics and steer toward food.

Launch:
    .venv/bin/python fly_autonomous.py
"""

import os
import sys
import time

os.environ["MUJOCO_GL"] = "glfw"
os.environ.pop("WAYLAND_DISPLAY", None)
os.environ["LIBDECOR_PLUGIN_DIR"] = "/tmp/libdecor_cairo_only"

import numpy as np
import mujoco as mj
import mujoco.viewer as mjviewer

from flygym.anatomy import (
    Skeleton, JointPreset, AxisOrder, ActuatedDOFPreset,
)
from flygym.compose import Fly, ActuatorType, KinematicPosePreset
from flygym.utils.math import Rotation3D
from flygym import Simulation
from flygym_demo.spotlight_data import MotionSnippet

from biome import (
    BiomeWorld, BiomeParams,
    FOREST_FLOOR, MEADOW, WETLAND, SANDY_ARID, FRUIT_GARDEN,
)
from biome_effects import BiomeEffectsEngine
from fly_vitals import VitalsManager
from flight import FlightController

MAX_FOOD = 15


# ---------------------------------------------------------------------------
# Walking controller (real recorded kinematics)
# ---------------------------------------------------------------------------
class WalkingController:
    def __init__(self, fly, sim_timestep):
        snippet = MotionSnippet()
        dof_order = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        self.joint_angles = snippet.get_joint_angles(
            output_timestep=sim_timestep, output_dof_order=dof_order,
        )
        self.n_steps = self.joint_angles.shape[0]
        self.n_dofs = self.joint_angles.shape[1]
        self.dpl = self.n_dofs // 6
        self.idx = 0
        self._skip_counter = 0
        self._speed_factor = 1.0

    def set_speed_factor(self, factor: float):
        self._speed_factor = max(0.2, min(factor, 2.0))

    def step(self, turn_bias=0.0):
        # Speed factor: skip frames to slow down, or double-step to speed up
        self._skip_counter += self._speed_factor
        if self._skip_counter < 1.0:
            # Don't advance — repeat last frame (slowed down)
            pass
        else:
            steps = int(self._skip_counter)
            self.idx += steps
            self._skip_counter -= steps

        angles = self.joint_angles[self.idx % self.n_steps].copy()
        for leg_idx in range(6):
            b = leg_idx * self.dpl
            if leg_idx < 3:
                angles[b + 1] *= (1.0 + turn_bias * 0.3)
            else:
                angles[b + 1] *= (1.0 - turn_bias * 0.3)
        return angles


# ---------------------------------------------------------------------------
# Food manager
# ---------------------------------------------------------------------------
class FoodManager:
    def __init__(self, mj_model, mj_data, effects_engine, spawn_range=None):
        self.mj_model, self.mj_data = mj_model, mj_data
        self.effects = effects_engine
        self.rng = np.random.RandomState(99)
        self.spawn_range = spawn_range
        self.food_ids = []
        for i in range(MAX_FOOD):
            bid = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, f"food_{i}")
            self.food_ids.append(bid)
        self.active = np.zeros(MAX_FOOD, dtype=bool)
        self.positions = np.zeros((MAX_FOOD, 2))
        for _ in range(6):
            self._spawn()

    def _spawn(self):
        slots = np.where(~self.active)[0]
        if len(slots) == 0:
            return
        s = slots[self.rng.randint(len(slots))]
        # Weighted spawn: try random positions, accept based on food_density
        for _ in range(20):
            x = self.rng.uniform(-self.spawn_range, self.spawn_range)
            y = self.rng.uniform(-self.spawn_range, self.spawn_range)
            weight = self.effects.get_food_spawn_weight(x, y)
            if self.rng.random() < weight / 3.0:
                break
        self.positions[s] = [x, y]
        self.active[s] = True
        mid = self.mj_model.body_mocapid[self.food_ids[s]]
        if mid >= 0:
            self.mj_data.mocap_pos[mid] = [x, y, 0.3]

    def _hide(self, s):
        self.active[s] = False
        mid = self.mj_model.body_mocapid[self.food_ids[s]]
        if mid >= 0:
            self.mj_data.mocap_pos[mid] = [0, 0, -10]

    def update(self, fly_positions_dict):
        """Update food. Returns set of fly names that ate food this tick."""
        ate = set()
        for s in range(MAX_FOOD):
            if not self.active[s]:
                continue
            for fly_name, fp in fly_positions_dict.items():
                if np.linalg.norm(fp[:2] - self.positions[s]) < 1.5:
                    self._hide(s)
                    ate.add(fly_name)
                    break
        if self.active.sum() < 5:
            self._spawn()
            self._spawn()
        elif self.active.sum() < 8 and self.rng.random() < 0.3:
            self._spawn()
        return ate

    def get_active_positions(self):
        return self.positions[self.active]


# ---------------------------------------------------------------------------
# Fly factory
# ---------------------------------------------------------------------------
def make_fly(name):
    fly = Fly(name=name)
    skeleton = Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL,
        joint_preset=JointPreset.ALL_BIOLOGICAL,  # includes wing joints
    )
    fly.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)

    # Leg actuators (position control for walking)
    leg_dofs = skeleton.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY
    )
    fly.add_actuators(
        leg_dofs, ActuatorType.POSITION,
        kp=150.0, neutral_input=KinematicPosePreset.NEUTRAL,
        ctrlrange=(-3.14, 3.14),
    )

    # Wing actuators (motor control for flight — direct torque)
    wing_dofs = [d for d in skeleton.iter_jointdofs() if "wing" in d.name]
    fly.add_actuators(
        wing_dofs, ActuatorType.POSITION,
        kp=20.0,
        ctrlrange=(-3.14, 3.14),
    )

    fly.add_leg_adhesion()
    fly.colorize()
    return fly


# ---------------------------------------------------------------------------
# Build biome grid
# ---------------------------------------------------------------------------
def make_default_biome_grid():
    """5x5 grid with all 5 biome types arranged naturally."""
    F = FOREST_FLOOR
    M = MEADOW
    W = WETLAND
    S = SANDY_ARID
    G = FRUIT_GARDEN
    return [
        [S, S, M, M, G],
        [S, M, M, G, G],
        [M, M, F, G, W],
        [M, F, F, W, W],
        [F, F, W, W, W],
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    timestep = 1e-4
    num_flies = 3
    realtime_factor = 1.0  # true real-time: 1 wall-second = 1 sim-second
    zone_size = 20.0  # mm per biome zone

    biome_grid = make_default_biome_grid()
    world = BiomeWorld(biome_grid, zone_size=zone_size, n_food=MAX_FOOD)

    # Multi-fly keyframe patch
    import flygym.compose.world as _wm
    def _fix(self):
        mm, _ = self.compile()
        nq, nc = np.zeros(mm.nq), np.zeros(mm.nu)
        aj = {j.full_identifier: j for j in self.mjcf_root.find_all("joint")}
        for jn, ns in self.world_dof_neutral_states.items():
            je = aj.get(jn)
            if not je: continue
            jt = "free" if je.tag == "freejoint" else je.type
            jid = mj.mj_name2id(mm, mj.mjtObj.mjOBJ_JOINT, je.full_identifier)
            a = mm.jnt_qposadr[jid]
            nq[a:a + _wm._STATE_DIM_BY_JOINT_TYPE[jt]] = ns
        for fn, f in self.fly_lookup.items():
            q = f._get_neutral_qpos(mm); nq[q.nonzero()] = q[q.nonzero()]
            c = f._get_neutral_ctrl(mm); nc[c.nonzero()] = c[c.nonzero()]
        self._neutral_keyframe.qpos = nq
        self._neutral_keyframe.ctrl = nc
    _wm.BaseWorld._rebuild_neutral_keyframe = _fix

    # Spawn flies in the center (forest floor region)
    spawns = [
        ((0, 0, 0.7), Rotation3D("quat", (1, 0, 0, 0))),
        ((3, -2, 0.7), Rotation3D("quat", (0.924, 0, 0, 0.383))),
        ((-2, 3, 0.7), Rotation3D("quat", (0.924, 0, 0, -0.383))),
    ]

    flies = []
    for i, (pos, rot) in enumerate(spawns):
        fly = make_fly(f"fly_{i}")
        world.add_fly(fly, pos, rot)
        flies.append(fly)

    sim = Simulation(world)
    sim.reset()
    sim.warmup(duration_s=0.05)

    world.upload_textures(sim.mj_model)
    world.apply_atmosphere(sim.mj_model)

    for fly in flies:
        sim.set_leg_adhesion_states(fly.name, np.ones(6, dtype=bool))

    # Environmental effects
    effects = BiomeEffectsEngine(world, sim)

    # Walking controllers
    controllers = {fly.name: WalkingController(fly, timestep) for fly in flies}
    turn_biases = {fly.name: 0.0 for fly in flies}
    rng = np.random.RandomState(123)

    # Food
    spawn_range = (world.ncols * zone_size) / 2 - zone_size / 2
    food_mgr = FoodManager(sim.mj_model, sim.mj_data, effects, spawn_range=spawn_range)

    # Vitals
    vitals_mgr = VitalsManager([fly.name for fly in flies])
    flies_that_ate = set()

    # Flight controllers
    flight_ctrls = {}
    for fly in flies:
        flight_ctrls[fly.name] = FlightController(sim, fly.name, fly, timestep)

    # Viewer
    viewer = mjviewer.launch_passive(
        sim.mj_model, sim.mj_data,
        show_left_ui=True, show_right_ui=True,
    )

    for i in range(sim.mj_model.ntex):
        name = mj.mj_id2name(sim.mj_model, mj.mjtObj.mjOBJ_TEXTURE, i)
        if name and name.startswith("tex_"):
            viewer.update_texture(i)

    viewer.cam.type = mj.mjtCamera.mjCAMERA_FREE
    viewer.cam.distance = 25.0
    viewer.cam.elevation = -35.0
    viewer.cam.azimuth = 135.0
    viewer.cam.lookat[:] = [0, 0, 0]

    grid_w = world.ncols * zone_size
    grid_h = world.nrows * zone_size
    print("=" * 70)
    print("  NeuroMechFly v2 — Multi-Biome World")
    print(f"  {world.nrows}x{world.ncols} biome grid ({grid_w:.0f}x{grid_h:.0f}mm)")
    print(f"  {num_flies} flies | {realtime_factor}x speed")
    print(f"  Biomes: forest, meadow, wetland, sand, fruit garden")
    print("  Close window or Ctrl+C to stop.")
    print("=" * 70)

    step_count = 0
    turn_update_steps = int(0.5 / timestep)
    food_update_steps = int(1.0 / timestep)
    effects_update_steps = int(0.1 / timestep)
    sync_interval = 500
    metrics_interval = int(2.0 / timestep)
    wall_start = time.perf_counter()

    try:
        while viewer.is_running():
            # --- Update biome effects ---
            if step_count % effects_update_steps == 0:
                fly_pos = {}
                for fly in flies:
                    bp = sim.get_body_positions(fly.name)
                    fly_pos[fly.name] = bp[0]
                effects.update_biomes(fly_pos)
                effects.clear_forces()
                effects.apply_wind()

                # Apply temperature → speed and humidity → adhesion
                for fly in flies:
                    sf = effects.get_speed_factor(fly.name)
                    controllers[fly.name].set_speed_factor(sf)
                    am = effects.get_adhesion_modifier(fly.name)
                    sim.set_leg_adhesion_states(
                        fly.name, np.ones(6, dtype=bool) * am
                    )

            # --- Navigation + flight/walk decision ---
            if step_count % turn_update_steps == 0:
                food_pos = food_mgr.get_active_positions()
                for fly in flies:
                    bp = sim.get_body_positions(fly.name)
                    xy = bp[0, :2]
                    v = vitals_mgr.get(fly.name)
                    fc = flight_ctrls[fly.name]

                    # Compute direction to nearest food
                    food_dir = None
                    food_dist = 999
                    if len(food_pos) > 0:
                        dists = np.linalg.norm(food_pos - xy, axis=1)
                        food_dist = dists.min()
                        to_food = food_pos[np.argmin(dists)] - xy
                        food_dir = to_food / (np.linalg.norm(to_food) + 1e-8)

                        rot = sim.get_body_rotations(fly.name)
                        q = rot[0]
                        fx = 1 - 2 * (q[2]**2 + q[3]**2)
                        fy = 2 * (q[1]*q[2] + q[0]*q[3])
                        cross = fx * to_food[1] - fy * to_food[0]
                        turn_biases[fly.name] = np.clip(cross * 0.15, -0.8, 0.8)
                    else:
                        turn_biases[fly.name] = rng.uniform(-0.3, 0.3)

                    # Flight decision: fly if food is far and energy is adequate
                    should_fly = (
                        food_dist > 8.0
                        and v.energy > 30
                        and v.hunger < 50
                        and v.alive
                    )

                    if should_fly and not fc.is_flying:
                        fc.start_flying()
                    elif (not should_fly or food_dist < 3.0) and fc.is_flying:
                        fc.stop_flying()

            # --- Walk or fly ---
            for fly in flies:
                fc = flight_ctrls[fly.name]
                if fc.is_flying:
                    # Flight mode: wing controller + aerodynamics
                    move_dir = None
                    food_pos = food_mgr.get_active_positions()
                    if len(food_pos) > 0:
                        bp = sim.get_body_positions(fly.name)
                        dists = np.linalg.norm(food_pos - bp[0, :2], axis=1)
                        to_food = food_pos[np.argmin(dists)] - bp[0, :2]
                        move_dir = to_food * 0.01  # gentle direction bias
                    fc.step(move_direction=move_dir, turn=turn_biases[fly.name])
                    # Still send walking angles to keep legs in a neutral pose
                    angles = controllers[fly.name].step(0)
                    sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, angles)
                else:
                    # Walking mode
                    angles = controllers[fly.name].step(turn_biases[fly.name])
                    sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, angles)

            sim.step()
            step_count += 1

            # --- Food + Vitals ---
            if step_count % food_update_steps == 0:
                fps_dict = {f.name: sim.get_body_positions(f.name)[0] for f in flies}
                flies_that_ate = food_mgr.update(fps_dict)

                # Update vitals for all flies
                vitals_data = {}
                for fly in flies:
                    bp = sim.get_body_positions(fly.name)
                    biome = effects.get_current_biome(fly.name)
                    ca, _, *__ = sim.get_ground_contact_info(fly.name)
                    vitals_data[fly.name] = {
                        "pos": bp[0],
                        "biome_temp": biome.temperature if biome else 22.0,
                        "biome_humidity": biome.humidity if biome else 0.5,
                        "ate_food": fly.name in flies_that_ate,
                        "is_walking": ca.sum() >= 2,
                    }
                vitals_mgr.update_all(food_update_steps * timestep, vitals_data)

            # --- Viewer sync + pacing ---
            if step_count % sync_interval == 0:
                viewer.sync()
                sim_t = step_count * timestep
                wall_t = time.perf_counter() - wall_start
                target = sim_t / realtime_factor
                if wall_t < target:
                    time.sleep(target - wall_t)

            # --- Metrics ---
            if step_count % metrics_interval == 0:
                sim_t = step_count * timestep
                fp = food_mgr.get_active_positions()
                n_alive = sum(1 for f in flies if vitals_mgr.get(f.name).alive)
                print(f"\n{'='*65}")
                print(f"  t={sim_t:.0f}s | food={len(fp)} | alive={n_alive}/{num_flies}")
                print(f"{'='*65}")
                for fly in flies:
                    bp = sim.get_body_positions(fly.name)
                    v = vitals_mgr.get(fly.name)
                    biome_info = effects.get_biome_summary(fly.name)
                    fc = flight_ctrls[fly.name]
                    mode = "FLYING" if fc.is_flying else "WALKING"
                    print(f"\n  {fly.name} @ ({bp[0,0]:+6.1f},{bp[0,1]:+6.1f},z={bp[0,2]:.1f}) [{biome_info}] {mode}")
                    print(v.get_status_bar())
                    print(f"  food={v.food_eaten} dist={v.distance_traveled:.1f}mm")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        viewer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
