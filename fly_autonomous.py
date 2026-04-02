#!/usr/bin/env python3
"""
fly_autonomous.py - Watch multiple NeuroMechFly v2 flies on a forest floor

Multiple flies walk autonomously using CPG-driven leg control with food-seeking
behavior. Food spawns at random locations periodically. Metrics are printed live.

Launch:
    .venv/bin/python fly_autonomous.py
"""

import os
import sys
import time

os.environ["MUJOCO_GL"] = "glfw"
os.environ.pop("WAYLAND_DISPLAY", None)
# Use cairo libdecor plugin (not broken GTK one) for proper title bar on Wayland
os.environ["LIBDECOR_PLUGIN_DIR"] = "/tmp/libdecor_cairo_only"

import numpy as np
import mujoco as mj
import mujoco.viewer as mjviewer
import dm_control.mjcf as mjcf

from flygym.anatomy import (
    Skeleton, JointPreset, AxisOrder, ActuatedDOFPreset, ContactBodiesPreset,
)
from flygym.compose import Fly, ActuatorType, FlatGroundWorld, KinematicPosePreset
from flygym.compose.physics import ContactParams
from flygym.utils.math import Rotation3D
from flygym import Simulation

MAX_FOOD = 12  # pre-allocated food slot count


# ---------------------------------------------------------------------------
# Forest-floor world with relocatable food
# ---------------------------------------------------------------------------
class ForestFloorWorld(FlatGroundWorld):
    """FlatGroundWorld with forest visuals and a pool of movable food markers."""

    def __init__(self, food_radius=0.4, name="forest_floor", half_size=1000):
        super().__init__(name=name, half_size=half_size)
        self.food_radius = food_radius

        # Forest floor texture
        forest_tex = self.mjcf_root.asset.add(
            "texture", name="forest_floor_tex", type="2d", builtin="flat",
            width=512, height=512, rgb1=(0.28, 0.22, 0.14), rgb2=(0.18, 0.14, 0.08),
        )
        forest_mat = self.mjcf_root.asset.add(
            "material", name="forest_floor_mat", texture=forest_tex,
            texrepeat=(8, 8), reflectance=0.02, specular=0.05,
        )
        self.ground_geom.material = forest_mat
        self._forest_tex_name = "forest_floor_tex"

        # Skybox
        self.mjcf_root.asset.add(
            "texture", name="forest_sky", type="skybox", builtin="gradient",
            rgb1=(0.55, 0.65, 0.45), rgb2=(0.25, 0.35, 0.2), width=512, height=512,
        )

        # Warm lighting
        self.mjcf_root.worldbody.add(
            "light", name="sun", mode="fixed", directional=True, castshadow=True,
            pos=(0, 0, 100), dir=(0.3, 0.2, -1),
            ambient=(0.2, 0.18, 0.12), diffuse=(0.75, 0.65, 0.45),
            specular=(0.25, 0.22, 0.15),
        )
        self.mjcf_root.worldbody.add(
            "light", name="canopy_fill", mode="fixed", directional=True,
            castshadow=False, pos=(0, 0, 80), dir=(-0.2, -0.3, -1),
            ambient=(0.05, 0.08, 0.04), diffuse=(0.12, 0.2, 0.08),
            specular=(0.03, 0.05, 0.02),
        )

        # Pre-allocate food slots as mocap bodies (hidden far away initially)
        self._food_body_names = []
        for i in range(MAX_FOOD):
            body = self.mjcf_root.worldbody.add(
                "body", name=f"food_{i}", pos=(0, 0, -10), mocap=True,
            )
            body.add(
                "geom", type="sphere", size=(food_radius,),
                rgba=(0.85, 0.15, 0.1, 0.9), contype=0, conaffinity=0,
            )
            body.add(
                "geom", type="cylinder", size=(food_radius * 2.5, 0.01),
                pos=(0, 0, -food_radius + 0.01),
                rgba=(0.9, 0.3, 0.1, 0.25), contype=0, conaffinity=0,
            )
            self._food_body_names.append(f"food_{i}")

    def _set_ground_contact(self, fly, bodysegs_with_ground_contact, ground_contact_params):
        for body_segment in bodysegs_with_ground_contact:
            body_geom = fly.mjcf_root.find("geom", f"{body_segment.name}")
            self.mjcf_root.contact.add(
                "pair", geom1=body_geom, geom2=self.ground_geom,
                name=f"{fly.name}_{body_segment.name}-ground",
                friction=ground_contact_params.get_friction_tuple(),
                solref=ground_contact_params.get_solref_tuple(),
                solimp=ground_contact_params.get_solimp_tuple(),
                margin=ground_contact_params.margin,
            )

    def _add_ground_contact_sensors(self, fly, bodysegs_with_ground_contact):
        from collections import defaultdict
        from flygym.anatomy import LEG_LINKS

        if self.legpos_to_groundcontactsensors_by_fly is None:
            self.legpos_to_groundcontactsensors_by_fly = defaultdict(dict)

        contact_geoms_by_leg = defaultdict(list)
        for bodyseg in bodysegs_with_ground_contact:
            if bodyseg.is_leg():
                contact_geoms_by_leg[bodyseg.pos].append(bodyseg)
        for leg, contact_geoms in contact_geoms_by_leg.items():
            sorted_segs = sorted(contact_geoms, key=lambda s: LEG_LINKS.index(s.link))
            subtree_rootseg = sorted_segs[0]
            subtree_rootseg_body = fly.bodyseg_to_mjcfbody[subtree_rootseg]
            sensor = self.mjcf_root.sensor.add(
                "contact", subtree1=subtree_rootseg_body, geom2=self.ground_geom,
                num=1, reduce="netforce",
                data="found force torque pos normal tangent",
                name=f"{fly.name}_ground_contact_{leg}_leg",
            )
            self.legpos_to_groundcontactsensors_by_fly[fly.name][leg] = sensor

    def upload_texture(self, mj_model):
        tex_img = _generate_forest_floor_texture(512, 512)
        for i in range(mj_model.ntex):
            name = mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_TEXTURE, i)
            if name == self._forest_tex_name:
                h, w = mj_model.tex_height[i], mj_model.tex_width[i]
                adr = mj_model.tex_adr[i]
                flat = tex_img[:h, :w, :].flatten()
                mj_model.tex_data[adr : adr + len(flat)] = flat
                break

    def apply_atmosphere(self, mj_model):
        mj_model.vis.rgba.fog[:] = [0.25, 0.32, 0.2, 1.0]
        mj_model.vis.map.fogstart = 20.0
        mj_model.vis.map.fogend = 80.0


# ---------------------------------------------------------------------------
# Food manager — spawns/despawns food by moving mocap bodies
# ---------------------------------------------------------------------------
class FoodManager:
    def __init__(self, mj_model, mj_data, max_food, food_radius,
                 spawn_range=30.0, spawn_interval_s=3.0, despawn_dist=2.0):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.max_food = max_food
        self.food_radius = food_radius
        self.spawn_range = spawn_range
        self.spawn_interval_s = spawn_interval_s
        self.despawn_dist = despawn_dist
        self.rng = np.random.RandomState(99)

        # Find mocap body IDs
        self.food_body_ids = []
        for i in range(max_food):
            bid = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, f"food_{i}")
            self.food_body_ids.append(bid)

        # Track which slots are active and their positions
        self.active = np.zeros(max_food, dtype=bool)
        self.positions = np.zeros((max_food, 2))  # xy

        # Start with a few food items
        for _ in range(4):
            self._spawn_one()

    def _spawn_one(self):
        """Spawn food at a random location in an inactive slot."""
        inactive = np.where(~self.active)[0]
        if len(inactive) == 0:
            return
        slot = inactive[self.rng.randint(len(inactive))]
        x = self.rng.uniform(-self.spawn_range, self.spawn_range)
        y = self.rng.uniform(-self.spawn_range, self.spawn_range)
        self.positions[slot] = [x, y]
        self.active[slot] = True
        # Move the mocap body to the new position
        mocap_idx = self.mj_model.body_mocapid[self.food_body_ids[slot]]
        if mocap_idx >= 0:
            self.mj_data.mocap_pos[mocap_idx] = [x, y, self.food_radius]

    def _hide(self, slot):
        """Move food far away (despawn)."""
        self.active[slot] = False
        mocap_idx = self.mj_model.body_mocapid[self.food_body_ids[slot]]
        if mocap_idx >= 0:
            self.mj_data.mocap_pos[mocap_idx] = [0, 0, -10]

    def update(self, sim_time, fly_positions):
        """Called periodically: despawn eaten food, spawn new food."""
        # Check if any fly is close enough to eat food
        for slot in range(self.max_food):
            if not self.active[slot]:
                continue
            for fpos in fly_positions:
                dist = np.linalg.norm(fpos[:2] - self.positions[slot])
                if dist < self.despawn_dist:
                    self._hide(slot)
                    break

        # Spawn new food periodically (keep ~4-6 active)
        n_active = self.active.sum()
        if n_active < 4:
            self._spawn_one()
            self._spawn_one()
        elif n_active < 6 and self.rng.random() < 0.3:
            self._spawn_one()

    def get_active_positions(self):
        return self.positions[self.active]


def _generate_forest_floor_texture(width=512, height=512):
    rng = np.random.RandomState(42)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    xs = np.arange(width) / width
    ys = np.arange(height) / height
    xx, yy = np.meshgrid(xs, ys)
    noise = (
        np.sin(xx * 13.7 + yy * 9.3) * 0.3
        + np.sin(xx * 27.1 + yy * 19.7) * 0.15
        + np.sin(xx * 53.3 + yy * 41.1) * 0.08
    )
    base_noise = rng.normal(0, 6, (height, width))
    img[:, :, 0] = np.clip(62 + noise * 40 + base_noise, 30, 100).astype(np.uint8)
    img[:, :, 1] = np.clip(45 + noise * 30 + base_noise * 0.8, 20, 75).astype(np.uint8)
    img[:, :, 2] = np.clip(28 + noise * 15 + base_noise * 0.5, 10, 50).astype(np.uint8)

    for _ in range(200):
        cx, cy = rng.randint(0, width), rng.randint(0, height)
        r = rng.randint(3, 12)
        yy2, xx2 = np.ogrid[-cy : height - cy, -cx : width - cx]
        mask = xx2**2 + yy2**2 <= r**2
        img[mask] = (img[mask] * rng.uniform(0.5, 0.8)).astype(np.uint8)

    leaf_colors = [(85, 60, 20), (100, 80, 25), (55, 65, 30), (70, 30, 20), (40, 50, 25)]
    for _ in range(120):
        cx, cy = rng.randint(0, width), rng.randint(0, height)
        rx, ry = rng.randint(2, 8), rng.randint(4, 14)
        angle = rng.uniform(0, np.pi)
        color = np.array(leaf_colors[rng.randint(len(leaf_colors))])
        yy2, xx2 = np.ogrid[-cy : height - cy, -cx : width - cx]
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        xr = cos_a * xx2 + sin_a * yy2
        yr = -sin_a * xx2 + cos_a * yy2
        mask = (xr / max(rx, 1)) ** 2 + (yr / max(ry, 1)) ** 2 <= 1
        blend = rng.uniform(0.4, 0.8)
        img[mask] = (img[mask] * (1 - blend) + color * blend).astype(np.uint8)

    for _ in range(400):
        x, y = rng.randint(0, width), rng.randint(0, height)
        b = rng.randint(90, 140)
        img[y, x] = [b, int(b * 0.85), int(b * 0.65)]

    for _ in range(25):
        x0, y0 = rng.randint(0, width), rng.randint(0, height)
        angle = rng.uniform(0, np.pi)
        for t in range(rng.randint(10, 35)):
            px = int(x0 + t * np.cos(angle))
            py = int(y0 + t * np.sin(angle))
            if 0 <= px < width and 0 <= py < height:
                img[py, px] = (img[py, px] * 0.4).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Gentle CPG for stable walking
# ---------------------------------------------------------------------------
class SimpleCPG:
    """Minimal CPG for tripod gait.

    Per-leg DOF layout (7 DOFs each):
      [0] coxa-yaw       (turning left/right)
      [1] coxa-pitch      (forward/back swing)
      [2] coxa-roll
      [3] femur-pitch     (lift/lower leg)
      [4] femur-roll
      [5] tibia-pitch     (extend/retract)
      [6] tarsus1-pitch
    """

    def __init__(self, n_actuated_dofs, timestep, freq=12.0):
        self.n_dofs = n_actuated_dofs
        self.timestep = timestep
        self.freq = freq
        self.phase = 0.0
        self.dpl = n_actuated_dofs // 6  # 7 DOFs per leg

    def step(self, neutral_ctrl, turn_bias=0.0):
        self.phase += 2 * np.pi * self.freq * self.timestep
        offsets = np.zeros(self.n_dofs)

        for leg_idx in range(6):
            leg_phase = self.phase + (0 if leg_idx % 2 == 0 else np.pi)
            b = leg_idx * self.dpl
            sin_p = np.sin(leg_phase)
            swing_up = max(0, sin_p)

            # [1] Coxa PITCH — forward/back swing
            swing = 0.5 * sin_p
            if leg_idx < 3:
                swing *= (1.0 + turn_bias * 0.4)
            else:
                swing *= (1.0 - turn_bias * 0.4)
            offsets[b + 1] = swing

            # [3] Femur PITCH — lift leg during swing
            offsets[b + 3] = -0.6 * swing_up

            # [5] Tibia PITCH — extend during swing
            offsets[b + 5] = 0.3 * swing_up

            # [0] Coxa YAW — steering
            offsets[b + 0] = turn_bias * 0.08

        return neutral_ctrl + offsets


# ---------------------------------------------------------------------------
# Fly factory
# ---------------------------------------------------------------------------
def make_fly(name):
    fly = Fly(name=name)
    skeleton = Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL,
        joint_preset=JointPreset.LEGS_ONLY,
    )
    neutral_pose = KinematicPosePreset.NEUTRAL
    fly.add_joints(skeleton, neutral_pose=neutral_pose)
    actuated_dofs = skeleton.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY
    )
    fly.add_actuators(
        actuated_dofs, ActuatorType.POSITION,
        kp=50.0, neutral_input=neutral_pose, ctrlrange=(-3.14, 3.14),
    )
    fly.add_leg_adhesion()
    fly.colorize()
    fly.add_tracking_camera(name=f"{name}_cam")
    return fly


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    timestep = 1e-4
    num_flies = 3
    # Physics runs at 10x real-time so fly movement is visible
    # (flies are ~2mm long and walk ~1.5mm/s — invisible at 1:1)
    # 1 wall-second = 10 simulated seconds
    realtime_factor = 10.0

    # Quaternions for rotation around Z axis only (w, x, y, z):
    # (1,0,0,0) = forward, (cos(a/2),0,0,sin(a/2)) = yaw by angle a
    spawn_configs = [
        ((0, 0, 0.7), Rotation3D("quat", (1, 0, 0, 0))),             # facing +x
        ((8, -6, 0.7), Rotation3D("quat", (0.924, 0, 0, 0.383))),    # ~45° yaw
        ((-5, 4, 0.7), Rotation3D("quat", (0.707, 0, 0, -0.707))),   # ~-90° yaw
    ]

    world = ForestFloorWorld()

    # Monkey-patch multi-fly neutral keyframe conflict
    import flygym.compose.world as _world_mod
    def _patched_rebuild(self):
        mj_model, _ = self.compile()
        neutral_qpos = np.zeros(mj_model.nq)
        neutral_ctrl = np.zeros(mj_model.nu)
        all_world_joints = {
            j.full_identifier: j for j in self.mjcf_root.find_all("joint")
        }
        for joint_name, neutral_state in self.world_dof_neutral_states.items():
            joint_element = all_world_joints.get(joint_name)
            if joint_element is None:
                continue
            joint_type = "free" if joint_element.tag == "freejoint" else joint_element.type
            internal_jointid = mj.mj_name2id(
                mj_model, mj.mjtObj.mjOBJ_JOINT, joint_element.full_identifier
            )
            adr = mj_model.jnt_dofadr[internal_jointid]
            end = adr + _world_mod._STATE_DIM_BY_JOINT_TYPE[joint_type]
            neutral_qpos[adr:end] = neutral_state
        for fly_name, fly in self.fly_lookup.items():
            qpos = fly._get_neutral_qpos(mj_model)
            idx = qpos.nonzero()
            neutral_qpos[idx] = qpos[idx]
            ctrl = fly._get_neutral_ctrl(mj_model)
            idx = ctrl.nonzero()
            neutral_ctrl[idx] = ctrl[idx]
        self._neutral_keyframe.qpos = neutral_qpos
        self._neutral_keyframe.ctrl = neutral_ctrl
    _world_mod.BaseWorld._rebuild_neutral_keyframe = _patched_rebuild

    # Build flies
    flies = []
    for i in range(num_flies):
        fly = make_fly(f"fly_{i}")
        pos, rot = spawn_configs[i]
        world.add_fly(fly, pos, rot)
        flies.append(fly)

    sim = Simulation(world)
    sim.reset()
    sim.warmup(duration_s=0.02)

    # Visuals
    world.upload_texture(sim.mj_model)
    world.apply_atmosphere(sim.mj_model)

    # Adhesion on
    for fly in flies:
        sim.set_leg_adhesion_states(fly.name, np.ones(6, dtype=bool))

    # CPGs + neutral ctrl values (the actuator targets at the standing pose)
    cpgs = {}
    neutral_ctrls = {}
    rng = np.random.RandomState(123)
    for fly in flies:
        n_act = len(fly.get_actuated_jointdofs_order(ActuatorType.POSITION))
        cpgs[fly.name] = SimpleCPG(n_act, timestep)
        # Grab this fly's neutral ctrl from the compiled data
        # (set_actuator_inputs expects absolute angles, not offsets)
        act_ids = sim._intern_actuatorids_by_type_by_fly[ActuatorType.POSITION][fly.name]
        neutral_ctrls[fly.name] = sim.mj_data.ctrl[act_ids].copy()

    turn_biases = {fly.name: 0.0 for fly in flies}
    turn_update_steps = int(0.5 / timestep)

    # Food manager
    food_mgr = FoodManager(sim.mj_model, sim.mj_data, MAX_FOOD, food_radius=0.4,
                           spawn_range=25.0, spawn_interval_s=3.0, despawn_dist=2.0)
    food_update_steps = int(1.0 / timestep)

    # Launch viewer
    viewer = mjviewer.launch_passive(
        sim.mj_model, sim.mj_data,
        show_left_ui=True, show_right_ui=True,
    )

    for i in range(sim.mj_model.ntex):
        name = mj.mj_id2name(sim.mj_model, mj.mjtObj.mjOBJ_TEXTURE, i)
        if name == world._forest_tex_name:
            viewer.update_texture(i)
            break

    viewer.cam.type = mj.mjtCamera.mjCAMERA_FREE
    viewer.cam.distance = 15.0
    viewer.cam.elevation = -30.0
    viewer.cam.azimuth = 135.0
    viewer.cam.lookat[:] = [0, 0, 0]

    print("=" * 65)
    print("  NeuroMechFly v2 — Forest Floor Simulation")
    print(f"  {num_flies} flies | food spawns randomly every ~3s")
    print("  Orbit/zoom/pan with mouse. Close window or Ctrl+C to stop.")
    print("=" * 65)

    step_count = 0
    metrics_interval = int(1.0 / timestep)
    sync_interval = 200  # sync viewer every N steps
    wall_start = time.perf_counter()
    sim_start = 0.0

    try:
        while viewer.is_running():
            # --- Turning decisions ---
            if step_count % turn_update_steps == 0:
                food_pos = food_mgr.get_active_positions()
                for fly in flies:
                    body_pos = sim.get_body_positions(fly.name)
                    fly_xy = body_pos[0, :2]

                    if len(food_pos) > 0:
                        dists = np.linalg.norm(food_pos - fly_xy, axis=1)
                        nearest = food_pos[np.argmin(dists)]
                        to_food = nearest - fly_xy

                        body_rot = sim.get_body_rotations(fly.name)
                        quat = body_rot[0]
                        fwd_x = 1 - 2 * (quat[2]**2 + quat[3]**2)
                        fwd_y = 2 * (quat[1]*quat[2] + quat[0]*quat[3])
                        cross = fwd_x * to_food[1] - fwd_y * to_food[0]
                        turn_biases[fly.name] = np.clip(cross * 0.15, -0.8, 0.8)
                    else:
                        turn_biases[fly.name] = rng.uniform(-0.3, 0.3)

            # --- CPG control ---
            for fly in flies:
                ctrl = cpgs[fly.name].step(
                    neutral_ctrls[fly.name], turn_biases[fly.name]
                )
                sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, ctrl)

            sim.step()
            step_count += 1

            # --- Food updates ---
            if step_count % food_update_steps == 0:
                fly_positions = []
                for fly in flies:
                    bp = sim.get_body_positions(fly.name)
                    fly_positions.append(bp[0])
                food_mgr.update(step_count * timestep, fly_positions)

            # --- Viewer sync + pacing ---
            if step_count % sync_interval == 0:
                viewer.sync()
                # Pace to realtime_factor
                sim_elapsed = step_count * timestep
                wall_elapsed = time.perf_counter() - wall_start
                target_wall = sim_elapsed / realtime_factor
                if wall_elapsed < target_wall:
                    time.sleep(target_wall - wall_elapsed)

            # --- Metrics ---
            if step_count % metrics_interval == 0:
                sim_time = step_count * timestep
                food_pos = food_mgr.get_active_positions()
                n_food = len(food_pos)
                print(f"\n--- t={sim_time:.1f}s  food={n_food} ---")
                for fly in flies:
                    bp = sim.get_body_positions(fly.name)
                    xyz = bp[0]
                    contact_active, forces, *_ = sim.get_ground_contact_info(fly.name)
                    legs = int(contact_active.sum())
                    grf = np.linalg.norm(forces, axis=1).sum()

                    if len(food_pos) > 0:
                        dists = np.linalg.norm(food_pos - xyz[:2], axis=1)
                        fd = dists.min()
                    else:
                        fd = float('inf')

                    print(
                        f"  {fly.name}: "
                        f"({xyz[0]:+6.1f},{xyz[1]:+6.1f},{xyz[2]:4.2f}) "
                        f"legs={legs}/6 grf={grf:5.0f} "
                        f"food={fd:5.1f}mm"
                    )

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        viewer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
