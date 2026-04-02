#!/usr/bin/env python3
"""
fly_autonomous.py - Watch multiple NeuroMechFly v2 flies on a forest floor

Uses real recorded walking kinematics (from Spotlight motion capture) looped
continuously, with turning modulation toward food. Food spawns randomly.

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
from collections import defaultdict

from flygym.anatomy import (
    Skeleton, JointPreset, AxisOrder, ActuatedDOFPreset, LEG_LINKS,
)
from flygym.compose import Fly, ActuatorType, FlatGroundWorld, KinematicPosePreset
from flygym.utils.math import Rotation3D
from flygym import Simulation
from flygym_demo.spotlight_data import MotionSnippet

MAX_FOOD = 12


# ---------------------------------------------------------------------------
# Multi-fly safe world with food
# ---------------------------------------------------------------------------
class ForestFloorWorld(FlatGroundWorld):
    def __init__(self, n_food=MAX_FOOD, **kw):
        super().__init__(**kw)
        # Forest texture
        ftex = self.mjcf_root.asset.add(
            "texture", name="forest_floor_tex", type="2d", builtin="flat",
            width=512, height=512, rgb1=(0.28, 0.22, 0.14), rgb2=(0.18, 0.14, 0.08))
        fmat = self.mjcf_root.asset.add(
            "material", name="forest_floor_mat", texture=ftex,
            texrepeat=(8, 8), reflectance=0.02, specular=0.05)
        self.ground_geom.material = fmat
        self._forest_tex_name = "forest_floor_tex"

        # Skybox
        self.mjcf_root.asset.add(
            "texture", name="forest_sky", type="skybox", builtin="gradient",
            rgb1=(0.55, 0.65, 0.45), rgb2=(0.25, 0.35, 0.2), width=512, height=512)

        # Lights
        self.mjcf_root.worldbody.add(
            "light", name="sun", mode="fixed", directional=True, castshadow=True,
            pos=(0, 0, 100), dir=(0.3, 0.2, -1),
            ambient=(0.2, 0.18, 0.12), diffuse=(0.75, 0.65, 0.45),
            specular=(0.25, 0.22, 0.15))

        # Food slots (mocap bodies, hidden initially)
        for i in range(n_food):
            body = self.mjcf_root.worldbody.add(
                "body", name=f"food_{i}", pos=(0, 0, -10), mocap=True)
            body.add("geom", type="sphere", size=(0.3,),
                     rgba=(0.85, 0.15, 0.1, 0.9), contype=0, conaffinity=0)

    def _set_ground_contact(self, fly, bodysegs, params):
        for b in bodysegs:
            g = fly.mjcf_root.find("geom", b.name)
            self.mjcf_root.contact.add(
                "pair", geom1=g, geom2=self.ground_geom,
                name=f"{fly.name}_{b.name}-ground",
                friction=params.get_friction_tuple(),
                solref=params.get_solref_tuple(),
                solimp=params.get_solimp_tuple(), margin=params.margin)

    def _add_ground_contact_sensors(self, fly, bodysegs):
        if self.legpos_to_groundcontactsensors_by_fly is None:
            self.legpos_to_groundcontactsensors_by_fly = defaultdict(dict)
        bl = defaultdict(list)
        for b in bodysegs:
            if b.is_leg(): bl[b.pos].append(b)
        for leg, segs in bl.items():
            segs.sort(key=lambda s: LEG_LINKS.index(s.link))
            body = fly.bodyseg_to_mjcfbody[segs[0]]
            s = self.mjcf_root.sensor.add(
                "contact", subtree1=body, geom2=self.ground_geom,
                num=1, reduce="netforce",
                data="found force torque pos normal tangent",
                name=f"{fly.name}_ground_contact_{leg}_leg")
            self.legpos_to_groundcontactsensors_by_fly[fly.name][leg] = s

    def upload_texture(self, mj_model):
        tex_img = _generate_forest_floor_texture(512, 512)
        for i in range(mj_model.ntex):
            name = mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_TEXTURE, i)
            if name == self._forest_tex_name:
                h, w = mj_model.tex_height[i], mj_model.tex_width[i]
                adr = mj_model.tex_adr[i]
                mj_model.tex_data[adr:adr + h*w*3] = tex_img[:h, :w, :].flatten()
                break

    def apply_atmosphere(self, mj_model):
        mj_model.vis.rgba.fog[:] = [0.25, 0.32, 0.2, 1.0]
        mj_model.vis.map.fogstart = 20.0
        mj_model.vis.map.fogend = 80.0


def _generate_forest_floor_texture(w=512, h=512):
    rng = np.random.RandomState(42)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    xs, ys = np.meshgrid(np.arange(w)/w, np.arange(h)/h)
    n = np.sin(xs*13.7+ys*9.3)*0.3 + np.sin(xs*27.1+ys*19.7)*0.15
    bn = rng.normal(0, 6, (h, w))
    img[:,:,0] = np.clip(62+n*40+bn, 30, 100).astype(np.uint8)
    img[:,:,1] = np.clip(45+n*30+bn*0.8, 20, 75).astype(np.uint8)
    img[:,:,2] = np.clip(28+n*15+bn*0.5, 10, 50).astype(np.uint8)
    for _ in range(150):
        cx, cy, r = rng.randint(0,w), rng.randint(0,h), rng.randint(3,12)
        yy, xx = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask = xx**2+yy**2 <= r**2
        img[mask] = (img[mask]*rng.uniform(0.5,0.8)).astype(np.uint8)
    colors = [(85,60,20),(100,80,25),(55,65,30),(70,30,20)]
    for _ in range(100):
        cx,cy,rx,ry = rng.randint(0,w), rng.randint(0,h), rng.randint(2,8), rng.randint(4,14)
        a = rng.uniform(0,np.pi); c = np.array(colors[rng.randint(len(colors))])
        yy,xx = np.ogrid[-cy:h-cy, -cx:w-cx]
        ca,sa = np.cos(a),np.sin(a)
        mask = ((ca*xx+sa*yy)/max(rx,1))**2 + ((-sa*xx+ca*yy)/max(ry,1))**2 <= 1
        bl = rng.uniform(0.4,0.8)
        img[mask] = (img[mask]*(1-bl)+c*bl).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Food manager
# ---------------------------------------------------------------------------
class FoodManager:
    def __init__(self, mj_model, mj_data, spawn_range=10.0):
        self.mj_model, self.mj_data = mj_model, mj_data
        self.rng = np.random.RandomState(99)
        self.spawn_range = spawn_range
        self.food_ids = []
        for i in range(MAX_FOOD):
            bid = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, f"food_{i}")
            self.food_ids.append(bid)
        self.active = np.zeros(MAX_FOOD, dtype=bool)
        self.positions = np.zeros((MAX_FOOD, 2))
        for _ in range(5): self._spawn()

    def _spawn(self):
        slots = np.where(~self.active)[0]
        if len(slots) == 0: return
        s = slots[self.rng.randint(len(slots))]
        x, y = self.rng.uniform(-self.spawn_range, self.spawn_range, 2)
        self.positions[s] = [x, y]
        self.active[s] = True
        mid = self.mj_model.body_mocapid[self.food_ids[s]]
        if mid >= 0: self.mj_data.mocap_pos[mid] = [x, y, 0.3]

    def _hide(self, s):
        self.active[s] = False
        mid = self.mj_model.body_mocapid[self.food_ids[s]]
        if mid >= 0: self.mj_data.mocap_pos[mid] = [0, 0, -10]

    def update(self, fly_positions):
        for s in range(MAX_FOOD):
            if not self.active[s]: continue
            for fp in fly_positions:
                if np.linalg.norm(fp[:2] - self.positions[s]) < 1.5:
                    self._hide(s); break
        if self.active.sum() < 4:
            self._spawn(); self._spawn()
        elif self.active.sum() < 6 and self.rng.random() < 0.3:
            self._spawn()

    def get_active_positions(self):
        return self.positions[self.active]


# ---------------------------------------------------------------------------
# Walking controller using real recorded kinematics
# ---------------------------------------------------------------------------
class WalkingController:
    """Loops recorded walking kinematics with turning modulation."""

    def __init__(self, fly, sim_timestep):
        snippet = MotionSnippet()
        dof_order = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        self.joint_angles = snippet.get_joint_angles(
            output_timestep=sim_timestep,
            output_dof_order=dof_order,
        )
        self.n_steps = self.joint_angles.shape[0]
        self.n_dofs = self.joint_angles.shape[1]
        self.dpl = self.n_dofs // 6
        self.idx = 0

    def step(self, turn_bias=0.0):
        """Return target joint angles for this timestep."""
        angles = self.joint_angles[self.idx % self.n_steps].copy()

        # Apply turning: modulate coxa-pitch (DOF index 1 per leg) amplitude
        for leg_idx in range(6):
            b = leg_idx * self.dpl
            if leg_idx < 3:  # left legs
                angles[b + 1] *= (1.0 + turn_bias * 0.3)
            else:  # right legs
                angles[b + 1] *= (1.0 - turn_bias * 0.3)

        self.idx += 1
        return angles


# ---------------------------------------------------------------------------
# Fly factory
# ---------------------------------------------------------------------------
def make_fly(name):
    fly = Fly(name=name)
    skeleton = Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL,
        joint_preset=JointPreset.LEGS_ONLY,
    )
    fly.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
    actuated_dofs = skeleton.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY
    )
    fly.add_actuators(
        actuated_dofs, ActuatorType.POSITION,
        kp=150.0, neutral_input=KinematicPosePreset.NEUTRAL,
        ctrlrange=(-3.14, 3.14),
    )
    fly.add_leg_adhesion()
    fly.colorize()
    return fly


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    timestep = 1e-4
    num_flies = 3
    realtime_factor = 5.0  # 5x speed so movement is visible

    spawns = [
        ((0, 0, 0.7), Rotation3D("quat", (1, 0, 0, 0))),
        ((3, -2, 0.7), Rotation3D("quat", (0.924, 0, 0, 0.383))),
        ((-2, 2, 0.7), Rotation3D("quat", (0.924, 0, 0, -0.383))),
    ]

    world = ForestFloorWorld()

    # Multi-fly keyframe patch
    import flygym.compose.world as _wm
    _orig = _wm.BaseWorld._rebuild_neutral_keyframe
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
        self._neutral_keyframe.qpos = nq; self._neutral_keyframe.ctrl = nc
    _wm.BaseWorld._rebuild_neutral_keyframe = _fix

    flies = []
    for i, (pos, rot) in enumerate(spawns):
        fly = make_fly(f"fly_{i}")
        world.add_fly(fly, pos, rot)
        flies.append(fly)

    sim = Simulation(world)
    sim.reset()
    sim.warmup(duration_s=0.05)

    world.upload_texture(sim.mj_model)
    world.apply_atmosphere(sim.mj_model)

    for fly in flies:
        sim.set_leg_adhesion_states(fly.name, np.ones(6, dtype=bool))

    # Walking controllers (using real recorded kinematics)
    controllers = {fly.name: WalkingController(fly, timestep) for fly in flies}
    turn_biases = {fly.name: 0.0 for fly in flies}
    rng = np.random.RandomState(123)
    turn_update_steps = int(0.5 / timestep)

    # Food
    food_mgr = FoodManager(sim.mj_model, sim.mj_data, spawn_range=10.0)
    food_update_steps = int(1.0 / timestep)

    # Viewer
    viewer = mjviewer.launch_passive(
        sim.mj_model, sim.mj_data,
        show_left_ui=True, show_right_ui=True)

    for i in range(sim.mj_model.ntex):
        name = mj.mj_id2name(sim.mj_model, mj.mjtObj.mjOBJ_TEXTURE, i)
        if name == world._forest_tex_name:
            viewer.update_texture(i); break

    viewer.cam.type = mj.mjtCamera.mjCAMERA_FREE
    viewer.cam.distance = 15.0
    viewer.cam.elevation = -30.0
    viewer.cam.azimuth = 135.0
    viewer.cam.lookat[:] = [0, 0, 0]

    print("=" * 60)
    print("  NeuroMechFly v2 — Forest Floor")
    print(f"  {num_flies} flies | real walking kinematics | {realtime_factor}x speed")
    print("  Close window or Ctrl+C to stop.")
    print("=" * 60)

    step_count = 0
    sync_interval = 500
    metrics_interval = int(2.0 / timestep)
    wall_start = time.perf_counter()

    try:
        while viewer.is_running():
            if step_count % turn_update_steps == 0:
                food_pos = food_mgr.get_active_positions()
                for fly in flies:
                    bp = sim.get_body_positions(fly.name)
                    xy = bp[0, :2]
                    if len(food_pos) > 0:
                        dists = np.linalg.norm(food_pos - xy, axis=1)
                        to_food = food_pos[np.argmin(dists)] - xy
                        rot = sim.get_body_rotations(fly.name)
                        q = rot[0]
                        fx = 1 - 2*(q[2]**2+q[3]**2)
                        fy = 2*(q[1]*q[2]+q[0]*q[3])
                        cross = fx*to_food[1] - fy*to_food[0]
                        turn_biases[fly.name] = np.clip(cross * 0.15, -0.8, 0.8)
                    else:
                        turn_biases[fly.name] = rng.uniform(-0.3, 0.3)

            for fly in flies:
                angles = controllers[fly.name].step(turn_biases[fly.name])
                sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, angles)

            sim.step()
            step_count += 1

            if step_count % food_update_steps == 0:
                fps = [sim.get_body_positions(f.name)[0] for f in flies]
                food_mgr.update(fps)

            if step_count % sync_interval == 0:
                viewer.sync()
                sim_t = step_count * timestep
                wall_t = time.perf_counter() - wall_start
                target = sim_t / realtime_factor
                if wall_t < target:
                    time.sleep(target - wall_t)

            if step_count % metrics_interval == 0:
                sim_t = step_count * timestep
                fp = food_mgr.get_active_positions()
                print(f"\n--- t={sim_t:.0f}s  food={len(fp)} ---")
                for fly in flies:
                    bp = sim.get_body_positions(fly.name)
                    ca, fo, *_ = sim.get_ground_contact_info(fly.name)
                    fd = np.linalg.norm(fp - bp[0,:2], axis=1).min() if len(fp) else 999
                    print(f"  {fly.name}: ({bp[0,0]:+6.1f},{bp[0,1]:+6.1f},{bp[0,2]:.2f}) "
                          f"legs={int(ca.sum())}/6 food={fd:.1f}mm")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        viewer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
