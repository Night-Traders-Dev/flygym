#!/usr/bin/env python3
"""Headless audit: runs the full multi-fly sim and saves screenshots at intervals."""
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["LIBDECOR_PLUGIN_DIR"] = "/tmp/libdecor_cairo_only"

import numpy as np
import mujoco as mj
import imageio.v3 as iio
from collections import defaultdict

from flygym.anatomy import Skeleton, JointPreset, AxisOrder, ActuatedDOFPreset, LEG_LINKS
from flygym.compose import Fly, ActuatorType, FlatGroundWorld, KinematicPosePreset
from flygym.utils.math import Rotation3D
from flygym import Simulation
import flygym.compose.world as _w

# --- Multi-fly patches ---
def _patched_rebuild(self):
    mm, _ = self.compile()
    nq, nc = np.zeros(mm.nq), np.zeros(mm.nu)
    aj = {j.full_identifier: j for j in self.mjcf_root.find_all("joint")}
    for jn, ns in self.world_dof_neutral_states.items():
        je = aj.get(jn)
        if not je: continue
        jt = "free" if je.tag == "freejoint" else je.type
        jid = mj.mj_name2id(mm, mj.mjtObj.mjOBJ_JOINT, je.full_identifier)
        a = mm.jnt_qposadr[jid]
        nq[a : a + _w._STATE_DIM_BY_JOINT_TYPE[jt]] = ns
    for fn, f in self.fly_lookup.items():
        q = f._get_neutral_qpos(mm); nq[q.nonzero()] = q[q.nonzero()]
        c = f._get_neutral_ctrl(mm); nc[c.nonzero()] = c[c.nonzero()]
    self._neutral_keyframe.qpos = nq
    self._neutral_keyframe.ctrl = nc
_w.BaseWorld._rebuild_neutral_keyframe = _patched_rebuild


class MultiFlySafeWorld(FlatGroundWorld):
    """FlatGroundWorld with multi-fly contact name dedup + food slots."""
    def __init__(self, n_food=6, **kw):
        super().__init__(**kw)
        for i in range(n_food):
            body = self.mjcf_root.worldbody.add(
                "body", name=f"food_{i}", pos=(0, 0, -10), mocap=True
            )
            body.add("geom", type="sphere", size=(0.3,),
                     rgba=(0.85, 0.15, 0.1, 0.9), contype=0, conaffinity=0)

    def _set_ground_contact(self, fly, bodysegs, params):
        for bs in bodysegs:
            geom = fly.mjcf_root.find("geom", bs.name)
            self.mjcf_root.contact.add(
                "pair", geom1=geom, geom2=self.ground_geom,
                name=f"{fly.name}_{bs.name}-ground",
                friction=params.get_friction_tuple(),
                solref=params.get_solref_tuple(),
                solimp=params.get_solimp_tuple(), margin=params.margin,
            )

    def _add_ground_contact_sensors(self, fly, bodysegs):
        if self.legpos_to_groundcontactsensors_by_fly is None:
            self.legpos_to_groundcontactsensors_by_fly = defaultdict(dict)
        by_leg = defaultdict(list)
        for bs in bodysegs:
            if bs.is_leg(): by_leg[bs.pos].append(bs)
        for leg, segs in by_leg.items():
            segs.sort(key=lambda s: LEG_LINKS.index(s.link))
            body = fly.bodyseg_to_mjcfbody[segs[0]]
            sensor = self.mjcf_root.sensor.add(
                "contact", subtree1=body, geom2=self.ground_geom,
                num=1, reduce="netforce",
                data="found force torque pos normal tangent",
                name=f"{fly.name}_ground_contact_{leg}_leg",
            )
            self.legpos_to_groundcontactsensors_by_fly[fly.name][leg] = sensor


def make_fly(name):
    fly = Fly(name=name)
    sk = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ONLY)
    np_ = KinematicPosePreset.NEUTRAL
    fly.add_joints(sk, neutral_pose=np_)
    ad = sk.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(ad, ActuatorType.POSITION, kp=50.0, neutral_input=np_,
                      ctrlrange=(-3.14, 3.14))
    fly.add_leg_adhesion()
    fly.colorize()
    return fly


def take_screenshot(sim, renderer, flies, label, cam_dist=6.0):
    positions = [sim.get_body_positions(f.name)[0] for f in flies]
    center = np.mean(positions, axis=0)

    mj.mj_forward(sim.mj_model, sim.mj_data)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.distance = cam_dist
    cam.elevation = -20.0
    cam.azimuth = 160.0
    cam.lookat[:] = center
    cam.lookat[2] = 0.5  # look at ground level
    renderer.update_scene(sim.mj_data, cam)
    frame = renderer.render()
    path = f"/tmp/sim_{label}.png"
    iio.imwrite(path, frame)

    for fly in flies:
        bp = sim.get_body_positions(fly.name)
        ca, fo, *_ = sim.get_ground_contact_info(fly.name)
        print(f"  {fly.name}: ({bp[0,0]:+6.2f},{bp[0,1]:+6.2f},{bp[0,2]:.3f}) "
              f"legs={int(ca.sum())}/6")
    return path


def main():
    spawns = [
        ((0, 0, 0.7), Rotation3D("quat", (1, 0, 0, 0))),
        ((3, -2, 0.7), Rotation3D("quat", (0.924, 0, 0, 0.383))),
        ((-2, 2, 0.7), Rotation3D("quat", (0.924, 0, 0, -0.383))),
    ]

    world = MultiFlySafeWorld(n_food=6)
    flies = []
    for i, (pos, rot) in enumerate(spawns):
        fly = make_fly(f"fly_{i}")
        world.add_fly(fly, pos, rot)
        flies.append(fly)

    sim = Simulation(world)
    sim.reset()
    sim.warmup(duration_s=0.05)
    sim.mj_model.vis.global_.offwidth = 1280
    sim.mj_model.vis.global_.offheight = 720

    for fly in flies:
        sim.set_leg_adhesion_states(fly.name, np.ones(6, dtype=bool))

    # Place food
    rng = np.random.RandomState(42)
    for i in range(4):
        fx, fy = rng.uniform(-8, 8), rng.uniform(-8, 8)
        bid = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_BODY, f"food_{i}")
        mid = sim.mj_model.body_mocapid[bid]
        if mid >= 0:
            sim.mj_data.mocap_pos[mid] = [fx, fy, 0.3]

    # Neutral ctrls
    neutrals = {}
    phases = {}
    for fly in flies:
        ids = sim._intern_actuatorids_by_type_by_fly[ActuatorType.POSITION][fly.name]
        neutrals[fly.name] = sim.mj_data.ctrl[ids].copy()
        phases[fly.name] = 0.0

    renderer = mj.Renderer(sim.mj_model, height=720, width=1280)
    freq, dt = 12.0, 1e-4

    print("=== t=0s ===")
    take_screenshot(sim, renderer, flies, "t0")

    # Run and screenshot at intervals
    for target_s in [2, 5, 10]:
        steps = int(target_s * 10000) - int((target_s - (2 if target_s == 2 else target_s - [2,5,10][[2,5,10].index(target_s)-1])) * 0)
        # Just run to the target
        current_step = int(sum(phases.values()) / (2 * np.pi * freq * dt) / 3)  # approx
        run_steps = int(target_s / dt) - (0 if target_s == 2 else int([0,2,5][[2,5,10].index(target_s)] / dt))

        for step in range(run_steps):
            for fly in flies:
                na = len(neutrals[fly.name]); dpl = na // 6
                phases[fly.name] += 2 * np.pi * freq * dt
                off = np.zeros(na)
                for li in range(6):
                    lp = phases[fly.name] + (0 if li % 2 == 0 else np.pi)
                    b = li * dpl
                    sp, su = np.sin(lp), max(0, np.sin(lp))
                    off[b+1] = 0.5*sp; off[b+3] = -0.6*su; off[b+5] = 0.3*su
                sim.set_actuator_inputs(fly.name, ActuatorType.POSITION,
                                        neutrals[fly.name] + off)
            sim.step()

        print(f"\n=== t={target_s}s ===")
        take_screenshot(sim, renderer, flies, f"t{target_s}")

    renderer.close()
    print("\nDone. Screenshots: /tmp/sim_t0.png, t2, t5, t10")


if __name__ == "__main__":
    main()
