#!/usr/bin/env python3
"""
fly_live.py - live NeuroMechFly / FlyGym viewer

For a real on-screen window on Linux:
- run this in a graphical desktop session
- prefer an Xorg/X11 session
- launch with: MUJOCO_GL=glfw python fly_live.py
"""

import os
import sys
import time
import pickle
from pathlib import Path

# For live viewing, prefer GLFW unless caller explicitly set something else.
os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
from flygym import Fly, ZStabilizedCamera, SingleFlySimulation, get_data_path
from flygym.preprogrammed import all_leg_dofs


def main() -> int:
    run_time = 2.0          # longer so you can watch it
    timestep = 1e-4
    play_sleep = 0.002      # pacing so it does not blast by too fast
    actuated_joints = all_leg_dofs

    print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
    print("XDG_SESSION_TYPE =", os.environ.get("XDG_SESSION_TYPE"))
    print("DISPLAY =", os.environ.get("DISPLAY"))
    print("WAYLAND_DISPLAY =", os.environ.get("WAYLAND_DISPLAY"))

    data_path = get_data_path("flygym", "data")
    with open(data_path / "behavior" / "210902_pr_fly1.pkl", "rb") as f:
        data = pickle.load(f)

    target_num_steps = int(run_time / timestep)
    data_block = np.zeros((len(actuated_joints), target_num_steps), dtype=np.float64)

    input_t = np.arange(len(data["joint_LFCoxa"])) * data["meta"]["timestep"]
    output_t = np.arange(target_num_steps) * timestep

    for i, joint in enumerate(actuated_joints):
        data_block[i, :] = np.interp(output_t, input_t, data[joint])

    fly = Fly(
        init_pose="stretch",
        actuated_joints=actuated_joints,
        control="position",
    )

    cam = ZStabilizedCamera(
        attachment_point=fly.model.worldbody,
        camera_name="camera_left",
        targeted_fly_names=fly.name,
        play_speed=0.1,
    )

    sim = None
    try:
        sim = SingleFlySimulation(
            fly=fly,
            cameras=[cam],
            timestep=timestep,
        )

        obs, info = sim.reset()

        for i in range(target_num_steps):
            action = {"joints": data_block[:, i]}
            obs, reward, terminated, truncated, info = sim.step(action)

            # Live window update
            sim.render()

            # Slow down enough to watch
            time.sleep(play_sleep)

            if terminated or truncated:
                break

        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "kinematic_replay.mp4"
        cam.save_video(out_file)

        print("done")
        print(f"saved: {out_file}")
        return 0

    except Exception as e:
        print("\nLive rendering failed.")
        print(f"Error: {e}")
        print("\nTry this exact launch command from an Xorg desktop session:")
        print("  MUJOCO_GL=glfw python fly_live.py")
        print("\nIf you only need offscreen video, use:")
        print("  MUJOCO_GL=osmesa python fly_live.py")
        return 1

    finally:
        if sim is not None:
            try:
                sim.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
