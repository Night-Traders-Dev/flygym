#!/usr/bin/env python3
"""
fly.py - minimal NeuroMechFly / FlyGym kinematic replay with stable headless rendering

This version:
- defaults to OSMesa for software rendering on Linux
- replays bundled walking kinematics
- writes outputs/kinematic_replay.mp4
- does explicit cleanup to reduce dm_control shutdown noise
"""

import os

# IMPORTANT:
# Set rendering backend BEFORE importing flygym / dm_control / mujoco.
# OSMesa = software rendering, good for headless/video output.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import gc
import pickle
from pathlib import Path

import numpy as np

from flygym import Fly, ZStabilizedCamera, SingleFlySimulation, get_data_path
from flygym.preprogrammed import all_leg_dofs


def main() -> None:
    run_time = 0.2
    timestep = 1e-4
    actuated_joints = all_leg_dofs

    fly = None
    cam = None
    sim = None

    try:
        # Load bundled example walking data
        data_path = get_data_path("flygym", "data")
        with open(data_path / "behavior" / "210902_pr_fly1.pkl", "rb") as f:
            data = pickle.load(f)

        target_num_steps = int(run_time / timestep)
        data_block = np.zeros((len(actuated_joints), target_num_steps), dtype=np.float64)

        input_t = np.arange(len(data["joint_LFCoxa"])) * data["meta"]["timestep"]
        output_t = np.arange(target_num_steps) * timestep

        for i, joint in enumerate(actuated_joints):
            data_block[i, :] = np.interp(output_t, input_t, data[joint])

        # Build fly + camera + simulation
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

        sim = SingleFlySimulation(
            fly=fly,
            cameras=[cam],
            timestep=timestep,
        )

        obs, info = sim.reset()

        for i in range(target_num_steps):
            action = {"joints": data_block[:, i]}
            obs, reward, terminated, truncated, info = sim.step(action)

            # Required to generate frames for the attached camera/video.
            sim.render()

            if terminated or truncated:
                break

        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "kinematic_replay.mp4"
        cam.save_video(out_file)

        print("done")
        print(f"saved: {out_file}")

    finally:
        # Close simulation first
        if sim is not None:
            try:
                sim.close()
            except Exception as e:
                print(f"warning: sim.close() raised: {e}")

        # Drop references explicitly before interpreter shutdown
        sim = None
        cam = None
        fly = None

        # Encourage cleanup before atexit finalization
        gc.collect()


if __name__ == "__main__":
    main()
