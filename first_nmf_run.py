# save as first_flygym_run.py
import numpy as np
import pickle
from pathlib import Path

from flygym import Fly, ZStabilizedCamera, SingleFlySimulation, get_data_path
from flygym.preprogrammed import all_leg_dofs

run_time = 0.2
timestep = 1e-4
actuated_joints = all_leg_dofs

data_path = get_data_path("flygym", "data")
with open(data_path / "behavior" / "210902_pr_fly1.pkl", "rb") as f:
    data = pickle.load(f)

target_num_steps = int(run_time / timestep)
data_block = np.zeros((len(actuated_joints), target_num_steps))

input_t = np.arange(len(data["joint_LFCoxa"])) * data["meta"]["timestep"]
output_t = np.arange(target_num_steps) * timestep

for i, joint in enumerate(actuated_joints):
    data_block[i, :] = np.interp(output_t, input_t, data[joint])

fly = Fly(init_pose="stretch", actuated_joints=actuated_joints, control="position")
cam = ZStabilizedCamera(
    attachment_point=fly.model.worldbody,
    camera_name="camera_left",
    targeted_fly_names=fly.name,
    play_speed=0.1,
)
sim = SingleFlySimulation(fly=fly, cameras=[cam], timestep=timestep)

obs, info = sim.reset()

for i in range(target_num_steps):
    action = {"joints": data_block[:, i]}
    obs, reward, terminated, truncated, info = sim.step(action)
    sim.render()
    if terminated or truncated:
        break

out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)
cam.save_video(out_dir / "kinematic_replay.mp4")
sim.close()

print("done")
print("saved:", out_dir / "kinematic_replay.mp4")
