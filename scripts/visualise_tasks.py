import os
os.environ["MUJOCO_GL"] = "egl"
import gymnasium as gym
import sys
import time
import imageio
import numpy as np

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import metaworld
from metaworld.utils.obs_wrapper import ProprioImageObsWrapper, ProprioMultiImageObsWrapper

# choose task names
PICK_ENV = "pick-place-v3"
DRAWER_ENV = "drawer-open-v3"

def render_episode(env_name, out_path="out.gif", episode_length=500, random_actions=True):
    env = gym.make("Meta-World/MT1", env_name=env_name, render_mode="rgb_array", camera_name="topview")
    env = ProprioImageObsWrapper(env, image_height=128, image_width=128)
    obs, info = env.reset()
    frames = []
    time_stamp = time.time()
    for t in range(episode_length):
        if random_actions:
            action = env.action_space.sample()
        else:
            action = np.array([0.02, -0.02, 0.01, 0.1])
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"EE Pos: {obs['proprio'][0]:.3f}, {obs['proprio'][1]:.3f}, {obs['proprio'][2]:.3f}, EE velocity: {obs['proprio'][3]:.3f}, {obs['proprio'][4]:.3f}, {obs['proprio'][5]:.3f}, Gripper Val: {obs['proprio'][6]:.3f}")
        # frame = env.render()
        frames.append(obs["image"])
        if terminated or truncated:
            break
    env.close()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=30)
    print(f"Wrote {out_path} ({len(frames)} frames) in {time.time()-time_stamp:.2f} seconds.")

def render_episode_multi_camera(env_name, out_path="out.gif", episode_length=500, random_actions=True):
    env = gym.make("Meta-World/MT1", env_name=env_name, render_mode="rgb_array")
    env = ProprioMultiImageObsWrapper(env,
                                      image_height=128,
                                      image_width=128,
                                      camera_names=["topview", "corner", "front"])
    obs, info = env.reset()
    frames = []
    time_stamp = time.time()
    for t in range(episode_length):
        if random_actions:
            action = env.action_space.sample()
        else:
            action = np.array([0.02, -0.02, 0.01, 0.1])
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"EE Pos: {obs['proprio'][0]:.3f}, {obs['proprio'][1]:.3f}, {obs['proprio'][2]:.3f}, EE velocity: {obs['proprio'][3]:.3f}, {obs['proprio'][4]:.3f}, {obs['proprio'][5]:.3f}, Gripper Val: {obs['proprio'][6]:.3f}")
        # frame = env.render()
        # Combine multiple camera images side by side
        combined_frame = np.concatenate(obs["images"], axis=1)
        frames.append(combined_frame)
        if terminated or truncated:
            break
    env.close()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=30)
    print(f"Wrote {out_path} ({len(frames)} frames) in {time.time()-time_stamp:.2f} seconds.")

if __name__ == "__main__":
    render_episode(PICK_ENV, out_path="pick_place.gif", episode_length=100, random_actions=False)
    render_episode(DRAWER_ENV, out_path="drawer_open.gif", episode_length=100, random_actions=False)
    render_episode_multi_camera(PICK_ENV, out_path="pick_place_multi_cam.gif", episode_length=100, random_actions=False)
    render_episode_multi_camera(DRAWER_ENV, out_path="drawer_open_multi_cam.gif", episode_length=100, random_actions=False)