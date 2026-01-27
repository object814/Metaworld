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
from metaworld.wrappers import ProprioImageObsWrapper, ProprioMultiImageObsWrapper
from metaworld.policies.sawyer_pick_place_v3_policy import SawyerPickPlaceV3Policy as pick_policy
from metaworld.policies.sawyer_drawer_open_v3_policy import SawyerDrawerOpenV3Policy as drawer_policy

def render_episode(env_name,
                   out_path="out.gif",
                   episode_length=500,
                   image_size=(480, 480),
                   action_policy="random",
                   camera_name=["topview", "front", "gripperPOV"],
                   verbose=False):
    multiple_cameras = None
    if len(camera_name) == 1:
        camera_name = camera_name[0]
        env = gym.make("Meta-World/MT1", env_name=env_name, render_mode="rgb_array", camera_name=camera_name)
        env = ProprioImageObsWrapper(env,
                                     image_height=image_size[0],
                                     image_width=image_size[1])
        multiple_cameras = False
    elif len(camera_name) > 1:
        env = gym.make("Meta-World/MT1", env_name=env_name, render_mode="rgb_array")
        env = ProprioMultiImageObsWrapper(env,
                                          image_height=image_size[0],
                                          image_width=image_size[1],
                                          camera_names=camera_name)
        multiple_cameras = True
    else:
        raise ValueError("camera_name should be a list with at least one camera name.")
    
    if action_policy == "policy":
        if env_name == "pick-place-v3":
            policy = pick_policy()
        elif env_name == "drawer-open-v3":
            policy = drawer_policy()
        else:
            raise NotImplementedError(f"Policy for {env_name} is not implemented.")
    
    frames = []
    time_stamp = time.time()

    obs, info = env.reset()
    for t in range(episode_length):
        if action_policy == "random":
            action = env.action_space.sample()
        elif action_policy == "policy":
            action = policy.get_action(obs["original_obs"])
        else:
            action = np.array([0.02, -0.02, 0.01, 0.1]) # Simple hardcoded action for testing
        obs, reward, terminated, truncated, info = env.step(action)
        if verbose:
            print(f"EE Pos: {obs['proprio'][0]:.3f}, {obs['proprio'][1]:.3f}, {obs['proprio'][2]:.3f}, EE velocity: {obs['proprio'][3]:.3f}, {obs['proprio'][4]:.3f}, {obs['proprio'][5]:.3f}, Gripper Val: {obs['proprio'][6]:.3f}")
        if not multiple_cameras:
            frames.append(obs["image"])
        else:
            # image observations from multiple cameras are [H, W, 3*num_cameras]
            img = obs["image"]
            h, w, c = img.shape
            num_cameras = c // 3
            camera_frames = []
            for i in range(num_cameras):
                camera_frame = img[:,:, i*3:(i+1)*3]
                camera_frames.append(camera_frame)
            combined_frame = np.concatenate(camera_frames, axis=1)
            frames.append(combined_frame)

        if terminated or truncated:
            break
    env.close()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=30)
    print(f"Wrote {out_path} ({len(frames)} frames) in {time.time()-time_stamp:.2f} seconds.")

if __name__ == "__main__":
    # choose task names
    PICK_ENV = "pick-place-v3"
    DRAWER_ENV = "drawer-open-v3"
    COMPO_ENV = "compo-draweropen-pickplace"

    render_episode(PICK_ENV,
                   out_path="gifs/pick_place_policy.gif",
                   episode_length=100,
                   action_policy="policy",
                   camera_name=["topview", "front", "gripperPOV"])
    render_episode(DRAWER_ENV,
                   out_path="gifs/drawer_open_policy.gif",
                   episode_length=100,
                   action_policy="policy",
                   camera_name=["topview", "front", "gripperPOV"])