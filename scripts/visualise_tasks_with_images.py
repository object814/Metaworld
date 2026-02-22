import os
os.environ["MUJOCO_GL"] = "egl"
import gymnasium as gym
import sys
import time
import imageio
import numpy as np
import cv2

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
                   verbose=False,
                   n_intermediate_frames=5):
    multiple_cameras = None
    if isinstance(camera_name, str):
        camera_name = [camera_name]

    if len(camera_name) == 1:
        env = gym.make("Meta-World/MT1", env_name=env_name, render_mode="rgb_array", camera_name=camera_name[0])
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
        elif env_name == "drawer-open-v3" or env_name == "compo-draweropen-pickplace":
            policy = drawer_policy()
        else:
            raise NotImplementedError(f"Policy for {env_name} is not implemented.")
    
    # Storage for frames per camera
    frames_dict = {cam: [] for cam in camera_name}
    time_stamp = time.time()
    
    # Calculate indices for intermediate frames
    if n_intermediate_frames is not None and n_intermediate_frames > 0:
        save_indices = np.linspace(0, episode_length-1, n_intermediate_frames, dtype=int)
    else:
        save_indices = []

    obs, info = env.reset()
    for t in range(episode_length):
        print(f"Step {t+1}/{episode_length}")
        if action_policy == "random":
            action = env.action_space.sample()
        elif action_policy == "policy":
            action = policy.get_action(obs["original_obs"])
        else:
            action = np.array([0.02, -0.02, 0.01, 0.1]) # Simple hardcoded action for testing
        obs, reward, terminated, truncated, info = env.step(action)
        if verbose:
            print(f"EE Pos: {obs['proprio'][0]:.3f}, {obs['proprio'][1]:.3f}, {obs['proprio'][2]:.3f}, EE velocity: {obs['proprio'][3]:.3f}, {obs['proprio'][4]:.3f}, {obs['proprio'][5]:.3f}, Gripper Val: {obs['proprio'][6]:.3f}")
        
        img = obs["image"]
        current_step_frames = {}

        if not multiple_cameras:
            frames_dict[camera_name[0]].append(img)
            current_step_frames[camera_name[0]] = img
        else:
            # image observations from multiple cameras are [H, W, 3*num_cameras]
            # split them
            h, w, c = img.shape
            num_cameras = c // 3
            for i, cam_name in enumerate(camera_name):
                camera_frame = img[:,:, i*3:(i+1)*3]
                frames_dict[cam_name].append(camera_frame)
                current_step_frames[cam_name] = camera_frame

        if t in save_indices:
            # Save intermediate frames
            out_stem = Path(out_path).stem
            out_dir = Path(out_path).parent
            for cam_name, frame in current_step_frames.items():
                frame_filename = f"{out_stem}_{cam_name}_step_{t}.png"
                frame_path = out_dir / frame_filename
                out_dir.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(frame_path, frame)
                print(f"Saved intermediate frame: {frame_path}")

        if terminated or truncated:
            break
    env.close()
    
    print(f"Episode finished after {t+1} steps with reward {reward:.2f}.")
    
    # Save GIFs for each camera
    out_dir = Path(out_path).parent
    out_stem = Path(out_path).stem
    
    for cam_name, cam_frames in frames_dict.items():
        if len(cam_frames) > 0:
            gif_filename = f"{out_stem}_{cam_name}.gif"
            gif_path = out_dir / gif_filename
            print(f"Rendering episode for camera '{cam_name}' to {gif_path}...")
            out_dir.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(gif_path, cam_frames, fps=15)
            print(f"Wrote {gif_path} ({len(cam_frames)} frames) in {time.time()-time_stamp:.2f} seconds.")

if __name__ == "__main__":
    # choose task names
    PICK_ENV = "pick-place-v3"
    DRAWER_ENV = "drawer-open-v3"
    COMPO_ENV = "compo-draweropen-pickplace"
    COMPO_PICKPLACE_ENV = "compo-pickplace"
    
    # N intermediate frames to save
    N_INTERMEDIATE = 5

    render_episode(PICK_ENV,
                   out_path="gifs/pick_place_policy.gif",
                   episode_length=1,
                   action_policy="policy",
                   camera_name=["topview", "front", "corner"],
                   n_intermediate_frames=N_INTERMEDIATE)
    # render_episode(DRAWER_ENV,
    #                out_path="gifs/drawer_open_policy.gif",
    #                episode_length=1,
    #                action_policy="policy",
    #                camera_name=["topview", "front", "corner"],
    #                n_intermediate_frames=N_INTERMEDIATE)
    # render_episode(COMPO_ENV,
    #                out_path="gifs/compo_draweropen_pickplace_policy.gif",
    #                episode_length=1,
    #                action_policy="policy",
    #                camera_name=["topview", "front", "corner"],
    #                n_intermediate_frames=N_INTERMEDIATE)
    # render_episode(COMPO_PICKPLACE_ENV,
    #                out_path="gifs/compo_pickplace_policy.gif",
    #                episode_length=1,
    #                action_policy="random",
    #                camera_name=["topview", "front", "corner"],
    #                n_intermediate_frames=N_INTERMEDIATE)