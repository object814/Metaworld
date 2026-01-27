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
        pickplace_policy = pick_policy()
        draweropen_policy = drawer_policy()

    
    frames = []
    time_stamp = time.time()

    drawer_opened = False
    drawer_open_step = 0
    stabled = False
    lifted = False

    obs, info = env.reset()
    for t in range(episode_length):
        if action_policy == "random":
            action = env.action_space.sample()
        elif action_policy == "policy":
            if t == 0:
                action = np.array([0.0, 0.0, 0.0, 0.0])
            else:
                if info["drawer_opened"] < 0.99 and t > 0:
                    action = draweropen_policy.get_action(obs["original_obs"])
                elif info["drawer_opened"] >= 0.99 and drawer_opened == False:
                    drawer_opened = True
                    drawer_open_step = t
                    action = np.array([0.0, 0.0, 0.0, 0.0])
                    print("DEBUG: Drawer opened at step", t)
                elif drawer_opened and not stabled and not lifted:
                    action = np.array([0.0, 0.0, 0.0, 0.0])
                    if t - drawer_open_step >= 20:
                        stabled = True
                        print("DEBUG: Stabled at step", t)
                elif drawer_opened and stabled and not lifted:
                    action = np.array([0.0, -0.01, 0.2, 0.0])
                    if t - drawer_open_step >= 100:
                        lifted = True
                        print("DEBUG: Lifted up at step", t)
                        print("DEBUG: Start pick and place")
                else:
                    action = pickplace_policy.get_action(obs["original_obs"])
        else:
            action = np.array([0.0, 0.0, 0.2, 0.0]) # Simple hardcoded action for testing
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
            # Add reward text to top-left corner of the frame
            combined_frame = combined_frame.copy()
            combined_frame = cv2.putText(combined_frame,
                                         f"Reward: {reward:.2f}",
                                         org=(10,30),
                                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                         fontScale=1,
                                         color=(255,0,0),
                                         thickness=2)
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

    render_episode(COMPO_ENV,
                   out_path="gifs/compo_policy.gif",
                   episode_length=300,
                   action_policy="policy",
                   camera_name=["topview", "front", "gripperPOV"])