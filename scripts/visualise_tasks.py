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
from metaworld.policies.sawyer_door_open_v3_policy import SawyerDoorOpenV3Policy as door_policy
from metaworld.policies.sawyer_door_close_v3_policy import SawyerDoorCloseV3Policy as door_close_policy
from metaworld.policies.sawyer_door_unlock_v3_policy import SawyerDoorUnlockV3Policy as door_unlock_policy
from metaworld.policies.sawyer_door_lock_v3_policy import SawyerDoorLockV3Policy as door_lock_policy
from metaworld.policies.sawyer_assembly_v3_policy import SawyerAssemblyV3Policy as assembly_policy
from metaworld.policies.sawyer_disassemble_v3_policy import SawyerDisassembleV3Policy as disassembly_policy

from metaworld.policies.compo_draweropen_pickplace_policy import CompoDrawerOpenPickPlacePolicy
from metaworld.policies.compo_dooropen_doorclose_policy import CompoDoorOpenDoorClosePolicy
import argparse

def render_episode(env_name,
                   out_path="out.gif",
                   episode_length=500,
                   image_size=(480, 480),
                   action_policy="random",
                   camera_name=["topview", "front", "gripperPOV"],
                   verbose=False,
                   env_kwargs=None):
    if env_kwargs is None:
        env_kwargs = {}
    multiple_cameras = None
    if len(camera_name) == 1:
        camera_name = camera_name[0]
        env = gym.make("Meta-World/MT1", env_name=env_name, render_mode="rgb_array", camera_name=camera_name, **env_kwargs)
        env = ProprioImageObsWrapper(env,
                                     image_height=image_size[0],
                                     image_width=image_size[1])
        multiple_cameras = False
    elif len(camera_name) > 1:
        env = gym.make("Meta-World/MT1", env_name=env_name, render_mode="rgb_array", **env_kwargs)
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
        elif env_name == "door-open-v3":
            policy = door_policy()
        elif env_name == "door-close-v3":
            policy = door_close_policy()
        elif env_name == "door-unlock-v3":
            policy = door_unlock_policy()
        elif env_name == "door-lock-v3":
            policy = door_lock_policy()
        elif env_name == "assembly-v3":
            policy = assembly_policy()
        elif env_name == "disassemble-v3":
            policy = disassembly_policy()
        elif env_name == "compo-draweropen-pickplace":
            policy = CompoDrawerOpenPickPlacePolicy()
        elif env_name == "compo-dooropen-doorclose":
            policy = CompoDoorOpenDoorClosePolicy()
        else:
            raise NotImplementedError(f"Policy for {env_name} is not implemented.")
    
    frames = []
    time_stamp = time.time()

    obs, info = env.reset()
    for t in range(episode_length):
        print(f"Step {t+1}/{episode_length}")
        if action_policy == "random":
            action = env.action_space.sample()
        elif action_policy == "policy":
            action = policy.get_action(obs["original_obs"])
        else:
            action = np.array([0.2, -0.2, 0.1, 0.1]) # Simple hardcoded action for testing
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
    
    print(f"Episode finished after {t+1} steps with reward {reward:.2f}.")
    print(f"Rendering episode to {out_path}...")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=15)
    print(f"Wrote {out_path} ({len(frames)} frames) in {time.time()-time_stamp:.2f} seconds.")

if __name__ == "__main__":
    
    # Mapping of task names to environment names
    TASK_MAPPING = {
        "pickplace": "pick-place-v3",
        "draweropen": "drawer-open-v3",
        "dooropen": "door-open-v3",
        "doorclose": "door-close-v3",
        "doorunlock": "door-unlock-v3",
        "doorlock": "door-lock-v3",
        "assembly": "assembly-v3",
        "disassemble": "disassemble-v3",
        "compo-draweropen-pickplace": "compo-draweropen-pickplace",
        "compo-dooropen-doorclose": "compo-dooropen-doorclose",
        "compo-pickplace": "compo-pickplace",
    }
    
    parser = argparse.ArgumentParser(description="Visualize Meta-World tasks")
    parser.add_argument("--tasks", nargs="+", required=True, help="Task names to visualize")
    parser.add_argument("--agent", nargs="+", default=["policy"], help="Agent type per task (policy/random)")
    parser.add_argument("--length", type=int, nargs="+", help="Episode length per task")
    parser.add_argument("--camera-name", nargs="+", default=["topview", "front", "gripperPOV"], help="Camera names")
    parser.add_argument("--env-kwargs", nargs="*", default=[], help="Extra env kwargs as key=value pairs (e.g. initialise_region=large)")
    
    args = parser.parse_args()
    
    # Parse env kwargs
    env_kwargs = {}
    for kv in args.env_kwargs:
        if "=" not in kv:
            parser.error(f"Invalid env kwarg format: {kv}. Expected key=value.")
        k, v = kv.split("=", 1)
        env_kwargs[k] = v
    
    # Handle defaults for length and agent
    if len(args.agent) == 1:
        args.agent = args.agent * len(args.tasks)
    if args.length is None:
        args.length = [200] * len(args.tasks)
    elif len(args.length) == 1:
        args.length = args.length * len(args.tasks)
    
    assert len(args.tasks) == len(args.agent), "Number of agents must match number of tasks"
    assert len(args.tasks) == len(args.length), "Number of lengths must match number of tasks"
    
    for task, agent, length in zip(args.tasks, args.agent, args.length):
        if task not in TASK_MAPPING:
            print(f"Unknown task: {task}")
            continue
        env_name = TASK_MAPPING[task]
        out_path = f"gifs/{task}_{agent}.gif"
        render_episode(env_name,
                       out_path=out_path,
                       episode_length=length,
                       action_policy=agent,
                       camera_name=args.camera_name,
                       env_kwargs=env_kwargs)
