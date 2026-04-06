import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["EGL_LOG_LEVEL"] = "fatal"
import warnings
warnings.filterwarnings("ignore", message="Constant.*may be too high")
import gymnasium as gym
gym.logger.min_level = gym.logger.ERROR
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
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy as reach_policy
from metaworld.policies.sawyer_reach_xy_v3_policy import SawyerReachXYV3Policy as reach_xy_policy
from metaworld.policies.sawyer_reach_xz_v3_policy import SawyerReachXZV3Policy as reach_xz_policy
from metaworld.policies.sawyer_reach_yz_v3_policy import SawyerReachYZV3Policy as reach_yz_policy
from metaworld.policies.sawyer_reach_xyz_v3_policy import SawyerReachXYZV3Policy as reach_xyz_policy
from metaworld.policies.sawyer_drawer_open_v3_policy import SawyerDrawerOpenV3Policy as drawer_policy
from metaworld.policies.sawyer_door_open_v3_policy import SawyerDoorOpenV3Policy as door_policy
from metaworld.policies.sawyer_door_close_v3_policy import SawyerDoorCloseV3Policy as door_close_policy
from metaworld.policies.sawyer_door_unlock_v3_policy import SawyerDoorUnlockV3Policy as door_unlock_policy
from metaworld.policies.sawyer_door_lock_v3_policy import SawyerDoorLockV3Policy as door_lock_policy
from metaworld.policies.sawyer_grasp_policy import SawyerGraspV3Policy as grasp_policy
from metaworld.policies.sawyer_assembly_v3_policy import SawyerAssemblyV3Policy as assembly_policy
from metaworld.policies.sawyer_disassemble_v3_policy import SawyerDisassembleV3Policy as disassembly_policy
from metaworld.policies.sawyer_coffee_pull_v3_policy import SawyerCoffeePullV3Policy as coffee_pull_policy
from metaworld.policies.sawyer_coffee_push_v3_policy import SawyerCoffeePushV3Policy as coffee_push_policy
from metaworld.policies.sawyer_coffee_button_v3_policy import SawyerCoffeeButtonV3Policy as coffee_button_policy
from metaworld.policies.sawyer_bin_picking_v3_policy import SawyerBinPickingV3Policy as bin_picking_policy
from metaworld.policies.sawyer_pick_place_block_v3_policy import SawyerPickPlaceBlockV3Policy as pick_place_block_policy
from metaworld.policies.sawyer_box_close_v3_policy import SawyerBoxCloseV3Policy as box_close_policy

from metaworld.policies.compo_draweropen_pickplace_policy import CompoDrawerOpenPickPlacePolicy
from metaworld.policies.compo_dooropen_doorclose_policy import CompoDoorOpenDoorClosePolicy
from metaworld.policies.compo_assembly_disassembly_policy import CompoAssemblyDisassemblyPolicy
from metaworld.policies.compo_coffee_push_button_pull_policy import CompoCoffeePushButtonPullPolicy
from metaworld.policies.compo_pickplace_block_policy import CompoPickPlaceBlockPolicy
from metaworld.policies.compo_pickplace_boxclose_policy import CompoPickPlaceBoxClosePolicy
import argparse


def _extract_step_success(info):
    """Return whether the current step indicates task success."""
    if not isinstance(info, dict):
        return False
    success_val = info.get("success", info.get("is_success", 0.0))
    if isinstance(success_val, (list, tuple, np.ndarray)):
        return np.any(np.asarray(success_val) > 0)
    try:
        return float(success_val) > 0
    except (TypeError, ValueError):
        return bool(success_val)

def render_episode(env_name,
                   out_path="out.gif",
                   episode_length=500,
                   image_size=(480, 480),
                   action_policy="random",
                   camera_name=["topview", "front", "gripperPOV"],
                   verbose=False,
                   env_kwargs=None,
                   plot_reward=False,
                   reward_plot_path=None,
                   reward_gif_path=None,
                   fps=15,
                   save_media=True,
                   save_on_failure=False,
                   episode_idx=None,
                   num_episodes=None):
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
        elif env_name in ("bin-picking-v3", "bin-picking-redblue-v3",
                          "bin-picking-yellowblue-v3", "bin-picking-redpurple-v3",
                          "bin-picking-yellowpurple-v3"):
            policy = bin_picking_policy()
        elif env_name == "door-open-v3":
            policy = door_policy()
        elif env_name == "box-close-v3":
            policy = box_close_policy()
        elif env_name == "door-close-v3":
            policy = door_close_policy()
        elif env_name == "door-unlock-v3":
            policy = door_unlock_policy()
        elif env_name == "door-lock-v3":
            policy = door_lock_policy()
        elif env_name == "grasp-v3":
            policy = grasp_policy()
        elif env_name == "assembly-v3":
            policy = assembly_policy()
        elif env_name == "reach-v3":
            policy = reach_policy()
        elif env_name == "reach-xy-v3":
            policy = reach_xy_policy()
        elif env_name == "reach-xz-v3":
            policy = reach_xz_policy()
        elif env_name == "reach-yz-v3":
            policy = reach_yz_policy()
        elif env_name == "reach-xyz-v3":
            policy = reach_xyz_policy()
        elif env_name == "disassemble-v3":
            policy = disassembly_policy()
        elif env_name == "coffee-pull-v3":
            policy = coffee_pull_policy()
        elif env_name == "coffee-push-v3":
            policy = coffee_push_policy()
        elif env_name == "coffee-button-v3":
            policy = coffee_button_policy()
        elif env_name in ("pick-place-block-v3", "pick-place-redblock-v3",
                          "pick-place-greenblock-v3"):
            policy = pick_place_block_policy()
        elif env_name == "compo-pickplace-block":
            policy = CompoPickPlaceBlockPolicy()
        elif env_name == "compo-draweropen-pickplace":
            policy = CompoDrawerOpenPickPlacePolicy()
        elif env_name == "compo-dooropen-doorclose":
            policy = CompoDoorOpenDoorClosePolicy()
        elif env_name == "compo-assembly-disassembly":
            policy = CompoAssemblyDisassemblyPolicy()
        elif env_name == "compo-coffeepushbuttonpull":
            policy = CompoCoffeePushButtonPullPolicy()
        elif env_name == "compo-pickplace-boxclose":
            policy = CompoPickPlaceBoxClosePolicy()
        else:
            raise NotImplementedError(f"Policy for {env_name} is not implemented.")
    
    collect_media = save_media or save_on_failure
    frames = [] if collect_media else None
    rewards = []
    time_stamp = time.time()

    obs, info = env.reset()
    episode_success = _extract_step_success(info)
    reward = 0.0
    for t in range(episode_length):
        episode_prefix = ""
        if episode_idx is not None and num_episodes is not None:
            episode_prefix = f"[Episode {episode_idx}/{num_episodes}] "
        print(f"{episode_prefix}Step {t+1}/{episode_length} | Reward: {reward:.2f}")
        if action_policy == "random":
            action = env.action_space.sample()
        elif action_policy == "policy":
            action = policy.get_action(obs["original_obs"])
        else:
            action = np.array([0.2, -0.2, 0.1, 0.1]) # Simple hardcoded action for testing
        obs, reward, terminated, truncated, info = env.step(action)
        episode_success = episode_success or _extract_step_success(info)
        rewards.append(reward)
        if terminated or truncated:
            if collect_media:
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
            break
        if verbose:
            print(f"EE Pos: {obs['proprio'][0]:.3f}, {obs['proprio'][1]:.3f}, {obs['proprio'][2]:.3f}, EE velocity: {obs['proprio'][3]:.3f}, {obs['proprio'][4]:.3f}, {obs['proprio'][5]:.3f}, Gripper Val: {obs['proprio'][6]:.3f}")
        if not collect_media:
            continue
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

    env.close()

    should_save_media = save_media or (save_on_failure and not episode_success)
    
    print(
        f"Episode finished after {t+1} steps with reward {reward:.2f}. "
        f"Success: {episode_success}."
    )
    if should_save_media:
        print(f"Rendering episode to {out_path}...")

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out_path, frames, fps=fps)
        print(f"Wrote {out_path} ({len(frames)} frames) in {time.time()-time_stamp:.2f} seconds.")
    else:
        print("Skipped gif/reward artifact save for this successful non-final episode.")

    if plot_reward and should_save_media:
        if reward_plot_path is None:
            out_path_obj = Path(out_path)
            reward_plot_path = str(out_path_obj.with_name(f"{out_path_obj.stem}_reward.png"))

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            steps = np.arange(1, len(rewards) + 1)
            plt.figure(figsize=(8, 4.5))
            plt.plot(steps, rewards, linewidth=2)
            plt.xlabel("Step")
            plt.ylabel("Reward")
            plt.title(f"Reward Curve: {env_name} ({action_policy})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            Path(reward_plot_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(reward_plot_path, dpi=150)
            plt.close()
            print(f"Wrote reward plot to {reward_plot_path}.")
        except ImportError:
            print("Could not generate reward plot because matplotlib is not installed.")

    return {
        "success": bool(episode_success),
        "steps": t + 1,
        "final_reward": float(reward),
        "saved_media": bool(should_save_media),
    }

if __name__ == "__main__":
    
    # Mapping of task names to environment names
    TASK_MAPPING = {
        "pnp": "pick-place-v3",
        "reach": "reach-v3",
        "reachxy": "reach-xy-v3",
        "reachxz": "reach-xz-v3",
        "reachyz": "reach-yz-v3",
        "reachxyz": "reach-xyz-v3",
        "binpnp": "bin-picking-v3",
        "binpnprb": "bin-picking-redblue-v3",
        "binpnpyb": "bin-picking-yellowblue-v3",
        "binpnprp": "bin-picking-redpurple-v3",
        "binpnpyp": "bin-picking-yellowpurple-v3",
        "boxclose": "box-close-v3",
        "draweropen": "drawer-open-v3",
        "dooropen": "door-open-v3",
        "doorclose": "door-close-v3",
        "doorunlock": "door-unlock-v3",
        "doorlock": "door-lock-v3",
        "grasp": "grasp-v3",
        "assemble": "assembly-v3",
        "disassemble": "disassemble-v3",
        "coffeepull": "coffee-pull-v3",
        "coffeepush": "coffee-push-v3",
        "coffeebutton": "coffee-button-v3",
        "pnpblock": "pick-place-block-v3",
        "pnpredblock": "pick-place-redblock-v3",
        "pnpgreenblock": "pick-place-greenblock-v3",
        "compodopnp": "compo-draweropen-pickplace",
        "compododc": "compo-dooropen-doorclose",
        "compoad": "compo-assembly-disassembly",
        "compopnpblock": "compo-pickplace-block",
        "compocoffee": "compo-coffeepushbuttonpull",
        "compopnpboxclose": "compo-pickplace-boxclose",
    }
    
    parser = argparse.ArgumentParser(description="Visualize Meta-World tasks")
    parser.add_argument("--tasks", nargs="+", required=True, help="Task names to visualize")
    parser.add_argument("--agent", nargs="+", default=["policy"], help="Agent type per task (policy/random)")
    parser.add_argument("--length", type=int, nargs="+", help="Episode length per task")
    parser.add_argument("--camera-name", nargs="+", default=["topview", "back", "gripperPOV"], help="Camera names")
    parser.add_argument("--frame-size", type=int, default=480, help="Square image size per frame in pixels")
    parser.add_argument("--env-kwargs", nargs="*", default=[], help="Extra env kwargs as key=value pairs (e.g. initialise_region=large)")
    parser.add_argument("--plot-reward", action="store_true", help="Save a reward-vs-step plot for each task")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to run per task")
    
    args = parser.parse_args()
    
    # Parse env kwargs
    env_kwargs = {}
    for kv in args.env_kwargs:
        if "=" not in kv:
            parser.error(f"Invalid env kwarg format: {kv}. Expected key=value.")
        k, v = kv.split("=", 1)
        env_kwargs[k] = v

    # Handle all tasks if "all" is specified
    if "all" in args.tasks:
        args.tasks = list(TASK_MAPPING.keys())

    if args.num_episodes < 1:
        parser.error("--num-episodes must be >= 1")
    if args.frame_size < 1:
        parser.error("--frame-size must be >= 1")
    
    # Handle defaults for length and agent
    if len(args.agent) == 1:
        args.agent = args.agent * len(args.tasks)
    if args.length is None:
        args.length = [500] * len(args.tasks)
    elif len(args.length) == 1:
        args.length = args.length * len(args.tasks)
    
    assert len(args.tasks) == len(args.agent), "Number of agents must match number of tasks"
    assert len(args.tasks) == len(args.length), "Number of lengths must match number of tasks"
    
    task_success_summary = {}

    for task, agent, length in zip(args.tasks, args.agent, args.length):
        if task not in TASK_MAPPING:
            print(f"Unknown task: {task}")
            continue
        env_name = TASK_MAPPING[task]
        successes = 0

        for episode_idx in range(1, args.num_episodes + 1):
            is_last_episode = episode_idx == args.num_episodes
            if args.num_episodes == 1:
                out_path = f"gifs/{task}_{agent}.gif"
                reward_plot_path = f"gifs/{task}_{agent}_reward.png"
            else:
                out_path = f"gifs/{task}_{agent}_ep{episode_idx:03d}.gif"
                reward_plot_path = f"gifs/{task}_{agent}_ep{episode_idx:03d}_reward.png"
            reward_gif_path = f"gifs/{task}_{agent}_reward.gif"

            result = render_episode(
                env_name,
                out_path=out_path,
                episode_length=length,
                image_size=(args.frame_size, args.frame_size),
                action_policy=agent,
                camera_name=args.camera_name,
                env_kwargs=env_kwargs,
                plot_reward=args.plot_reward,
                reward_plot_path=reward_plot_path,
                reward_gif_path=reward_gif_path,
                save_media=(args.num_episodes == 1) or is_last_episode,
                save_on_failure=args.num_episodes > 1,
                episode_idx=episode_idx,
                num_episodes=args.num_episodes,
            )

            if result["success"]:
                successes += 1

        task_success_summary[task] = {
            "successes": successes,
            "episodes": args.num_episodes,
        }

    if task_success_summary:
        print("\n=== Task Success Summary ===")
        for task, stats in task_success_summary.items():
            success_rate = 100.0 * stats["successes"] / stats["episodes"]
            print(
                f"{task}: {stats['successes']}/{stats['episodes']} "
                f"({success_rate:.1f}% success)"
            )
