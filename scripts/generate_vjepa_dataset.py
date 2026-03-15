"""
Generate a DROID-like dataset from Metaworld environments.

This script rolls out a scripted policy in a Metaworld environment and saves
the resulting demonstrations in a format that mirrors the DROID dataset
structure, so that the VJEPA training pipeline can consume them with minimal
changes.

Dataset layout (mirrors DROID):
    <output_dir>/
        episodes.csv              # one line per episode directory
        episode_0000/
            metadata.json         # camera view → path mapping
            trajectory.h5         # robot states, actions, camera extrinsics
            recordings/MP4/
                topview.mp4
                front.mp4
                gripperPOV.mp4
        episode_0001/
        ...

Usage:
    python scripts/generate_vjepa_dataset.py \
        --num-episodes 100 \
        --episode-length 150 \
        --output-dir datasets/metaworld_pickplace \
        --env-name pick-place-v3 \
        --camera-names topview front gripperPOV \
        --random-action-ratio 0.3 \
        --image-size 224 \
        --fps 15 \
        --seed 42
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import gymnasium as gym
import h5py
import numpy as np
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import metaworld  # noqa: F401  (registers envs)
from metaworld.policies import (
    SawyerPickPlaceV3Policy,
    SawyerDrawerOpenV3Policy,
    SawyerDrawerCloseV3Policy,
    SawyerDoorOpenV3Policy,
    SawyerDoorCloseV3Policy,
    SawyerButtonPressV3Policy,
    CompoDoorOpenDoorClosePolicy,
)
from metaworld.policies.compo_draweropen_pickplace_policy import CompoDrawerOpenPickPlacePolicy

# ── helpers ──────────────────────────────────────────────────────────────────

ENV_TO_POLICY = {
    "pick-place-v3": SawyerPickPlaceV3Policy,
    "drawer-open-v3": SawyerDrawerOpenV3Policy,
    "drawer-close-v3": SawyerDrawerCloseV3Policy,
    "door-open-v3": SawyerDoorOpenV3Policy,
    "door-close-v3": SawyerDoorCloseV3Policy,
    "compo-dooropen-doorclose": CompoDoorOpenDoorClosePolicy,
    "compo-draweropen-pickplace": CompoDrawerOpenPickPlacePolicy,
}


def _get_camera_extrinsics(env: gym.Env, camera_name: str) -> np.ndarray:
    """Return a 7-D vector (pos3 + quat4) for the named camera from MuJoCo."""
    model = env.unwrapped.model
    data = env.unwrapped.data
    cam_id = None
    for cid in range(model.ncam):
        import mujoco
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cid)
        if name == camera_name:
            cam_id = cid
            break
    if cam_id is None:
        return np.zeros(7, dtype=np.float64)
    pos = data.cam_xpos[cam_id].copy()          # (3,)
    rotmat = data.cam_xmat[cam_id].reshape(3, 3)
    # Convert rotation matrix → quaternion (w, x, y, z)
    from scipy.spatial.transform import Rotation as R
    quat = R.from_matrix(rotmat).as_quat()       # (x, y, z, w)
    quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # → (w, x, y, z)
    return np.concatenate([pos, quat])             # (7,)


def _make_env_and_policy(
    env_name: str,
    image_size: int,
    seed: int | None = None,
    policy_type: str = "expert",
):
    """Create a Metaworld env (single-cam, rgb_array) and optional scripted policy."""
    # We render with a dummy single camera; actual multi-cam frames are grabbed manually.
    env = gym.make(
        "Meta-World/MT1",
        env_name=env_name,
        render_mode="rgb_array",
        camera_name="topview",   # default cam for render()
    )
    if seed is not None:
        env.reset(seed=seed)

    if policy_type == "expert":
        policy_cls = ENV_TO_POLICY.get(env_name)
        if policy_cls is None:
            raise NotImplementedError(
                f"No scripted policy for '{env_name}'. "
                f"Available: {list(ENV_TO_POLICY.keys())}"
            )
        return env, policy_cls()

    if policy_type == "random":
        return env, None

    raise ValueError(f"Unknown policy type '{policy_type}'. Expected 'expert' or 'random'.")


def _render_camera(env: gym.Env, camera_name: str, height: int, width: int) -> np.ndarray:
    """Render a single camera at the requested resolution.  Returns uint8 (H,W,3)."""
    import mujoco
    model = env.unwrapped.model
    data = env.unwrapped.data
    renderer = mujoco.Renderer(model, height=height, width=width)
    # find cam id
    cam_id = None
    for cid in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cid)
        if name == camera_name:
            cam_id = cid
            break
    if cam_id is None:
        raise ValueError(f"Camera '{camera_name}' not found in model.")
    renderer.update_scene(data, camera=cam_id)
    img = renderer.render().copy()
    try:
        renderer.close()
    except Exception:
        pass
    return img.astype(np.uint8)


# ── main collection loop ────────────────────────────────────────────────────

def collect_episode(
    env: gym.Env,
    policy,
    episode_length: int,
    camera_names: list[str],
    image_size: int,
    policy_type: str = "expert",
    seed: int | None = None,
) -> dict:
    """Roll out one episode.  Returns a dict of arrays."""
    # storage
    frames = {cam: [] for cam in camera_names}
    proprios = []     # (T, 7)  ee_pos(3) + ee_vel(3) + gripper(1)
    actions_raw = []  # (T, 4)  raw Metaworld actions
    extrinsics = {cam: [] for cam in camera_names}  # (T, 7)

    obs_raw, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    # Extract proprio from raw obs
    hand_pos = obs_raw[:3].astype(np.float32)
    hand_vel = obs_raw[18:21].astype(np.float32)
    gripper = np.array([obs_raw[3]], dtype=np.float32)
    proprio = np.concatenate([hand_pos, hand_vel, gripper])
    proprios.append(proprio)

    # Build original_obs the same way the wrapper does (remove velocity duplicate)
    original_obs = np.concatenate([obs_raw[:18], obs_raw[21:39], obs_raw[42:]])

    # grab initial frames and extrinsics
    for cam in camera_names:
        frames[cam].append(_render_camera(env, cam, image_size, image_size))
        extrinsics[cam].append(_get_camera_extrinsics(env, cam))

    for t in range(episode_length):
        if policy_type == "expert":
            action = policy.get_action(original_obs)
        elif policy_type == "random":
            action = env.action_space.sample()
        else:
            raise ValueError(f"Unknown policy type '{policy_type}'.")

        obs_raw, reward, terminated, truncated, info = env.step(action)

        hand_pos = obs_raw[:3].astype(np.float32)
        hand_vel = obs_raw[18:21].astype(np.float32)
        gripper = np.array([obs_raw[3]], dtype=np.float32)
        proprio = np.concatenate([hand_pos, hand_vel, gripper])
        proprios.append(proprio)

        original_obs = np.concatenate([obs_raw[:18], obs_raw[21:39], obs_raw[42:]])
        actions_raw.append(action.astype(np.float32))

        for cam in camera_names:
            frames[cam].append(_render_camera(env, cam, image_size, image_size))
            extrinsics[cam].append(_get_camera_extrinsics(env, cam))

        if terminated or truncated:
            break

    return {
        "frames": {cam: np.stack(frames[cam]) for cam in camera_names},  # (T+1, H, W, 3) per cam
        "proprios": np.stack(proprios),        # (T+1, 7)
        "actions": np.stack(actions_raw),      # (T, 4)
        "extrinsics": {cam: np.stack(extrinsics[cam]) for cam in camera_names},  # (T+1, 7) per cam
        "success": bool(info.get("success", False)),
    }


def save_episode(episode_data: dict, episode_dir: Path, camera_names: list[str], fps: int):
    """Persist one episode to disk in DROID-like layout."""
    episode_dir.mkdir(parents=True, exist_ok=True)
    mp4_dir = episode_dir / "recordings" / "MP4"
    mp4_dir.mkdir(parents=True, exist_ok=True)

    proprios = episode_data["proprios"]   # (T+1, 7)
    actions_raw = episode_data["actions"] # (T, 4)
    T = proprios.shape[0]

    # ── metadata.json ────────────────────────────────────────────────────
    metadata = {}
    for cam in camera_names:
        mp4_rel = f"recordings/MP4/{cam}.mp4"
        metadata[cam] = mp4_rel
    with open(episode_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ── write MP4 videos ─────────────────────────────────────────────────
    for cam in camera_names:
        vid_path = str(mp4_dir / f"{cam}.mp4")
        frames = episode_data["frames"][cam]  # (T, H, W, 3) uint8 RGB
        h, w = frames.shape[1], frames.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    # ── trajectory.h5 ────────────────────────────────────────────────────
    # State format matching DROID:
    #   cartesian_position (T, 6): ee_pos(3) + ee_vel(3)
    #   gripper_position   (T,) : scalar gripper state
    #
    # We also store the raw 4-D Metaworld actions for reference and
    # padded 7-D actions for direct VJEPA consumption.
    with h5py.File(str(episode_dir / "trajectory.h5"), "w") as hf:
        obs_grp = hf.create_group("observation")
        rs_grp = obs_grp.create_group("robot_state")
        rs_grp.create_dataset("cartesian_position", data=proprios[:, :6])  # (T, 6)
        rs_grp.create_dataset("gripper_position", data=proprios[:, 6])     # (T,)

        ext_grp = obs_grp.create_group("camera_extrinsics")
        for cam in camera_names:
            ext_grp.create_dataset(cam, data=episode_data["extrinsics"][cam])  # (T, 7)

        # Raw Metaworld actions (T-1, 4)
        hf.create_dataset("action_raw", data=actions_raw)
        # Padded 7-D actions (T-1, 7): delta_pos(3) + zeros(3) + gripper(1)
        padded_actions = np.zeros((actions_raw.shape[0], 7), dtype=np.float32)
        padded_actions[:, :3] = actions_raw[:, :3]
        padded_actions[:, 6] = actions_raw[:, 3]
        hf.create_dataset("action_padded", data=padded_actions)

        hf.attrs["fps"] = fps
        hf.attrs["episode_length"] = T


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate a DROID-like dataset from Metaworld.")
    p.add_argument("--num-episodes", type=int, default=100,
                   help="Total number of episodes to collect.")
    p.add_argument("--episode-length", type=int, default=150,
                   help="Max steps per episode.")
    p.add_argument("--output-dir", type=str, default="datasets/metaworld_pickplace",
                   help="Root directory for the dataset.")
    p.add_argument("--env-name", type=str, default="pick-place-v3",
                   help="Metaworld environment name.")
    p.add_argument("--camera-names", nargs="+", default=["topview", "back", "gripperPOV"],
                   help="Camera names to render.")
    p.add_argument("--random-action-ratio", type=float, default=0.0,
                   help="Fraction of episodes that use random policy (0.0=all expert, 1.0=all random).")
    p.add_argument("--image-size", type=int, default=224,
                   help="Height and width of rendered frames.")
    p.add_argument("--fps", type=int, default=15,
                   help="Frames per second for saved videos.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed.")
    p.add_argument("--only-successful", action="store_true",
                   help="If set, only keep episodes where the task was solved.")
    return p.parse_args()


def _discover_existing_episode_dirs(output_dir: Path) -> list[Path]:
    """Return existing episode directories sorted by numeric suffix."""
    episode_dirs = []
    for p in output_dir.iterdir():
        if not p.is_dir() or not p.name.startswith("episode_"):
            continue
        suffix = p.name.split("episode_", 1)[1]
        if suffix.isdigit():
            episode_dirs.append(p)
    episode_dirs.sort(key=lambda p: int(p.name.split("episode_", 1)[1]))
    return episode_dirs


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover existing episodes and rebuild/fill index CSV so interrupted runs
    # can be resumed safely.
    existing_episode_dirs = _discover_existing_episode_dirs(output_dir)
    csv_path = output_dir / "episodes.csv"

    all_episode_dirs: list[str] = [str(p.resolve()) for p in existing_episode_dirs]
    with open(csv_path, "w") as f:
        for d in all_episode_dirs:
            f.write(d + "\n")

    next_episode_idx = 0
    if existing_episode_dirs:
        next_episode_idx = int(existing_episode_dirs[-1].name.split("episode_", 1)[1]) + 1

    ratio = args.random_action_ratio
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("--random-action-ratio must be between 0.0 and 1.0")

    # Create env; also create expert policy unless every episode is random.
    if ratio < 1.0:
        env, expert_policy = _make_env_and_policy(
            args.env_name, args.image_size, seed=args.seed, policy_type="expert",
        )
    else:
        env, expert_policy = _make_env_and_policy(
            args.env_name, args.image_size, seed=args.seed, policy_type="random",
        )

    collected = len(existing_episode_dirs)
    attempted = len(existing_episode_dirs)
    t0 = time.time()

    if collected > 0:
        print(f"Found {collected} existing episodes in {output_dir}; resuming collection.")

    if collected >= args.num_episodes:
        env.close()
        print(
            f"Requested {args.num_episodes} episodes, but {collected} already exist. "
            f"Index CSV has been rebuilt at {csv_path}."
        )
        return

    # Build a schedule of policy types for the remaining episodes.
    remaining = args.num_episodes - collected
    num_random = round(remaining * ratio)
    num_expert = remaining - num_random
    episode_schedule = ["expert"] * num_expert + ["random"] * num_random
    schedule_idx = 0

    print(f"Remaining {remaining} episodes: {num_expert} expert, {num_random} random")

    with tqdm(
        total=args.num_episodes,
        initial=collected,
        desc="Collecting episodes",
        unit="ep",
    ) as pbar:
        while collected < args.num_episodes:
            ep_seed = args.seed + attempted
            attempted += 1

            current_policy_type = episode_schedule[schedule_idx]
            policy = expert_policy if current_policy_type == "expert" else None

            episode_data = collect_episode(
                env,
                policy,
                args.episode_length,
                args.camera_names,
                args.image_size,
                policy_type=current_policy_type,
                seed=ep_seed,
            )

            if args.only_successful and not episode_data["success"]:
                continue

            ep_name = f"episode_{next_episode_idx:05d}"
            ep_dir = output_dir / ep_name
            save_episode(episode_data, ep_dir, args.camera_names, args.fps)
            episode_path = str(ep_dir.resolve())
            all_episode_dirs.append(episode_path)
            with open(csv_path, "a") as f:
                f.write(episode_path + "\n")
            collected += 1
            next_episode_idx += 1
            schedule_idx += 1

            elapsed = time.time() - t0
            rate = collected / elapsed if elapsed > 0 else 0
            pbar.update(1)
            pbar.set_postfix(
                saved=ep_name,
                policy=current_policy_type,
                success=episode_data["success"],
                length=episode_data["proprios"].shape[0],
                rate=f"{rate:.1f} ep/s",
            )

    env.close()
    total = time.time() - t0
    print(f"\nDone. {collected} episodes saved to {output_dir}  ({total:.1f}s total)")
    print(f"Index CSV: {csv_path}")


if __name__ == "__main__":
    main()
