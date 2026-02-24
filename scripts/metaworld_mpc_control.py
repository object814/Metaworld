#!/usr/bin/env python3
"""
MPC control of Metaworld pick-place-v3 using a trained VJEPA2 world model.

This script:
1. Creates two Metaworld environments (ground truth + interactive)
2. Rolls out an expert policy on the ground truth env to get a demo trajectory
3. Samples N intermediate goal frames from the expert trajectory
4. Iteratively uses CEM-based MPC with the trained VJEPA2 world model to
   control the interactive environment towards each goal frame
5. Reports success/failure and optionally saves a comparison GIF

Key differences from the DROID energy_landscape_example:
- Metaworld states = [ee_pos(3), ee_vel(3), gripper(1)], not [pos(3), euler(3), gripper(1)]
- Metaworld actions = simple state diffs (states[t+1] - states[t]),
  NOT rotation-based poses_to_diff used in DROID
- Pose update uses simple addition, not rotation composition

Usage:
    python scripts/metaworld_mpc_control.py \\
        --checkpoint third_party/vjepa2/train/latest.pt \\
        --save-gif scripts/gifs/mpc_control.gif
"""

import os

os.environ["MUJOCO_GL"] = "egl"

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import cv2
import gymnasium as gym
import imageio
import mujoco
import numpy as np
import torch
import torch.nn.functional as F

# ── path setup ───────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
VJEPA_DIR = BASE_DIR / "third_party" / "vjepa2"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(VJEPA_DIR))

import metaworld  # noqa: F401  (registers envs)
from metaworld.policies import SawyerPickPlaceV3Policy

from app.vjepa_droid.transforms import make_transforms
from app.vjepa_droid.utils import init_video_model
from src.utils.checkpoint_loader import robust_checkpoint_loader

# ── import CEM via importlib to avoid namespace collision with scripts/ ─────

_mpc_spec = importlib.util.spec_from_file_location(
    "vjepa2_mpc_utils",
    str(VJEPA_DIR / "scripts" / "utils" / "mpc_utils.py"),
)
_mpc_module = importlib.util.module_from_spec(_mpc_spec)
_mpc_spec.loader.exec_module(_mpc_module)
cem_fn = _mpc_module.cem


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="MPC control of Metaworld using VJEPA2 world model"
    )
    # Model / checkpoint
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained VJEPA2 checkpoint (.pt file with encoder/predictor)",
    )
    p.add_argument(
        "--encoder-key",
        type=str,
        default="target_encoder",
        help="Key for encoder weights in checkpoint (default: target_encoder for EMA)",
    )

    # Environment
    p.add_argument("--env-name", type=str, default="pick-place-v3")              # Metaworld task name (must match generate_vjepa_dataset.py --env-name)
    p.add_argument("--camera-name", type=str, default="topview")                 # MuJoCo camera used for observations (must match training data camera)
    p.add_argument("--image-size", type=int, default=224)                        # Rendered frame resolution (H=W), should match training crop_size
    p.add_argument("--episode-length", type=int, default=150)                    # Max steps for the expert rollout on the ground-truth env
    p.add_argument("--seed", type=int, default=42)                               # Random seed for env resets (both GT and interactive use the same seed)

    # Model architecture (should match training config in params-pretrain.yaml)
    p.add_argument("--crop-size", type=int, default=224)                         # Image crop size fed to ViT encoder (determines tokens_per_frame)
    p.add_argument("--patch-size", type=int, default=16)                         # ViT patch size; tokens_per_frame = (crop_size / patch_size)^2
    p.add_argument("--model-name", type=str, default="vit_giant_xformers")       # ViT encoder variant name (must match training config model.model_name)
    p.add_argument("--pred-depth", type=int, default=24)                         # Predictor transformer depth (number of layers)
    p.add_argument("--pred-embed-dim", type=int, default=1024)                   # Predictor hidden embedding dimension
    p.add_argument("--pred-num-heads", type=int, default=16)                     # Number of attention heads in the predictor
    p.add_argument("--max-num-frames", type=int, default=8)                      # Max temporal frames the model was trained with (data.dataset_fpcs)
    p.add_argument("--tubelet-size", type=int, default=2)                        # Temporal tubelet size for video tokenisation

    # Goal tracking
    p.add_argument("--num-goals", type=int, default=5)                           # Number of intermediate goal frames sampled from the expert trajectory
    p.add_argument(
        "--max-steps-per-goal", type=int, default=60,                            # Max env steps allowed per subgoal before moving on (prevents getting stuck)
        help="Max env steps before giving up on a subgoal",
    )
    p.add_argument(
        "--threshold", type=float, default=0.15,                                 # L1 rep distance below which a goal is considered reached
        help="Mean L1 representation distance to consider a goal reached",
    )

    # CEM planner
    p.add_argument("--cem-rollout", type=int, default=15)                        # MPC planning horizon (number of future actions to plan)
    p.add_argument("--cem-samples", type=int, default=200)                       # Number of action trajectories sampled per CEM iteration
    p.add_argument("--cem-topk", type=int, default=20)                           # Top-k trajectories kept to refit the CEM distribution
    p.add_argument("--cem-steps", type=int, default=5)                           # Number of CEM optimisation iterations per planning step
    p.add_argument("--cem-maxnorm", type=float, default=0.075)                   # Max magnitude of sampled xyz action deltas (clips to [-maxnorm, maxnorm])
    p.add_argument("--cem-momentum-mean", type=float, default=0.15)              # Momentum for updating the CEM mean (0=full update, 1=no update)
    p.add_argument("--cem-momentum-std", type=float, default=0.75)               # Momentum for updating the CEM std (higher=more conservative shrinkage)

    # Runtime
    p.add_argument("--device", type=str, default="cuda")                         # Torch device for model inference ("cuda" or "cpu")
    p.add_argument("--dtype", type=str, default="float16", choices=["bfloat16", "float16", "float32"])   # Model precision (bfloat16 recommended for speed)
    p.add_argument("--save-gif", type=str, default=None, help="Path to save comparison GIF")             # If set, saves a side-by-side [Expert | MPC] GIF
    p.add_argument("--verbose", action="store_true")                             # Print per-step distance and timing info
    return p.parse_args()


# ── environment helpers ──────────────────────────────────────────────────────


def make_env(env_name: str, image_size: int, seed: int):
    """Create a Metaworld environment."""
    env = gym.make(
        "Meta-World/MT1",
        env_name=env_name,
        render_mode="rgb_array",
        camera_name="topview",
    )
    env.reset(seed=seed)
    return env


def render_camera(
    env: gym.Env, camera_name: str, height: int, width: int
) -> np.ndarray:
    """Render a named camera from the MuJoCo model. Returns uint8 (H, W, 3)."""
    model = env.unwrapped.model
    data = env.unwrapped.data
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam_id = None
    for cid in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cid)
        if name == camera_name:
            cam_id = cid
            break
    if cam_id is None:
        raise ValueError(f"Camera '{camera_name}' not found in model")
    renderer.update_scene(data, camera=cam_id)
    img = renderer.render().copy()
    try:
        renderer.close()
    except Exception:
        pass
    return img.astype(np.uint8)


def get_proprio(obs_raw: np.ndarray) -> np.ndarray:
    """Extract proprio = [ee_pos(3), ee_vel(3), gripper(1)] from raw Metaworld obs."""
    hand_pos = obs_raw[:3].astype(np.float32)
    hand_vel = obs_raw[18:21].astype(np.float32)
    gripper = np.array([obs_raw[3]], dtype=np.float32)
    return np.concatenate([hand_pos, hand_vel, gripper])


def get_original_obs(obs_raw: np.ndarray) -> np.ndarray:
    """Build the observation vector expected by Metaworld scripted policies."""
    return np.concatenate([obs_raw[:18], obs_raw[21:39], obs_raw[42:]])


# ── Metaworld-adapted world model ───────────────────────────────────────────


def compute_new_pose_metaworld(pose: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    Simple additive pose update for Metaworld.

    Unlike DROID's compute_new_pose (which uses rotation composition for
    dims 3:6), Metaworld stores ee_velocity in those dims and the training
    data uses simple state differences, so we use simple addition.

    Args:
        pose:   (B, 1, 7)
        action: (B, 1, 7)
    Returns:
        (B, 1, 7)
    """
    new_pose = pose + action
    # Clamp gripper closedness to [0, 1]
    new_pose = new_pose.clone()
    new_pose[:, :, -1:] = torch.clamp(new_pose[:, :, -1:], 0.0, 1.0)
    return new_pose


class MetaworldWorldModel:
    """
    World model wrapper for Metaworld, adapted from the DROID WorldModel.

    Key difference: uses additive pose updates instead of rotation-based ones,
    matching how the Metaworld data loader computes actions (simple state diffs).
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        predictor: torch.nn.Module,
        tokens_per_frame: int,
        transform,
        mpc_args: dict,
        normalize_reps: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.encoder = encoder
        self.predictor = predictor
        self.tokens_per_frame = tokens_per_frame
        self.transform = transform
        self.mpc_args = mpc_args
        self.normalize_reps = normalize_reps
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def encode_frame(self, image: np.ndarray) -> torch.Tensor:
        """
        Encode a single image frame.

        Args:
            image: (H, W, 3) uint8 numpy array
        Returns:
            (1, tokens_per_frame, D) float tensor on self.device
        """
        # transform expects (T, H, W, 3), returns (C, T, H, W)
        clip = np.expand_dims(image, axis=0)  # (1, H, W, 3)
        clip = self.transform(clip)[None, :]  # (1, C, 1, H, W)
        B, C, T, H, W = clip.size()
        # The encoder expects at least 2 temporal frames; duplicate the single frame
        clip = (
            clip.permute(0, 2, 1, 3, 4)      # (B, T, C, H, W)
            .flatten(0, 1)                     # (B*T, C, H, W)
            .unsqueeze(2)                      # (B*T, C, 1, H, W)
            .repeat(1, 1, 2, 1, 1)             # (B*T, C, 2, H, W)
        )
        clip = clip.to(self.device, dtype=self.dtype, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=self.dtype):
            h = self.encoder(clip)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
        if self.normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))
        return h  # (1, tokens_per_frame, D)

    @torch.no_grad()
    def infer_next_action(
        self,
        context_rep: torch.Tensor,
        context_pose: torch.Tensor,
        goal_rep: torch.Tensor,
        close_gripper: int | None = None,
    ) -> torch.Tensor:
        """
        Use CEM to find the best action sequence.

        Args:
            context_rep:  (1, 1, tokens_per_frame, D) - current frame representation
            context_pose: (1, 1, 7) - current robot state
            goal_rep:     (1, 1, tokens_per_frame, D) - goal frame representation
            close_gripper: step index at which to force gripper close (None = don't force)
        Returns:
            (rollout, 7) - planned action sequence
        """
        tpf = self.tokens_per_frame

        def step_predictor(reps, actions, poses):
            B, T, N_T, D = reps.size()
            reps_flat = reps.flatten(1, 2)  # (B, T*N_T, D)
            with torch.cuda.amp.autocast(dtype=self.dtype):
                next_rep = self.predictor(reps_flat, actions, poses)[:, -tpf:]
            if self.normalize_reps:
                next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))
            next_rep = next_rep.view(B, 1, N_T, D)
            next_pose = compute_new_pose_metaworld(poses[:, -1:], actions[:, -1:])
            return next_rep, next_pose

        planned = cem_fn(
            context_frame=context_rep,
            context_pose=context_pose,
            goal_frame=goal_rep,
            world_model=step_predictor,
            close_gripper=close_gripper,
            **self.mpc_args,
        )
        return planned[0]  # (rollout, 7)

    def rep_distance(self, rep_a: torch.Tensor, rep_b: torch.Tensor) -> float:
        """Mean L1 distance between two representations."""
        return torch.mean(torch.abs(rep_a - rep_b)).item()


# ── ground truth rollout ─────────────────────────────────────────────────────


def rollout_expert(
    env: gym.Env,
    policy,
    episode_length: int,
    camera_name: str,
    image_size: int,
    seed: int,
) -> dict:
    """
    Roll out the expert policy and record frames + proprios.

    Returns dict with:
        frames:   (T+1, H, W, 3) uint8
        proprios: (T+1, 7) float32
        actions:  (T, 4) float32
        success:  bool
    """
    frames, proprios, actions = [], [], []

    obs_raw, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()

    frames.append(render_camera(env, camera_name, image_size, image_size))
    proprios.append(get_proprio(obs_raw))
    original_obs = get_original_obs(obs_raw)

    for t in range(episode_length):
        action = policy.get_action(original_obs)
        obs_raw, reward, terminated, truncated, info = env.step(action)

        frames.append(render_camera(env, camera_name, image_size, image_size))
        proprios.append(get_proprio(obs_raw))
        actions.append(action.astype(np.float32))
        original_obs = get_original_obs(obs_raw)

        if terminated or truncated:
            break

    return {
        "frames": np.stack(frames),
        "proprios": np.stack(proprios),
        "actions": np.stack(actions),
        "success": bool(info.get("success", False)),
    }


# ── model loading ───────────────────────────────────────────────────────────


def load_model(args):
    """Initialize model architecture and load trained checkpoint weights."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map[args.dtype]

    # Determine tokens per frame
    tokens_per_frame = (args.crop_size // args.patch_size) ** 2
    print(f"Tokens per frame: {tokens_per_frame}")

    # Initialize model architecture (must match training config)
    encoder, predictor = init_video_model(
        device=device,
        patch_size=args.patch_size,
        max_num_frames=args.max_num_frames,
        tubelet_size=args.tubelet_size,
        model_name=args.model_name,
        crop_size=args.crop_size,
        pred_depth=args.pred_depth,
        pred_num_heads=args.pred_num_heads,
        pred_embed_dim=args.pred_embed_dim,
        uniform_power=True,
        use_sdpa=True,
        use_rope=True,
        use_silu=False,
        use_pred_silu=False,
        wide_silu=False,
        pred_is_frame_causal=True,
        use_activation_checkpointing=False,  # not needed for inference
        use_extrinsics=False,
    )

    # Load checkpoint
    ckpt_path = str(Path(args.checkpoint).resolve())
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = robust_checkpoint_loader(ckpt_path, map_location="cpu")
    epoch = checkpoint.get("epoch", "?")
    print(f"Checkpoint epoch: {epoch}")

    # Load encoder weights (prefer target_encoder / EMA version)
    encoder_key = args.encoder_key
    if encoder_key in checkpoint:
        enc_dict = checkpoint[encoder_key]
    elif "encoder" in checkpoint:
        enc_dict = checkpoint["encoder"]
        print(f"Warning: '{encoder_key}' not found, falling back to 'encoder'")
    else:
        raise KeyError(f"No encoder weights found in checkpoint (tried '{encoder_key}', 'encoder')")

    enc_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in enc_dict.items()}
    msg = encoder.load_state_dict(enc_dict, strict=False)
    print(f"Encoder loaded: {msg}")

    # Load predictor weights
    pred_dict = checkpoint["predictor"]
    pred_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in pred_dict.items()}
    msg = predictor.load_state_dict(pred_dict, strict=False)
    print(f"Predictor loaded: {msg}")

    del checkpoint

    # Set to eval mode and cast to target dtype
    encoder.eval().to(model_dtype)
    predictor.eval().to(model_dtype)
    print(f"Models on {device} in {model_dtype}")

    return encoder, predictor, tokens_per_frame, device


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    # Determine dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # ── 1. Load model ────────────────────────────────────────────────────
    print("=" * 60)
    print("Loading VJEPA2 world model...")
    print("=" * 60)

    encoder, predictor, tokens_per_frame, device = load_model(args)

    # Create inference-time transform (no augmentation)
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=args.crop_size,
    )

    # Create world model wrapper
    mpc_args = {
        "rollout": args.cem_rollout,
        "samples": args.cem_samples,
        "topk": args.cem_topk,
        "cem_steps": args.cem_steps,
        "momentum_mean": args.cem_momentum_mean,
        "momentum_mean_gripper": args.cem_momentum_mean,
        "momentum_std": args.cem_momentum_std,
        "momentum_std_gripper": 0.15,
        "maxnorm": args.cem_maxnorm,
        "verbose": args.verbose,
    }
    world_model = MetaworldWorldModel(
        encoder=encoder,
        predictor=predictor,
        tokens_per_frame=tokens_per_frame,
        transform=transform,
        mpc_args=mpc_args,
        normalize_reps=True,
        device=str(device),
        dtype=dtype,
    )

    # ── 2. Create environments ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Setting up environments...")
    print("=" * 60)
    gt_env = make_env(args.env_name, args.image_size, args.seed)
    interactive_env = make_env(args.env_name, args.image_size, args.seed)

    # ── 3. Roll out expert policy on ground-truth env ────────────────────
    print("\n" + "=" * 60)
    print("Rolling out expert policy on ground truth environment...")
    print("=" * 60)
    policy = SawyerPickPlaceV3Policy()
    gt_data = rollout_expert(
        gt_env, policy, args.episode_length,
        args.camera_name, args.image_size, args.seed,
    )
    gt_frames = gt_data["frames"]      # (T+1, H, W, 3)
    gt_proprios = gt_data["proprios"]  # (T+1, 7)
    gt_len = len(gt_frames)
    print(f"Expert rollout: {gt_len} frames, success={gt_data['success']}")

    # ── 4. Sample intermediate goal frames ───────────────────────────────
    # Evenly spaced through the trajectory (excluding first frame)
    goal_indices = np.linspace(
        gt_len // (args.num_goals + 1),
        gt_len - 1,
        args.num_goals,
        dtype=int,
    )
    print(f"\nGoal frame indices: {goal_indices.tolist()} (out of {gt_len})")

    goal_frames = gt_frames[goal_indices]     # (num_goals, H, W, 3)
    goal_proprios = gt_proprios[goal_indices]  # (num_goals, 7)

    # Pre-encode all goal frames on GPU
    print("Encoding goal frames...")
    goal_reps = []
    for i, gf in enumerate(goal_frames):
        rep = world_model.encode_frame(gf)  # (1, tokens_per_frame, D)
        goal_reps.append(rep)
        print(f"  Goal {i}: frame idx {goal_indices[i]}, "
              f"ee_pos=({goal_proprios[i, 0]:.3f}, {goal_proprios[i, 1]:.3f}, {goal_proprios[i, 2]:.3f})")

    # ── 5. MPC control loop ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Starting MPC control loop...")
    print("=" * 60)

    # Reset interactive env with same seed
    obs_raw, info = interactive_env.reset(seed=args.seed)
    current_proprio = get_proprio(obs_raw)

    # Storage for visualisation
    mpc_frames = []
    total_steps = 0
    goals_reached = 0

    for goal_idx in range(args.num_goals):
        goal_rep = goal_reps[goal_idx].unsqueeze(1)  # (1, 1, tokens_per_frame, D)
        goal_frame = goal_frames[goal_idx]

        print(f"\n--- Goal {goal_idx + 1}/{args.num_goals} "
              f"(frame {goal_indices[goal_idx]}) ---")

        reached = False
        for step in range(args.max_steps_per_goal):
            t0 = time.time()

            # Render current observation
            current_frame = render_camera(
                interactive_env, args.camera_name,
                args.image_size, args.image_size,
            )
            mpc_frames.append(current_frame)

            # Encode current frame
            current_rep = world_model.encode_frame(current_frame)  # (1, tpf, D)

            # Check distance to goal
            dist = world_model.rep_distance(current_rep, goal_reps[goal_idx])

            if args.verbose or step % 5 == 0:
                ee = current_proprio[:3]
                print(
                    f"  step {step:3d} | dist={dist:.4f} | "
                    f"ee=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}) | "
                    f"dt={time.time() - t0:.2f}s"
                )

            if dist < args.threshold:
                print(f"  -> Goal {goal_idx + 1} REACHED at step {step} (dist={dist:.4f})")
                reached = True
                goals_reached += 1
                total_steps += step
                break

            # Prepare inputs for CEM
            ctx_rep = current_rep.unsqueeze(1)  # (1, 1, tpf, D)
            ctx_pose = (
                torch.from_numpy(current_proprio)
                .to(device, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )  # (1, 1, 7)

            # Run CEM planner
            planned_actions = world_model.infer_next_action(
                ctx_rep, ctx_pose, goal_rep
            )  # (rollout, 7)

            # Execute only the first planned action (receding horizon)
            first_action_7d = planned_actions[0].cpu().numpy()  # (7,)

            # Map world-model action (7D) to Metaworld env action (4D):
            #   env_action = [delta_x, delta_y, delta_z, gripper]
            env_action = np.array([
                first_action_7d[0],  # delta x
                first_action_7d[1],  # delta y
                first_action_7d[2],  # delta z
                first_action_7d[6],  # gripper command
            ], dtype=np.float32)

            # Clamp to env action range
            env_action = np.clip(env_action, -1.0, 1.0)

            # Step the interactive environment
            obs_raw, reward, terminated, truncated, info = interactive_env.step(env_action)
            current_proprio = get_proprio(obs_raw)

        if not reached:
            total_steps += args.max_steps_per_goal
            print(f"  -> Goal {goal_idx + 1} NOT reached (max steps exceeded, dist={dist:.4f})")

    # Capture final frame
    final_frame = render_camera(
        interactive_env, args.camera_name,
        args.image_size, args.image_size,
    )
    mpc_frames.append(final_frame)

    # ── 6. Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Goals reached: {goals_reached}/{args.num_goals}")
    print(f"Total MPC steps: {total_steps}")
    print(f"Task success (env info): {info.get('success', False)}")

    # ── 7. Save comparison GIF ───────────────────────────────────────────
    if args.save_gif:
        save_path = Path(args.save_gif)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Build side-by-side frames: [GT | goal markers | MPC]
        combined_frames = []
        n_mpc = len(mpc_frames)
        n_gt = len(gt_frames)

        # Subsample to match lengths
        max_frames = max(n_mpc, n_gt)
        min_len = min(n_mpc, n_gt)

        for i in range(min_len):
            gt_idx = min(i, n_gt - 1)
            mpc_idx = min(i, n_mpc - 1)
            gt_f = gt_frames[gt_idx]
            mpc_f = mpc_frames[mpc_idx]

            # Green border on goal frames in GT
            is_goal = gt_idx in goal_indices
            if is_goal:
                gt_f = gt_f.copy()
                gt_f[:5, :, :] = [0, 255, 0]
                gt_f[-5:, :, :] = [0, 255, 0]
                gt_f[:, :5, :] = [0, 255, 0]
                gt_f[:, -5:, :] = [0, 255, 0]

            # Separator
            sep = np.ones((args.image_size, 4, 3), dtype=np.uint8) * 128

            combined = np.concatenate([gt_f, sep, mpc_f], axis=1)
            combined_frames.append(combined)

        imageio.mimsave(str(save_path), combined_frames, fps=15, loop=0)
        print(f"\nComparison GIF saved to {save_path}")
        print(f"  Layout: [Expert (left) | MPC (right)], {len(combined_frames)} frames")

    gt_env.close()
    interactive_env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
