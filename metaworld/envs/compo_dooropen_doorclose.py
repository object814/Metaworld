from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils


class CompoDoorOpenDoorCloseEnv(SawyerXYZEnv):
    """Sawyer Compositional Door-Open then Door-Close Environment.

    Phase 1: Open the door (swing the hinge from closed → fully open).
    Phase 2: Close the door (push the door back from open → closed).

    Reward design
    -------------
    Each phase produces a reward in [0, 10].
    Phase 1 reward is taken directly from SawyerDoorEnvV3 (v2).
    Phase 2 reward is taken directly from SawyerDoorCloseEnvV3 (v2),
    plus a +10 offset so the combined range is [0, 20].
    The final reward is normalised to [-1, 1] via ``(r - 10) / 10``.

    When Phase 1 completes (door opened), the flag ``self.door_opened``
    is set, the target switches to the close position, and the agent
    receives the full +10 for Phase 1.  The +10 offset in Phase 2
    ensures no reward drop at transition, so the agent is always
    incentivised to progress.
    """

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        height: int = 480,
        width: int = 480,
        reward_function_version: str = "placeholder",
    ) -> None:
        # Hand control bounds (same as individual door envs)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)

        # Door body initialisation bounds
        obj_low = (0.0, 0.85, 0.15)
        obj_high = (0.1, 0.95, 0.15)

        # Task-specific flags
        self.door_opened = False
        self.door_closed = False

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            height=height,
            width=width,
        )

        self.init_config: InitConfigDict = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0.1, 0.95, 0.15]),
            "hand_init_pos": np.array([0, 0.6, 0.2]),
        }

        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self.door_qpos_adr = self.model.joint("doorjoint").qposadr.item()
        self.door_qvel_adr = self.model.joint("doorjoint").dofadr.item()

        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(
            np.array(hand_low), np.array(hand_high), dtype=np.float64
        )

        # Phase targets (set properly in reset_model)
        self._open_target_pos = np.zeros(3)
        self._close_target_pos = np.zeros(3)
        self._target_pos = np.zeros(3)

    # ------------------------------------------------------------------
    # Model / XML
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/compo_dooropen_doorclose.xml")

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[self.door_qpos_adr] = pos
        qvel[self.door_qvel_adr] = 0
        self.set_state(qpos.flatten(), qvel.flatten())

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        """Both phases use the door handle as the tracked object."""
        return self.data.geom("handle").xpos.copy()

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return Rotation.from_matrix(
            self.data.geom("handle").xmat.reshape(3, 3)
        ).as_quat()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()

        # Reset task flags
        self.door_opened = False
        self.door_closed = False

        # Randomise door body position
        self.obj_init_pos = self._get_state_rand_vec()
        self.model.body("door").pos = self.obj_init_pos

        # Door starts closed (joint angle = 0)
        self._set_obj_xyz(np.array(0))

        # Phase 1 target: where the handle should be when door is fully open
        self._open_target_pos = self.obj_init_pos + np.array([-0.3, -0.45, 0.0])
        # Phase 2 target: where the handle should be when door is closed
        self._close_target_pos = self.obj_init_pos + np.array([0.2, -0.2, 0.0])

        # Begin with Phase 1 target
        self._target_pos = self._open_target_pos.copy()
        self.model.site("goal").pos = self._target_pos

        # Max distance for Phase 1 (used in reward shaping)
        self.maxPullDist = np.linalg.norm(
            self.data.geom("handle").xpos[:-1] - self._open_target_pos[:-1]
        )

        self.objHeight = self.data.geom("handle").xpos[2]

        return self._get_obs()

    # ------------------------------------------------------------------
    # Reward helpers (from individual tasks)
    # ------------------------------------------------------------------

    @staticmethod
    def _reward_grab_effort(actions: npt.NDArray[Any]) -> float:
        """Grab effort reward (from SawyerDoorEnvV3)."""
        return float((np.clip(actions[3], -1, 1) + 1.0) / 2.0)

    @staticmethod
    def _reward_pos_open(
        obs: npt.NDArray[Any], theta: float
    ) -> tuple[float, float]:
        """Position-based door-open reward (from SawyerDoorEnvV3)."""
        hand = obs[:3]
        door = obs[4:7] + np.array([-0.05, 0, 0])

        threshold = 0.12
        # 3-D funnel floor centred on the door handle
        radius = np.linalg.norm(hand[:2] - door[:2])
        if radius <= threshold:
            floor = 0.0
        else:
            floor = 0.04 * np.log(radius - threshold) + 0.4

        # Keep the hand above the floor
        above_floor = (
            1.0
            if hand[2] >= floor
            else reward_utils.tolerance(
                floor - hand[2],
                bounds=(0.0, 0.01),
                margin=floor / 2.0,
                sigmoid="long_tail",
            )
        )

        # Move hand between handle and door body
        in_place = reward_utils.tolerance(
            float(np.linalg.norm(hand - door - np.array([0.05, 0.03, -0.01]))),
            bounds=(0, threshold / 2.0),
            margin=0.5,
            sigmoid="long_tail",
        )
        ready_to_open = reward_utils.hamacher_product(above_floor, in_place)

        # Actually open the door
        door_angle = -theta
        a = 0.2  # reward for *trying* to open
        b = 0.8  # reward for fully opening
        opened = a * float(theta < -np.pi / 90.0) + b * reward_utils.tolerance(
            np.pi / 2.0 + np.pi / 6 - door_angle,
            bounds=(0, 0.5),
            margin=np.pi / 3.0,
            sigmoid="long_tail",
        )

        return ready_to_open, opened

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert (
            self._target_pos is not None and self.obj_init_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."

        gripper = obs[:3]
        tcp_opened = obs[3]
        obj_pos = obs[4:7]  # door handle position (both phases)

        # =============================================================
        # PHASE 1 – OPEN THE DOOR
        # =============================================================
        if not self.door_opened:
            theta = float(self.data.joint("doorjoint").qpos.item())

            reward_grab = self._reward_grab_effort(action)
            reward_ready, reward_open = self._reward_pos_open(obs, theta)

            reward = (
                2.0 * reward_utils.hamacher_product(reward_ready, reward_grab)
                + 8.0 * reward_open
            )

            # Check door-open success
            open_success = abs(obs[4] - self._open_target_pos[0]) <= 0.08
            if open_success:
                reward = 10.0
                self.door_opened = True
                # Transition to Phase 2: update target to close position
                self._target_pos = self._close_target_pos.copy()
                self.model.site("goal").pos = self._target_pos

            # Combined task range [0, 20] → normalise to [-1, 1]
            reward = (reward - 10.0) / 10.0

            return (
                reward,
                float(np.linalg.norm(obj_pos - gripper)),
                tcp_opened,
                float(not open_success),  # handle_to_open_target (0 on success)
                reward_ready,
                reward_open,
            )

        # =============================================================
        # PHASE 2 – CLOSE THE DOOR
        # =============================================================
        else:
            _TARGET_RADIUS: float = 0.05
            tcp = self.tcp_center
            target = self._close_target_pos

            obj_to_target = float(np.linalg.norm(obj_pos - target))
            tcp_to_target = float(np.linalg.norm(tcp - target))

            # --- in-place reward (from SawyerDoorCloseEnvV3) ---
            in_place_margin = float(np.linalg.norm(self.obj_init_pos - target))
            in_place = reward_utils.tolerance(
                obj_to_target,
                bounds=(0, _TARGET_RADIUS),
                margin=in_place_margin,
                sigmoid="gaussian",
            )

            # --- hand-in-place reward (from SawyerDoorCloseEnvV3) ---
            hand_margin = float(
                np.linalg.norm(self.hand_init_pos - obj_pos)
            ) + 0.1
            hand_in_place = reward_utils.tolerance(
                tcp_to_target,
                bounds=(0, 0.25 * _TARGET_RADIUS),
                margin=hand_margin,
                sigmoid="gaussian",
            )

            reward = 3.0 * hand_in_place + 6.0 * in_place

            if obj_to_target < _TARGET_RADIUS:
                reward = 10.0
                self.door_closed = True

            # +10 offset for having completed Phase 1
            reward += 10.0

            # Combined task range [0, 20] → normalise to [-1, 1]
            reward = (reward - 10.0) / 10.0

            return (
                reward,
                float(np.linalg.norm(obj_pos - gripper)),
                tcp_opened,
                obj_to_target,
                hand_in_place,
                in_place,
            )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            tcp_to_obj,
            tcp_opened,
            dist_to_target,
            grasp_reward,
            in_place_reward,
        ) = self.compute_reward(action, obs)

        final_success = 1.0 if (self.door_opened and self.door_closed) else 0.0

        info = {
            "success": final_success,
            "door_opened": float(self.door_opened),
            "door_closed": float(self.door_closed),
            "near_object": float(tcp_to_obj <= 0.03),
            "grasp_reward": grasp_reward,
            "in_place_reward": in_place_reward,
            "obj_to_target": dist_to_target,
            "unscaled_reward": reward,
        }

        return reward, info
