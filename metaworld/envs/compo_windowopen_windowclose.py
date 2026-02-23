from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils


class CompoWindowOpenWindowCloseEnv(SawyerXYZEnv):
    """Sawyer Compositional Window-Open then Window-Close Environment.

    Phase 1: Slide the window open (push the handle from left → right).
    Phase 2: Slide the window closed (push the handle from right → left).

    Reward design
    -------------
    Each phase produces a reward in [0, 10].
    Phase 1 reward is taken from SawyerWindowOpenEnvV3 (v2):
        10 * hamacher(reach, in_place)  with long_tail reach sigmoid.
    Phase 2 reward is taken from SawyerWindowCloseEnvV3 (v2):
        10 * hamacher(reach, in_place)  with gaussian reach sigmoid,
        plus a +10 offset so the combined range is [0, 20].
    The final reward is normalised to [-1, 1] via ``(r - 10) / 10``.
    """

    TARGET_RADIUS: float = 0.05

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        height: int = 480,
        width: int = 480,
        reward_function_version: str = "placeholder",
    ) -> None:
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.7, 0.16)
        obj_high = (0.1, 0.9, 0.16)

        # Task-specific flags
        self.window_opened = False
        self.window_closed = False

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
            "obj_init_pos": np.array([-0.1, 0.785, 0.16], dtype=np.float32),
            "hand_init_pos": np.array([0, 0.4, 0.2], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(
            np.array(goal_low), np.array(goal_high), dtype=np.float64
        )

        self.maxPullDist = 0.2
        self._target_pos = np.zeros(3)

        # Phase targets (set properly in reset_model)
        self._open_target_pos = np.zeros(3)
        self._close_target_pos = np.zeros(3)

    # ------------------------------------------------------------------
    # Model / XML
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/compo_windowopen_windowclose.xml")

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        """Phase 1 tracks handleOpenStart, Phase 2 tracks handleCloseStart."""
        if not self.window_opened:
            return self._get_site_pos("handleOpenStart")
        else:
            return self._get_site_pos("handleCloseStart")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return np.zeros(4)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_hand(self, steps: int = 50) -> None:
        super()._reset_hand(steps=steps)
        self.init_tcp = self.tcp_center

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        # Reset task flags
        self.window_opened = False
        self.window_closed = False

        # Randomise window body position
        self.obj_init_pos = self._get_state_rand_vec()
        self.model.body("window").pos = self.obj_init_pos

        # Window starts closed (slide joint = 0)
        self.data.joint("window_slide").qpos = 0.0

        # Record initial handle position for reward shaping
        self.window_handle_pos_init = self._get_site_pos("handleOpenStart")

        # Phase 1 target: open position (+0.2 in X)
        self._open_target_pos = self.obj_init_pos + np.array([0.2, 0.0, 0.0])
        # Phase 2 target: closed position (back to obj_init_pos)
        self._close_target_pos = self.obj_init_pos.copy()

        # Start with Phase 1 target
        self._target_pos = self._open_target_pos.copy()
        self.model.site("goal").pos = self._target_pos

        return self._get_obs()

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

        # =============================================================
        # PHASE 1 – OPEN THE WINDOW
        # =============================================================
        if not self.window_opened:
            obj = self._get_site_pos("handleOpenStart")
            tcp = self.tcp_center
            target = self._open_target_pos.copy()

            # Distance along X from handle to open target
            target_to_obj = float(np.linalg.norm(obj[0] - target[0]))
            target_to_obj_init = float(
                np.linalg.norm(self.obj_init_pos[0] - target[0])
            )

            in_place = reward_utils.tolerance(
                target_to_obj,
                bounds=(0, self.TARGET_RADIUS),
                margin=abs(target_to_obj_init - self.TARGET_RADIUS),
                sigmoid="long_tail",
            )

            handle_radius = 0.02
            tcp_to_obj = float(np.linalg.norm(obj - tcp))
            tcp_to_obj_init = float(
                np.linalg.norm(self.window_handle_pos_init - self.init_tcp)
            )
            reach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, handle_radius),
                margin=abs(tcp_to_obj_init - handle_radius),
                sigmoid="long_tail",
            )

            object_grasped = reach
            reward = 10.0 * reward_utils.hamacher_product(reach, in_place)

            # Check window-open success
            if target_to_obj <= self.TARGET_RADIUS:
                reward = 10.0
                self.window_opened = True
                # Transition to Phase 2: update target
                self._target_pos = self._close_target_pos.copy()
                self.model.site("goal").pos = self._target_pos
                # Record handle position at start of Phase 2 for close reward
                self.window_handle_pos_init_close = self._get_site_pos(
                    "handleCloseStart"
                ) + np.array([0.0, 0.0, 0.0])

            # Combined task range [0, 20] → normalise to [-1, 1]
            reward = (reward - 10.0) / 10.0

            return (
                reward,
                tcp_to_obj,
                tcp_opened,
                target_to_obj,
                object_grasped,
                in_place,
            )

        # =============================================================
        # PHASE 2 – CLOSE THE WINDOW
        # =============================================================
        else:
            obj = self._get_site_pos("handleCloseStart")
            tcp = self.tcp_center
            target = self._close_target_pos.copy()

            # Distance along X from handle to close target
            target_to_obj = float(np.linalg.norm(obj[0] - target[0]))

            # Reference: handle position when Phase 2 started (window fully open)
            if hasattr(self, "window_handle_pos_init_close"):
                handle_init = self.window_handle_pos_init_close
            else:
                # Fallback: approximate open-position offset
                handle_init = self._open_target_pos.copy()

            target_to_obj_init = float(
                np.linalg.norm(handle_init[0] - target[0])
            )

            in_place = reward_utils.tolerance(
                target_to_obj,
                bounds=(0, self.TARGET_RADIUS),
                margin=abs(target_to_obj_init - self.TARGET_RADIUS),
                sigmoid="long_tail",
            )

            handle_radius = 0.02
            tcp_to_obj = float(np.linalg.norm(obj - tcp))
            tcp_to_obj_init = float(
                np.linalg.norm(handle_init - self.init_tcp)
            )
            reach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, handle_radius),
                margin=abs(tcp_to_obj_init - handle_radius),
                sigmoid="gaussian",
            )

            object_grasped = reach
            reward = 10.0 * reward_utils.hamacher_product(reach, in_place)

            if target_to_obj <= self.TARGET_RADIUS:
                reward = 10.0
                self.window_closed = True

            # +10 offset for having completed Phase 1
            reward += 10.0

            # Combined task range [0, 20] → normalise to [-1, 1]
            reward = (reward - 10.0) / 10.0

            return (
                reward,
                tcp_to_obj,
                tcp_opened,
                target_to_obj,
                object_grasped,
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
            target_to_obj,
            object_grasped,
            in_place,
        ) = self.compute_reward(action, obs)

        final_success = (
            1.0 if (self.window_opened and self.window_closed) else 0.0
        )

        info = {
            "success": final_success,
            "window_opened": float(self.window_opened),
            "window_closed": float(self.window_closed),
            "near_object": float(tcp_to_obj <= 0.05),
            "grasp_success": 1.0,
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
            "unscaled_reward": reward,
        }

        return reward, info
