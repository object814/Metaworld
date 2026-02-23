from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict, ObservationDict
from metaworld.utils import reward_utils
from metaworld.utils.reward_utils import tolerance


class CompoAssemblyDisassemblyEnv(SawyerXYZEnv):
    """Sawyer Compositional Assembly then Disassembly Environment.

    Phase 1 - Assembly: Pick up the wrench from the table and place it
              onto the peg (ring around the peg).
    Phase 2 - Disassembly: Pick the wrench back off the peg and lift it
              above a target height.

    Reward design
    -------------
    Each phase produces a reward in [0, 10].
    Phase 1 uses the SawyerNutAssemblyEnvV3 (v2) reward:
        (2*grab + 6*in_place) * quat,  overridden to 10 on success.
    Phase 2 uses the SawyerNutDisassembleEnvV3 (v2) reward:
        (2*grab + 6*in_place) * quat,  overridden to 10 on success,
        plus a +10 offset so the combined range is [0, 20].
    Final reward normalised to [-1, 1] via ``(r - 10) / 10``.
    """

    WRENCH_HANDLE_LENGTH: float = 0.02

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
        obj_low = (0.0, 0.6, 0.02)
        obj_high = (0.0, 0.6, 0.02)
        goal_low = (-0.1, 0.75, 0.1)
        goal_high = (0.1, 0.85, 0.1)

        # Task-specific flags
        self.assembly_completed = False
        self.disassembly_completed = False

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
            "obj_init_pos": np.array([0, 0.6, 0.02], dtype=np.float32),
            "hand_init_pos": np.array((0, 0.6, 0.2), dtype=np.float32),
        }

        self.goal = np.array([0.1, 0.8, 0.1], dtype=np.float32)
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
            dtype=np.float64,
        )
        self.goal_space = Box(
            np.array(goal_low), np.array(goal_high), dtype=np.float64
        )

        # Phase targets (set in reset_model)
        self._assembly_target = np.zeros(3)
        self._disassembly_target = np.zeros(3)
        self._target_pos = np.zeros(3)

    # ------------------------------------------------------------------
    # Model / XML
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/compo_assembly_disassembly.xml")

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _get_id_main_object(self) -> int:
        return self.model.geom_name2id("WrenchHandle")

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.data.site("RoundNut-8").xpos

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("RoundNut").xquat

    def _get_obs_dict(self) -> ObservationDict:
        obs_dict = super()._get_obs_dict()
        obs_dict["state_achieved_goal"] = self.get_body_com("RoundNut")
        return obs_dict

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        assert isinstance(
            self._target_pos, np.ndarray
        ), "`reset_model()` must be called before `_target_site_config`."
        return [("pegTop", self._target_pos)]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()

        # Reset task flags
        self.assembly_completed = False
        self.disassembly_completed = False

        # Randomize wrench and peg positions
        goal_pos = self._get_state_rand_vec()
        while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
            goal_pos = self._get_state_rand_vec()
        self.obj_init_pos = goal_pos[:3]

        # Peg top position (physical peg placement stays the same)
        peg_top_pos = goal_pos[-3:]
        peg_pos = peg_top_pos - np.array([0.0, 0.0, 0.05])
        self.model.body("peg").pos = peg_pos
        self.model.site("pegTop").pos = peg_top_pos

        # Phase 1 target: ring must slide all the way down to the table
        self._assembly_target = np.array(
            [peg_top_pos[0], peg_top_pos[1], 0.02]
        )

        # Phase 2 target: lift wrench well above the peg
        self._disassembly_target = peg_top_pos + np.array(
            [0.0, 0.0, 0.15]
        )

        # Start with Phase 1 target
        self._target_pos = self._assembly_target.copy()
        self.model.site("goal").pos = self._target_pos

        # Place wrench on ground
        self._set_obj_xyz(self.obj_init_pos)

        return self._get_obs()

    # ------------------------------------------------------------------
    # Reward helpers (from individual tasks)
    # ------------------------------------------------------------------

    @staticmethod
    def _reward_quat(obs: npt.NDArray[np.float64]) -> float:
        """Orientation reward — ideal laid-down wrench has quat [.707, 0, 0, .707]."""
        ideal = np.array([0.707, 0, 0, 0.707])
        error = float(np.linalg.norm(obs[7:11] - ideal))
        return max(1.0 - error / 0.4, 0.0)

    @staticmethod
    def _reward_pos_assembly(
        wrench_center: npt.NDArray[Any], target_pos: npt.NDArray[Any]
    ) -> tuple[float, bool]:
        """Assembly position reward.

        Success requires the ring to be XY-aligned with the peg *and*
        lowered to the table (target z = 0.02).  The 3-D funnel reward
        still shapes behaviour: fly high when far from the peg, descend
        when XY-aligned.
        """
        pos_error = target_pos - wrench_center

        radius = np.linalg.norm(pos_error[:2])

        aligned = radius < 0.02
        at_target_height = abs(pos_error[2]) < 0.03
        success = bool(aligned and at_target_height)

        threshold = 0.02 if success else 0.01
        target_height = 0.0
        if radius > threshold:
            target_height = 0.02 * np.log(radius - threshold) + 0.2

        pos_error[2] = target_height - wrench_center[2]

        scale = np.array([1.0, 1.0, 3.0])
        a = 0.1  # Relative importance of just *trying* to lift the wrench
        b = 0.9  # Relative importance of placing the wrench on the peg
        lifted = wrench_center[2] > 0.02 or radius < threshold
        in_place = a * float(lifted) + b * tolerance(
            float(np.linalg.norm(pos_error * scale)),
            bounds=(0, 0.02),
            margin=0.4,
            sigmoid="long_tail",
        )

        return in_place, success

    @staticmethod
    def _reward_pos_disassembly(
        wrench_center: npt.NDArray[Any], target_pos: npt.NDArray[Any]
    ) -> float:
        """Disassembly position reward (from SawyerNutDisassembleEnvV3)."""
        pos_error = target_pos + np.array([0.0, 0.0, 0.1]) - wrench_center

        a = 0.1  # Relative importance of just *trying* to lift the wrench
        b = 0.9  # Relative importance of lifting high enough
        lifted = wrench_center[2] > 0.02
        in_place = a * float(lifted) + b * reward_utils.tolerance(
            float(np.linalg.norm(pos_error)),
            bounds=(0, 0.02),
            margin=0.2,
            sigmoid="long_tail",
        )

        return in_place

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert (
            self._target_pos is not None and self.obj_init_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."

        hand = obs[:3]
        tcp_opened = obs[3]
        wrench = obs[4:7]
        wrench_center = self._get_site_pos("RoundNut")

        # Threshold wrench X for caging reward (both phases use this)
        wrench_threshed = wrench.copy()
        threshold = self.WRENCH_HANDLE_LENGTH / 2.0
        if abs(wrench[0] - hand[0]) < threshold:
            wrench_threshed[0] = hand[0]

        # =============================================================
        # PHASE 1 – ASSEMBLY
        # =============================================================
        if not self.assembly_completed:
            reward_quat = self._reward_quat(obs)
            reward_grab = self._gripper_caging_reward(
                action,
                wrench_threshed,
                object_reach_radius=0.01,
                obj_radius=0.015,
                pad_success_thresh=0.02,
                xz_thresh=0.01,
                medium_density=True,
            )
            reward_in_place, success = self._reward_pos_assembly(
                wrench_center, self._assembly_target
            )

            reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat

            if success:
                reward = 10.0
                self.assembly_completed = True
                # Transition to Phase 2
                self._target_pos = self._disassembly_target.copy()
                self.model.site("goal").pos = self._target_pos

            # Combined task range [0, 20] → normalise to [-1, 1]
            reward = (reward - 10.0) / 10.0

            return (
                reward,
                float(np.linalg.norm(wrench - hand)),
                tcp_opened,
                float(not success),
                reward_grab,
                reward_in_place,
            )

        # =============================================================
        # PHASE 2 – DISASSEMBLY
        # =============================================================
        else:
            reward_quat = self._reward_quat(obs)
            reward_grab = self._gripper_caging_reward(
                action,
                wrench_threshed,
                object_reach_radius=0.01,
                obj_radius=0.015,
                pad_success_thresh=0.02,
                xz_thresh=0.01,
                high_density=True,
            )
            reward_in_place = self._reward_pos_disassembly(
                wrench_center, self._disassembly_target
            )

            reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat

            # Success: wrench lifted above target height
            success = bool(obs[6] > self._disassembly_target[2])
            if success:
                reward = 10.0
                self.disassembly_completed = True

            # +10 offset for having completed Phase 1
            reward += 10.0

            # Combined task range [0, 20] → normalise to [-1, 1]
            reward = (reward - 10.0) / 10.0

            return (
                reward,
                float(np.linalg.norm(wrench - hand)),
                tcp_opened,
                float(not success),
                reward_grab,
                reward_in_place,
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

        final_success = (
            1.0
            if (self.assembly_completed and self.disassembly_completed)
            else 0.0
        )

        info = {
            "success": final_success,
            "assembly_completed": float(self.assembly_completed),
            "disassembly_completed": float(self.disassembly_completed),
            "near_object": float(tcp_to_obj <= 0.03),
            "grasp_reward": grasp_reward,
            "in_place_reward": in_place_reward,
            "obj_to_target": dist_to_target,
            "unscaled_reward": reward,
        }

        return reward, info
