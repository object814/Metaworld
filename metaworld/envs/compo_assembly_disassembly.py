from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils


class CompoAssemblyDisassemblyEnv(SawyerXYZEnv):
    """Sawyer Compositional Disassemble-then-Assemble Environment.

    Phase 1: Disassemble — the nut (RoundNut) starts on the peg.
             The agent must grasp the nut and lift it off the peg.
    Phase 2: Assemble — after the nut is removed, the agent must
             place the nut back onto the peg.

    Reward design
    -------------
    Each phase produces a reward in [0, 10].
    Phase 1 reward is taken from SawyerNutDisassembleEnvV3 (v2).
    Phase 2 reward is taken from SawyerNutAssemblyEnvV3 (v2),
    plus a +10 offset so the combined range is [0, 20].
    The final reward is normalised to [-1, 1] via ``(r - 10) / 10``.

    When Phase 1 completes (nut lifted above target height), the flag
    ``self.disassembled`` is set, the target switches to the assembly
    position, and the agent receives the full +10 for Phase 1.
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
        obj_low = (0.0, 0.6, 0.025)
        obj_high = (0.1, 0.75, 0.02501)
        # Goal space must cover both disassembly target (Z ~ 0.175)
        # and assembly target (Z ~ 0.025)
        goal_low = (-0.1, 0.6, 0.02)
        goal_high = (0.1, 0.75, 0.20)

        # Task-specific flags
        self.disassembled = False
        self.assembled = False

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
            "obj_init_pos": np.array([0, 0.7, 0.025]),
            "hand_init_pos": np.array((0, 0.4, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0, 0.8, 0.17])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
            dtype=np.float64,
        )
        self.goal_space = Box(
            np.array(goal_low),
            np.array(goal_high),
            dtype=np.float64,
        )

        # Phase targets (set properly in reset_model)
        self._disassemble_target_pos = np.zeros(3)
        self._assemble_target_pos = np.zeros(3)
        self._target_pos = np.zeros(3)

    # ------------------------------------------------------------------
    # Model / XML
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        # Both assembly and disassembly share the same XML scene
        return full_V3_path_for("sawyer_xyz/sawyer_assembly_peg.xml")

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `_target_site_config`."
        return [("pegTop", self._target_pos)]

    def _get_id_main_object(self) -> int:
        return self.model.geom_name2id("WrenchHandle")

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self._get_site_pos("RoundNut-8")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return self.data.body("RoundNut").xquat

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        obs_dict["state_achieved_goal"] = self.get_body_com("RoundNut")
        return obs_dict

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()

        # Reset task flags
        self.disassembled = False
        self.assembled = False

        self.obj_init_pos = np.array(self.init_config["obj_init_pos"])
        self.obj_init_angle = self.init_config["obj_init_angle"]

        # Randomise positions: nut on peg (disassembly start)
        goal_pos = self._get_state_rand_vec()
        while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
            goal_pos = self._get_state_rand_vec()
        self.obj_init_pos = goal_pos[:3]

        # Phase 1 target: above the peg (disassemble — lift nut off)
        self._disassemble_target_pos = self.obj_init_pos + np.array([0, 0, 0.15])

        # Phase 2 target: bottom of the peg (assemble — place nut back on)
        # This is the peg bottom position (same as assembly env uses)
        self._assemble_target_pos = self.obj_init_pos.copy()

        # Start with Phase 1 target
        self._target_pos = self._disassemble_target_pos.copy()

        # Set peg position (nut starts on the peg)
        peg_pos = self.obj_init_pos + np.array([0.0, 0.0, 0.03])
        peg_top_pos = self.obj_init_pos + np.array([0.0, 0.0, 0.08])
        self.model.body("peg").pos = peg_pos
        self.model.site("pegTop").pos = peg_top_pos
        # pegBottom is at -0.05 from peg body center, so it sits at obj_init_pos - 0.02
        self.model.site("pegBottom").pos = self._assemble_target_pos
        mujoco.mj_forward(self.model, self.data)
        self._set_obj_xyz(self.obj_init_pos)

        return self._get_obs()

    # ------------------------------------------------------------------
    # Reward helpers (from individual tasks)
    # ------------------------------------------------------------------

    @staticmethod
    def _reward_quat(obs: npt.NDArray[np.float64]) -> float:
        """Quaternion reward — ideal laid-down wrench has quat [.707, 0, 0, .707]."""
        ideal = np.array([0.707, 0, 0, 0.707])
        error = float(np.linalg.norm(obs[7:11] - ideal))
        return max(1.0 - error / 0.4, 0.0)

    @staticmethod
    def _reward_pos_disassemble(
        wrench_center: npt.NDArray[Any], target_pos: npt.NDArray[Any]
    ) -> float:
        """Position reward for disassembly (from SawyerNutDisassembleEnvV3)."""
        pos_error = target_pos + np.array([0.0, 0.0, 0.1]) - wrench_center

        a = 0.1  # Relative importance of just *trying* to lift the wrench
        b = 0.9  # Relative importance of placing the wrench on the peg
        lifted = wrench_center[2] > 0.02
        in_place = a * float(lifted) + b * reward_utils.tolerance(
            float(np.linalg.norm(pos_error)),
            bounds=(0, 0.02),
            margin=0.2,
            sigmoid="long_tail",
        )

        return in_place

    @staticmethod
    def _reward_pos_assemble(
        wrench_center: npt.NDArray[Any], target_pos: npt.NDArray[Any]
    ) -> tuple[float, bool]:
        """Position reward for assembly (from SawyerNutAssemblyEnvV3)."""
        pos_error = target_pos - wrench_center

        radius = np.linalg.norm(pos_error[:2])

        aligned = radius < 0.02
        hooked = pos_error[2] > 0.0
        success = bool(aligned and hooked)

        threshold = 0.02 if success else 0.01
        target_height = 0.0
        if radius > threshold:
            target_height = 0.02 * np.log(radius - threshold) + 0.2

        pos_error[2] = target_height - wrench_center[2]

        scale = np.array([1.0, 1.0, 3.0])
        a = 0.1
        b = 0.9
        lifted = wrench_center[2] > 0.02 or radius < threshold
        in_place = a * float(lifted) + b * reward_utils.tolerance(
            float(np.linalg.norm(pos_error * scale)),
            bounds=(0, 0.02),
            margin=0.4,
            sigmoid="long_tail",
        )

        return in_place, success

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_reward(
        self, actions: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."

        hand = obs[:3]
        wrench = obs[4:7]
        wrench_center = self._get_site_pos("RoundNut")

        # Shared: gripper caging reward (wrench handle threshold)
        wrench_threshed = wrench.copy()
        threshold = CompoAssemblyDisassemblyEnv.WRENCH_HANDLE_LENGTH / 2.0
        if abs(wrench[0] - hand[0]) < threshold:
            wrench_threshed[0] = hand[0]

        reward_quat = CompoAssemblyDisassemblyEnv._reward_quat(obs)
        reward_grab = self._gripper_caging_reward(
            actions,
            wrench_threshed,
            object_reach_radius=0.01,
            obj_radius=0.015,
            pad_success_thresh=0.02,
            xz_thresh=0.01,
            high_density=True,
        )

        # =============================================================
        # PHASE 1 — DISASSEMBLE (lift nut off peg)
        # =============================================================
        if not self.disassembled:
            reward_in_place = CompoAssemblyDisassemblyEnv._reward_pos_disassemble(
                wrench_center, self._disassemble_target_pos
            )

            reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat

            # Check disassemble success: nut Z > target Z
            disassemble_success = obs[6] > self._disassemble_target_pos[2]
            if disassemble_success:
                reward = 10.0
                self.disassembled = True
                # Transition to Phase 2: update target to assembly position
                self._target_pos = self._assemble_target_pos.copy()

            # Combined task range [0, 20] → normalise to [-1, 1]
            reward = (reward - 10.0) / 10.0

            return (
                reward,
                reward_grab,
                reward_quat,
                reward_in_place,
                float(disassemble_success),
                0.0,  # assemble_success placeholder
            )

        # =============================================================
        # PHASE 2 — ASSEMBLE (place nut onto peg)
        # =============================================================
        else:
            reward_in_place, assemble_success = (
                CompoAssemblyDisassemblyEnv._reward_pos_assemble(
                    wrench_center, self._assemble_target_pos
                )
            )

            reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat

            if assemble_success:
                reward = 10.0
                self.assembled = True

            # +10 offset for having completed Phase 1
            reward += 10.0

            # Combined task range [0, 20] → normalise to [-1, 1]
            reward = (reward - 10.0) / 10.0

            return (
                reward,
                reward_grab,
                reward_quat,
                reward_in_place,
                1.0,  # disassemble already done
                float(assemble_success),
            )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            reward_grab,
            reward_quat,
            reward_in_place,
            disassemble_success,
            assemble_success,
        ) = self.compute_reward(action, obs)

        final_success = 1.0 if (self.disassembled and self.assembled) else 0.0

        info = {
            "success": final_success,
            "disassembled": float(self.disassembled),
            "assembled": float(self.assembled),
            "grasp_success": reward_grab >= 0.5,
            "grasp_reward": reward_grab,
            "in_place_reward": reward_in_place,
            "obj_to_target": 0,
            "unscaled_reward": reward,
        }

        return reward, info
