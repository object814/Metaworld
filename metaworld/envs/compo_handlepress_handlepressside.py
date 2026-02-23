from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import CompoHandlePressInitConfigDict
from metaworld.utils import reward_utils


class CompoHandlePressHandlePressSideEnv(SawyerXYZEnv):
    """Compositional Handle Press then Handle Press Side Environment.

    Phase 1: Press the normal (front-facing) handle down to its goalPress.
    Phase 2: Press the sideways (rotated 90 deg) handle down to its goalPressSide.

    Both handle assets exist simultaneously in the same scene.
    Reward range: [-1, 1].  Phase 1 maps [0, 10] to [-1, 0],
    Phase 2 maps [10, 20] to [0, 1].
    """

    TARGET_RADIUS: float = 0.02

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
    ) -> None:
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1.0, 0.5)

        # Normal handle box position range
        box_low = (-0.1, 0.8, -0.001)
        box_high = (0.1, 0.9, 0.001)
        # Side handle box position range
        box_side_low = (-0.35, 0.65, -0.001)
        box_side_high = (-0.25, 0.75, 0.001)

        # Task phase flags
        self.normal_pressed = False
        self.side_pressed = False

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            height=height,
            width=width,
        )
        self.reward_function_version = reward_function_version

        self.init_config: CompoHandlePressInitConfigDict = {
            "hand_init_pos": np.array([0, 0.6, 0.2]),
            "box_init_pos": np.array([0, 0.88, 0.0]),
            "box_side_init_pos": np.array([-0.3, 0.7, 0.0]),
        }

        self.hand_init_pos = self.init_config["hand_init_pos"]
        self.box_init_pos = self.init_config["box_init_pos"]
        self.box_side_init_pos = self.init_config["box_side_init_pos"]

        self.goal = np.array([0, 0.88, 0.14])

        self._random_reset_space = Box(
            np.hstack((box_low, box_side_low)),
            np.hstack((box_high, box_side_high)),
            dtype=np.float64,
        )
        self.goal_space = Box(
            np.array(hand_low), np.array(hand_high), dtype=np.float64
        )

        self._target_pos = np.zeros(3)
        self._handle_init_pos = np.zeros(3)
        self._handle_side_init_pos = np.zeros(3)

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/compo_handlepress_handlepressside.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
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

        final_success = 1.0 if (self.normal_pressed and self.side_pressed) else 0.0

        info = {
            "success": final_success,
            "normal_pressed": float(self.normal_pressed),
            "side_pressed": float(self.side_pressed),
            "near_object": float(tcp_to_obj <= 0.05),
            "grasp_success": 1.0,
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
            "unscaled_reward": reward,
        }

        return reward, info

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        return []

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        """Return the active handle position based on current phase."""
        if not self.normal_pressed:
            return self._get_site_pos("handleStart")
        else:
            return self._get_site_pos("handleStartSide")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return np.zeros(4)

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()

        # Reset task flags
        self.normal_pressed = False
        self.side_pressed = False

        # Randomize positions for both boxes
        rand_vec = self._get_state_rand_vec()
        box_pos = rand_vec[:3]
        box_side_pos = rand_vec[3:]

        # Set normal handle box position
        self.model.body("box").pos = box_pos
        self.box_init_pos = box_pos.copy()

        # Set sideways handle box position (euler rotation preserved from XML)
        self.model.body("box_side").pos = box_side_pos
        self.box_side_init_pos = box_side_pos.copy()

        # Reset both handle slide joints to initial position
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        jid_normal = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "handleJoint"
        )
        adr_normal = self.model.jnt_qposadr[jid_normal]
        qpos[adr_normal] = -0.001
        qvel[adr_normal] = 0

        jid_side = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "handleJointSide"
        )
        adr_side = self.model.jnt_qposadr[jid_side]
        qpos[adr_side] = -0.001
        qvel[adr_side] = 0

        self.set_state(qpos, qvel)
        mujoco.mj_forward(self.model, self.data)

        # Phase 1 target: normal handle goalPress
        self._target_pos = self._get_site_pos("goalPress").copy()

        # Store initial handle positions for reward shaping
        self._handle_init_pos = self._get_site_pos("handleStart").copy()
        self._handle_side_init_pos = self._get_site_pos("handleStartSide").copy()

        # Update goal visualization
        self.model.site("goal").pos = self._target_pos

        # Store initial tcp
        self.init_tcp = self.tcp_center

        return self._get_obs()

    def compute_reward(
        self, actions: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."
        del actions

        tcp = self.tcp_center

        # ----------------------------------------------------------
        # PHASE 1 – Press the Normal (front-facing) Handle
        # ----------------------------------------------------------
        if not self.normal_pressed:
            obj = self._get_site_pos("handleStart")
            target = self._target_pos.copy()

            target_to_obj = float(np.linalg.norm(obj[2] - target[2]))
            target_to_obj_init = float(
                np.linalg.norm(self._handle_init_pos[2] - target[2])
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
                np.linalg.norm(self._handle_init_pos - self.init_tcp)
            )
            reach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, handle_radius),
                margin=abs(tcp_to_obj_init - handle_radius),
                sigmoid="long_tail",
            )

            object_grasped = reach
            reward = reward_utils.hamacher_product(reach, in_place)
            reward = 1.0 if target_to_obj <= self.TARGET_RADIUS else reward
            reward *= 10

            # Phase transition check
            if target_to_obj <= self.TARGET_RADIUS:
                self.normal_pressed = True
                # Switch target to side handle
                self._target_pos = self._get_site_pos("goalPressSide").copy()
                self._handle_side_init_pos = self._get_site_pos(
                    "handleStartSide"
                ).copy()
                self.model.site("goal").pos = self._target_pos

            # Normalise [0, 20] → [-1, 1]
            reward = (reward - 10.0) / 10.0
            return (reward, tcp_to_obj, 0.0, target_to_obj, object_grasped, in_place)

        # ----------------------------------------------------------
        # PHASE 2 – Press the Sideways Handle
        # ----------------------------------------------------------
        else:
            obj = self._get_site_pos("handleStartSide")
            target = self._target_pos.copy()

            target_to_obj = float(np.linalg.norm(obj[2] - target[2]))
            target_to_obj_init = float(
                np.linalg.norm(self._handle_side_init_pos[2] - target[2])
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
                np.linalg.norm(self._handle_side_init_pos - self.init_tcp)
            )
            reach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, handle_radius),
                margin=abs(tcp_to_obj_init - handle_radius),
                sigmoid="long_tail",
            )

            object_grasped = reach
            reward = reward_utils.hamacher_product(reach, in_place)
            reward = 1.0 if target_to_obj <= self.TARGET_RADIUS else reward
            reward *= 10

            # Phase 2 offset: +10
            reward += 10.0

            # Completion check
            if target_to_obj <= self.TARGET_RADIUS:
                self.side_pressed = True

            # Normalise [0, 20] → [-1, 1]
            reward = (reward - 10.0) / 10.0
            return (reward, tcp_to_obj, 0.0, target_to_obj, object_grasped, in_place)
