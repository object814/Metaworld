from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import CompoPickPlaceDrawerInitConfigDict
from metaworld.utils import reward_utils


class CompoPickPlaceDrawerOpenEnv(SawyerXYZEnv):
    """
    Sawyer Pick and Place + Drawer Open Environment.
    Combination of SawyerPickPlaceEnvV3 and SawyerDrawerOpenEnv.

    Drawer open -> Pick and place into the drawer.
    """
    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v3",
        height: int = 480,
        width: int = 480,
    ) -> None:
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        # Drawer base randomisation
        drawer_low = (-0.1, 0.9, 0.0)
        drawer_high = (0.1, 0.9, 0.0)
        # Object radomisation
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)

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

        self.init_config: CompoPickPlaceDrawerInitConfigDict = {
            "drawer_init_angle": 0.3,
            "drawer_init_pos": np.array([0.0, 0.9, 0.0], dtype=np.float32),
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0, 0.6, 0.02], dtype=np.float32),
            "hand_init_pos": np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.drawer_init_pos = self.init_config["drawer_init_pos"].copy()
        self.drawer_init_angle = self.init_config["drawer_init_angle"]
        self.obj_init_pos = self.init_config["obj_init_pos"].copy()
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"].copy()

        # Random reset
        self._random_reset_space = Box(
            low=np.hstack((drawer_low, obj_low)),
            high=np.hstack((drawer_high, obj_high)),
            dtype=np.float64,
        )

        # Drawer open parameters
        self.maxDist = 0.2
        self.target_reward = 1000 * self.maxDist + 1000 * 2

        # Task transition indicator
        self.drawer_opened = False
        self.obj_placed = False

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/compo_pick_place_drawer_open.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            gripper_error,
            gripped,
            handle_error,
            caging_reward,
            opening_reward,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(handle_error <= 0.03),
            "near_object": float(gripper_error <= 0.03),
            "grasp_success": float(gripped > 0),
            "grasp_reward": caging_reward,
            "in_place_reward": opening_reward,
            "obj_to_target": handle_error,
            "unscaled_reward": reward,
        }

        return reward, info

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        # Compute nightstand position
        self.obj_init_pos = self._get_state_rand_vec()
        # Set mujoco body to computed position
        self.model.body("drawer").pos = self.obj_init_pos

        # Set _target_pos to current drawer position (closed) minus an offset
        self._target_pos = self.obj_init_pos + np.array(
            [0.0, -0.16 - self.maxDist, 0.09]
        )
        self.model.site("goal").pos = self._target_pos
        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."
        if self.reward_function_version == "v3":
            gripper = obs[:3]
            handle = obs[4:7]

            handle_error = float(np.linalg.norm(handle - self._target_pos))

            reward_for_opening = reward_utils.tolerance(
                handle_error, bounds=(0, 0.02), margin=self.maxDist, sigmoid="long_tail"
            )

            handle_pos_init = self._target_pos + np.array([0.0, self.maxDist, 0.0])
            # Emphasize XY error so that gripper is able to drop down and cage
            # handle without running into it. By doing this, we are assuming
            # that the reward in the Z direction is small enough that the agent
            # will be willing to explore raising a finger above the handle, hook it,
            # and drop back down to re-gain Z reward
            scale = np.array([3.0, 3.0, 1.0])
            gripper_error = (handle - gripper) * scale
            gripper_error_init = (handle_pos_init - self.init_tcp) * scale

            reward_for_caging = reward_utils.tolerance(
                float(np.linalg.norm(gripper_error)),
                bounds=(0, 0.01),
                margin=np.linalg.norm(gripper_error_init),
                sigmoid="long_tail",
            )

            reward = reward_for_caging + reward_for_opening
            reward *= 5.0

            return (
                reward,
                float(np.linalg.norm(handle - gripper)),
                obs[3],
                handle_error,
                reward_for_caging,
                reward_for_opening,
            )
        else:
            del action

            objPos = obs[4:7]
            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2
            pullGoal = self._target_pos
            pullDist = np.abs(objPos[1] - pullGoal[1])
            reachDist = np.linalg.norm(objPos - fingerCOM)
            reachRew = -reachDist

            self.reachCompleted = reachDist < 0.05

            c1 = 1000
            c2 = 0.01
            c3 = 0.001

            if self.reachCompleted:
                pullRew = 1000 * (self.maxDist - pullDist) + c1 * (
                    np.exp(-(pullDist**2) / c2) + np.exp(-(pullDist**2) / c3)
                )
                pullRew = max(pullRew, 0)
            else:
                pullRew = 0

            reward = reachRew + pullRew

            return reward, 0.0, 0.0, pullDist, 0.0, 0.0
