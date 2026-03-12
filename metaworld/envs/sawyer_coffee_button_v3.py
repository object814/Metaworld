from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict


class SawyerCoffeeButtonEnvV3(SawyerXYZEnv):
    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        reward_function_version: str = "v2",
        height: int = 480,
        width: int = 480,
    ) -> None:
        self.max_dist = 0.03
        self._pre_press_offset = np.array([0.0, 0.0, -0.07])
        self._align_success_thresh = 0.05
        self._tcp_to_button_success_thresh = 0.13
        self._button_press_success_thresh = 0.02

        hand_low = (-0.5, 0.4, 0.05)
        hand_high = (0.5, 1.0, 0.5)
        obj_low = (-0.1, 0.8, -0.001)
        obj_high = (0.1, 0.9, +0.001)
        # goal_low[3] would be .1, but objects aren't fully initialized until a
        # few steps after reset(). In that time, it could be .01
        goal_low = obj_low + np.array([-0.001, -0.22 + self.max_dist, 0.299])
        goal_high = obj_high + np.array([+0.001, -0.22 + self.max_dist, 0.301])

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

        self.init_config: InitConfigDict = {
            "obj_init_pos": np.array([0, 0.9, 0.28]),
            "obj_init_angle": 0.3,
            "hand_init_pos": np.array([0.0, 0.4, 0.2]),
        }
        self.goal = np.array([0, 0.78, 0.33])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_coffee.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            near_button,
            button_pressed,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(obj_to_target <= 0.02),
            "near_object": float(tcp_to_obj <= 0.05),
            "grasp_success": float(tcp_open > 0),
            "grasp_reward": near_button,
            "in_place_reward": button_pressed,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
        }

        return reward, info

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `_target_site_config`."
        return [("coffee_goal", self._target_pos)]

    def _get_id_main_object(self):
        return None

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self._get_site_pos("buttonStart")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return np.array([1.0, 0.0, 0.0, 0.0])

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()
        qpos[0:3] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    @staticmethod
    def _progress_fraction(value: float, start: float, complete: float) -> float:
        return float(np.clip((start - value) / max(start - complete, 1e-6), 0.0, 1.0))

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()

        self.obj_init_pos = self._get_state_rand_vec()
        self.model.body("coffee_machine").pos = self.obj_init_pos

        pos_mug = self.obj_init_pos + np.array([0.0, -0.22, 0.0])
        self._set_obj_xyz(pos_mug)

        pos_button = self.obj_init_pos + np.array([0.0, -0.22, 0.3])
        self._target_pos = pos_button + np.array([0.0, self.max_dist, 0.0])

        assert self._target_pos is not None
        button_start = self._get_site_pos("buttonStart")
        self._obj_to_target_init = float(np.abs(button_start[1] - self._target_pos[1]))
        self._align_dist_init = float(
            np.linalg.norm((self.tcp_center - (button_start + self._pre_press_offset))[[0, 2]])
        )
        self._tcp_to_button_init = float(np.linalg.norm(self.tcp_center - button_start))
        self.maxDist = self._obj_to_target_init

        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."
        if self.reward_function_version == "v2":
            del action
            obj = obs[4:7]
            tcp = self.tcp_center

            tcp_to_obj = float(np.linalg.norm(obj - tcp))
            obj_to_target = float(np.abs(self._target_pos[1] - obj[1]))

            pre_press_pos = obj + self._pre_press_offset
            align_dist = float(np.linalg.norm((tcp - pre_press_pos)[[0, 2]]))

            align_progress = self._progress_fraction(
                align_dist,
                self._align_dist_init,
                self._align_success_thresh,
            )
            approach_progress = self._progress_fraction(
                tcp_to_obj,
                self._tcp_to_button_init,
                self._tcp_to_button_success_thresh,
            )
            near_button = 0.5 * (align_progress + approach_progress)
            button_pressed = self._progress_fraction(
                obj_to_target,
                self._obj_to_target_init,
                self._button_press_success_thresh,
            )

            reward = 10.0 * (
                0.4 * align_progress
                + 0.4 * approach_progress
                + 0.2 * button_pressed
            )
            
            # Normalise to [-1, 1]
            reward = (reward - 5.0) / 5.0

            return (
                reward,
                tcp_to_obj,
                obs[3],
                obj_to_target,
                near_button,
                button_pressed,
            )
        else:
            del action

            objPos = obs[4:7]

            leftFinger = self._get_site_pos("leftEndEffector")
            fingerCOM = leftFinger

            pressGoal = self._target_pos[1]

            pressDist = np.abs(objPos[1] - pressGoal)
            reachDist = np.linalg.norm(objPos - fingerCOM)

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            if reachDist < 0.05:
                pressRew = 1000 * (self.maxDist - pressDist) + c1 * (
                    np.exp(-(pressDist**2) / c2) + np.exp(-(pressDist**2) / c3)
                )
            else:
                pressRew = 0

            pressRew = max(pressRew, 0)
            reward = -reachDist + pressRew

            return float(reward), 0.0, 0.0, float(pressDist), 0.0, 0.0
