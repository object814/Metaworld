from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import CompoCoffeePushButtonPullInitConfigDict
from metaworld.utils import reward_utils


class CompoCoffeePushButtonPullEnv(SawyerXYZEnv):
    """Compositional Coffee Task: Push Mug -> Press Button -> Pull Mug.

    Phase 1:   Push the mug to the coffee machine (under the spout).
    Phase 1.5: Retreat arm to rest position.
    Phase 2:   Press the coffee machine button.
    Phase 2.5: Retreat arm to rest position.
    Phase 3:   Pull the mug away from the machine.

    Reward bands (smooth, no discontinuities at boundaries):
        Push:           [0,   2.5]
        Push retreat:   [2.5, 3.0]   (linear in distance to rest pose)
        Button:         [3.0, 5.5]
        Button retreat: [5.5, 6.0]   (linear in distance to rest pose)
        Pull:           [6.0, 10.0]
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
        # --- Push thresholds (from coffee-push) ---
        self._push_contact_offset = np.array([0.01, 0.0, 0.05])
        self._push_contact_success_thresh = 0.04
        self._push_success_thresh = 0.07

        # --- Button thresholds (from coffee-button) ---
        self.max_button_dist = 0.03
        self._pre_press_offset = np.array([0.0, 0.0, -0.07])
        self._align_success_thresh = 0.05
        self._tcp_to_button_success_thresh = 0.13
        self._button_press_success_thresh = 0.02

        # --- Pull thresholds (from coffee-pull) ---
        self._pull_grasp_offset = np.array([-0.005, 0.0, 0.05])
        self._pull_grasp_success_thresh = 0.04
        self._pull_success_thresh = 0.07

        # --- Retreat threshold ---
        self._retreat_complete_thresh = 0.08

        # Phase flags
        self.push_completed = False
        self.push_retreated = False
        self.button_completed = False
        self.button_retreated = False
        self.pull_completed = False

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1.0, 0.5)

        # Mug initial position range (in front of machine)
        mug_init_low = (-0.1, 0.50, -0.001)
        mug_init_high = (0.1, 0.60, +0.001)
        # Pull goal range (away from machine, toward robot)
        pull_goal_low = (-0.15, 0.40, -0.001)
        pull_goal_high = (0.15, 0.55, +0.001)

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            height=height,
            width=width,
        )

        # Fixed machine position
        self._machine_pos = np.array([0.0, 0.9, 0.0])

        self.init_config: CompoCoffeePushButtonPullInitConfigDict = {
            "hand_init_pos": np.array([0.0, 0.4, 0.2]),
            "mug_init_pos": np.array([0.0, 0.55, 0.0]),
            "machine_pos": self._machine_pos.copy(),
        }

        self.hand_init_pos = self.init_config["hand_init_pos"]
        self.obj_init_pos = self.init_config["mug_init_pos"]

        # Rest position for retreat phases (= hand init / starting pose)
        self._rest_pos = self.hand_init_pos.copy()

        self._random_reset_space = Box(
            np.hstack((mug_init_low, pull_goal_low)),
            np.hstack((mug_init_high, pull_goal_high)),
            dtype=np.float64,
        )
        self.goal_space = Box(
            np.array(hand_low), np.array(hand_high), dtype=np.float64
        )

        self._target_pos = np.zeros(3)

        # Lazy-init flags for per-phase starting distances
        self._push_retreat_init_computed = False
        self._button_init_computed = False
        self._button_retreat_init_computed = False
        self._pull_init_computed = False

        # Stored targets for each phase
        self._push_target = np.zeros(3)
        self._button_target = np.zeros(3)
        self._pull_target = np.zeros(3)

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/sawyer_coffee.xml")

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `_target_site_config`."
        return [("mug_goal", self._target_pos)]

    def _get_id_main_object(self):
        # Mug contact matters only during push and pull
        if not self.push_completed:
            return self.data.geom("mug").id
        elif not self.button_completed:
            return None  # push-retreat + button phases
        elif not self.button_retreated:
            return None  # button-retreat phase
        else:
            return self.data.geom("mug").id  # pull phase

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        """Return the currently-relevant object position.

        Push:           mug
        Push retreat:   button  (preparing for button phase)
        Button:         button
        Button retreat: mug     (preparing for pull phase)
        Pull:           mug
        """
        if not self.push_completed:
            return self.get_body_com("obj")
        elif not self.push_retreated:
            return self._get_site_pos("buttonStart")
        elif not self.button_completed:
            return self._get_site_pos("buttonStart")
        else:
            return self.get_body_com("obj")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        if not self.push_completed:
            geom_xmat = self.data.geom("mug").xmat.reshape(3, 3)
            return Rotation.from_matrix(geom_xmat).as_quat()
        elif not self.push_retreated:
            return np.array([1.0, 0.0, 0.0, 0.0])
        elif not self.button_completed:
            return np.array([1.0, 0.0, 0.0, 0.0])
        else:
            geom_xmat = self.data.geom("mug").xmat.reshape(3, 3)
            return Rotation.from_matrix(geom_xmat).as_quat()

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

        # Reset all phase flags
        self.push_completed = False
        self.push_retreated = False
        self.button_completed = False
        self.button_retreated = False
        self.pull_completed = False
        self._push_retreat_init_computed = False
        self._button_init_computed = False
        self._button_retreat_init_computed = False
        self._pull_init_computed = False

        # Randomize positions
        rand_vec = self._get_state_rand_vec()
        pos_mug_init = rand_vec[:3]
        pos_pull_goal = rand_vec[3:]

        # Ensure mug init and pull goal are separated
        while np.linalg.norm(pos_mug_init[:2] - pos_pull_goal[:2]) < 0.1:
            rand_vec = self._get_state_rand_vec()
            pos_mug_init = rand_vec[:3]
            pos_pull_goal = rand_vec[3:]

        # Set mug position
        self._set_obj_xyz(pos_mug_init)
        self.obj_init_pos = pos_mug_init

        # Set machine position
        self.model.body("coffee_machine").pos = self._machine_pos

        # Reset button joint to 0
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "goal_slidey"
        )
        qpos_adr = self.model.jnt_qposadr[joint_id]
        self.data.qpos[qpos_adr] = 0.0

        # Forward to update site positions
        mujoco.mj_forward(self.model, self.data)

        # Compute phase targets
        self._push_target = self._machine_pos + np.array([0.0, -0.22, 0.0])
        button_start = self._get_site_pos("buttonStart")
        self._button_target = button_start + np.array(
            [0.0, self.max_button_dist, 0.0]
        )
        self._pull_target = pos_pull_goal

        # Set initial target to push target
        self._target_pos = self._push_target.copy()

        # Compute push init distances
        self._push_contact_dist_init = float(
            np.linalg.norm(
                self.tcp_center - (pos_mug_init + self._push_contact_offset)
            )
        )
        self._push_obj_to_target_init = float(
            np.linalg.norm(pos_mug_init - self._push_target)
        )

        return self._get_obs()

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert self._target_pos is not None

        gripper = obs[:3]
        tcp_opened = obs[3]
        obj_pos = obs[4:7]
        tcp = self.tcp_center

        # Helper: mug-drift penalty (mug should stay at push target)
        def _mug_drift_penalty(scale: float = 0.25) -> float:
            mug_pos = self.get_body_com("obj")
            drift = float(np.linalg.norm(mug_pos[:2] - self._push_target[:2]))
            return (
                reward_utils.tolerance(
                    drift, bounds=(0, 0.02), margin=0.15, sigmoid="long_tail"
                )
                - 1.0
            ) * scale  # in [-scale, 0]

        # =============================================================
        # PHASE 1: PUSH MUG  —  reward in [0, 2.5]
        # =============================================================
        if not self.push_completed:
            target = self._push_target.copy()
            contact_target = obj_pos + self._push_contact_offset
            tcp_to_obj = float(np.linalg.norm(obj_pos - tcp))
            contact_dist = float(np.linalg.norm(tcp - contact_target))
            obj_to_target = float(np.linalg.norm(obj_pos - target))

            contact_reward = self._progress_fraction(
                contact_dist,
                self._push_contact_dist_init,
                self._push_contact_success_thresh,
            )
            in_place = self._progress_fraction(
                obj_to_target,
                self._push_obj_to_target_init,
                self._push_success_thresh,
            )

            raw = 10.0 * (0.5 * contact_reward + 0.5 * in_place)
            if obj_to_target <= self._push_success_thresh:
                in_place = 1.0
                raw = 10.0
                self.push_completed = True
                self._target_pos = self._button_target.copy()

            reward = float(np.clip(raw, 0.0, 10.0)) * 0.25  # [0, 2.5]


            # Normalise the overall reward to [-1, 1]
            reward = (reward - 5.0) / 5.0

            return (reward, tcp_to_obj, tcp_opened, obj_to_target, contact_reward, in_place)

        # =============================================================
        # PHASE 1.5: PUSH RETREAT  —  reward in [2.5, 3.0]
        #   Linear in distance to rest pose.  Mug-drift penalty applied.
        # =============================================================
        elif not self.push_retreated:
            if not self._push_retreat_init_computed:
                self._push_retreat_dist_init = max(
                    float(np.linalg.norm(tcp - self._rest_pos)), 0.01
                )
                self._push_retreat_init_computed = True

            dist_to_rest = float(np.linalg.norm(tcp - self._rest_pos))
            retreat_frac = float(
                np.clip(1.0 - dist_to_rest / self._push_retreat_dist_init, 0.0, 1.0)
            )

            mug_pen = _mug_drift_penalty(0.25)  # [-0.25, 0]
            reward = 2.5 + retreat_frac * 0.5 + mug_pen  # ~[2.25, 3.0]

            if dist_to_rest < self._retreat_complete_thresh:
                self.push_retreated = True

            # Normalise the overall reward to [-1, 1]
            reward = (reward - 5.0) / 5.0

            return (reward, dist_to_rest, tcp_opened, dist_to_rest, retreat_frac, 0.0)

        # =============================================================
        # PHASE 2: PRESS BUTTON  —  reward in [3.0, 5.5]
        #   Mug-drift penalty keeps mug in place for later pull.
        # =============================================================
        elif not self.button_completed:
            if not self._button_init_computed:
                button_start = self._get_site_pos("buttonStart")
                self._button_obj_to_target_init = float(
                    np.abs(button_start[1] - self._button_target[1])
                )
                self._button_align_dist_init = float(
                    np.linalg.norm(
                        (tcp - (button_start + self._pre_press_offset))[[0, 2]]
                    )
                )
                self._button_tcp_to_button_init = float(
                    np.linalg.norm(tcp - button_start)
                )
                self._button_init_computed = True

            tcp_to_obj = float(np.linalg.norm(obj_pos - tcp))
            obj_to_target = float(np.abs(self._button_target[1] - obj_pos[1]))

            pre_press_pos = obj_pos + self._pre_press_offset
            align_dist = float(np.linalg.norm((tcp - pre_press_pos)[[0, 2]]))

            align_progress = self._progress_fraction(
                align_dist, self._button_align_dist_init, self._align_success_thresh
            )
            approach_progress = self._progress_fraction(
                tcp_to_obj, self._button_tcp_to_button_init, self._tcp_to_button_success_thresh
            )
            button_pressed = self._progress_fraction(
                obj_to_target, self._button_obj_to_target_init, self._button_press_success_thresh
            )

            raw_button = 10.0 * (
                0.4 * align_progress + 0.4 * approach_progress + 0.2 * button_pressed
            )
            raw_button = float(np.clip(raw_button, 0.0, 10.0))

            mug_pen_raw = _mug_drift_penalty(1.0)  # [-1, 0], scaled inside phase band

            if obj_to_target <= self._button_press_success_thresh:
                button_pressed = 1.0
                raw_button = 10.0
                self.button_completed = True
                self._target_pos = self._pull_target.copy()

            reward = 3.0 + (raw_button + mug_pen_raw) * 0.25  # ~[2.75, 5.5]

            # Normalise the overall reward to [-1, 1]
            reward = (reward - 5.0) / 5.0
            return (reward, tcp_to_obj, tcp_opened, obj_to_target, align_progress, button_pressed)

        # =============================================================
        # PHASE 2.5: BUTTON RETREAT  —  reward in [5.5, 6.0]
        #   Linear in distance to rest pose.
        # =============================================================
        elif not self.button_retreated:
            if not self._button_retreat_init_computed:
                self._button_retreat_dist_init = max(
                    float(np.linalg.norm(tcp - self._rest_pos)), 0.01
                )
                self._button_retreat_init_computed = True

            dist_to_rest = float(np.linalg.norm(tcp - self._rest_pos))
            retreat_frac = float(
                np.clip(1.0 - dist_to_rest / self._button_retreat_dist_init, 0.0, 1.0)
            )

            reward = 5.5 + retreat_frac * 0.5  # [5.5, 6.0]

            if dist_to_rest < self._retreat_complete_thresh:
                self.button_retreated = True
            
            # Normalise the overall reward to [-1, 1]
            reward = (reward - 5.0) / 5.0

            return (reward, dist_to_rest, tcp_opened, dist_to_rest, retreat_frac, 0.0)

        # =============================================================
        # PHASE 3: PULL MUG  —  reward in [6.0, 10.0]
        # =============================================================
        else:
            if not self._pull_init_computed:
                mug_pos = self.get_body_com("obj")
                self._pull_grasp_dist_init = float(
                    np.linalg.norm(tcp - (mug_pos + self._pull_grasp_offset))
                )
                self._pull_obj_to_target_init = float(
                    np.linalg.norm(mug_pos - self._pull_target)
                )
                self._pull_init_computed = True

            target = self._pull_target.copy()
            grasp_target = obj_pos + self._pull_grasp_offset
            tcp_to_obj = float(np.linalg.norm(obj_pos - tcp))
            grasp_dist = float(np.linalg.norm(tcp - grasp_target))
            obj_to_target = float(np.linalg.norm(obj_pos - target))

            grasp_reward = self._progress_fraction(
                grasp_dist, self._pull_grasp_dist_init, self._pull_grasp_success_thresh
            )
            in_place = self._progress_fraction(
                obj_to_target, self._pull_obj_to_target_init, self._pull_success_thresh
            )

            raw_pull = 10.0 * (0.7 * grasp_reward + 0.3 * in_place)
            if obj_to_target <= self._pull_success_thresh:
                in_place = 1.0
                raw_pull = 10.0
                self.pull_completed = True

            reward = 6.0 + float(np.clip(raw_pull, 0.0, 10.0)) * 0.4  # [6.0, 10.0]


            # Normalise the overall reward to [-1, 1]
            reward = (reward - 5.0) / 5.0

            return (reward, tcp_to_obj, tcp_opened, obj_to_target, grasp_reward, in_place)

    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        (
            reward,
            tcp_to_obj,
            tcp_opened,
            dist_to_target,
            component_1,
            component_2,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(self.pull_completed),
            "push_completed": float(self.push_completed),
            "push_retreated": float(self.push_retreated),
            "button_completed": float(self.button_completed),
            "button_retreated": float(self.button_retreated),
            "pull_completed": float(self.pull_completed),
            "near_object": float(tcp_to_obj <= 0.03),
            "obj_to_target": dist_to_target,
            "unscaled_reward": reward,
        }

        return reward, info
