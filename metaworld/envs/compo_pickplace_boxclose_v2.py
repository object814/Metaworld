from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import CompoPickPlaceBoxCloseInitConfigDict
from metaworld.utils import reward_utils
import mujoco


class CompoPickPlaceBoxCloseEnv(SawyerXYZEnv):
    """
    Compositional task: Pick-place block into box, then close the box lid.

    Phase 1: Pick up the red block and place it inside the open box.
    Phase 2: Pick up the lid and close the box.
    """

    LID_INIT_Z = -0.02
    # Keep the lid separated from the box and block
    BOX_LID_MIN_DIST = 0.35
    # Avoid initial contacts between the block and the box walls.
    BOX_BLOCK_MIN_DIST = 0.20
    # Avoid initial contacts between the block and the lid.
    BLOCK_LID_MIN_DIST = 0.20

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

        box_low = (-0.25, 0.45)
        box_high = (0.25, 0.80)
        obj_low = (-0.25, 0.45)
        obj_high = (0.25, 0.80)
        lid_low = (-0.25, 0.45)
        lid_high = (0.25, 0.80)

        # Task-specific flags
        self.pickplace_completed = False
        self.boxclose_completed = False
        self.lift_completed = False
        self.liftThresh = 0.12

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            height=height,
            width=width,
        )

        self.init_config: CompoPickPlaceBoxCloseInitConfigDict = {
            "hand_init_pos": np.array([0, 0.6, 0.2]),
            "box_init_pos": np.array([0.0, 0.8, 0.0]),
            "obj_init_pos": np.array([0.15, 0.6, 0.02]),
            "lid_init_pos": np.array([-0.15, 0.6, self.LID_INIT_Z]),
        }

        self.hand_init_pos = self.init_config["hand_init_pos"]
        self.box_init_pos = self.init_config["box_init_pos"]
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.lid_init_pos = self.init_config["lid_init_pos"]

        # [box_x, box_y, block_x, block_y, lid_x, lid_y]
        self._random_reset_space = Box(
            np.array([box_low[0], box_low[1],
                       obj_low[0], obj_low[1],
                       lid_low[0], lid_low[1]]),
            np.array([box_high[0], box_high[1],
                       obj_high[0], obj_high[1],
                       lid_high[0], lid_high[1]]),
            dtype=np.float64,
        )
        self.goal_space = Box(
            np.array(hand_low), np.array(hand_high), dtype=np.float64
        )

        self._target_pos = np.zeros(3)
        self._lid_target_pos = np.zeros(3)

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_xyz/compo_pickplace_boxclose.xml")

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        """
        Phase 1: Returns block position.
        Phase 2: Returns lid (top_link) position.
        """
        if not self.pickplace_completed:
            return self.get_body_com("obj")
        else:
            return self.get_body_com("top_link")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        if not self.pickplace_completed:
            return Rotation.from_matrix(
                self.data.geom("objGeom").xmat.reshape(3, 3)
            ).as_quat()
        else:
            return self.data.body("top_link").xquat

    def _sample_non_overlapping_positions(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Sample box, block, and lid XY positions with pairwise separation constraints."""
        assert self._random_reset_space is not None
        low = self._random_reset_space.low
        high = self._random_reset_space.high
        rng = self.np_random if self.seeded_rand_vec else np.random

        for _ in range(1000):
            rand_vec = rng.uniform(low, high, size=low.size).astype(np.float64)
            box_xy = rand_vec[:2]
            block_xy = rand_vec[2:4]
            lid_xy = rand_vec[4:6]

            lid_box_ok = (
                np.linalg.norm(lid_xy - box_xy) >= self.BOX_LID_MIN_DIST
            )
            block_box_ok = (
                np.linalg.norm(block_xy - box_xy) >= self.BOX_BLOCK_MIN_DIST
            )
            block_lid_ok = (
                np.linalg.norm(block_xy - lid_xy) >= self.BLOCK_LID_MIN_DIST
            )
            if lid_box_ok and block_box_ok and block_lid_ok:
                self._last_rand_vec = rand_vec
                return box_xy, block_xy, lid_xy

        # Deterministic safe fallback inside the configured ranges.
        fallback = np.array([0.1, 0.85, 0.25, 0.5, -0.25, 0.5], dtype=np.float64)
        self._last_rand_vec = fallback
        return fallback[:2].copy(), fallback[2:4].copy(), fallback[4:6].copy()

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()

        # Reset task flags
        self.pickplace_completed = False
        self.boxclose_completed = False
        self.lift_completed = False

        # Sample randomised positions with overlap prevention.
        box_xy, block_xy, lid_xy = self._sample_non_overlapping_positions()

        # --- Box container ---
        box_height = self.get_body_com("boxbody")[2]
        self.model.body("boxbody").pos = np.array(
            [box_xy[0], box_xy[1], box_height]
        )
        self.box_init_pos = np.array([box_xy[0], box_xy[1], box_height])

        # --- Block (free joint "obj") ---
        block_pos = np.array([block_xy[0], block_xy[1], 0.02])
        block_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj"
        )
        block_qpos_adr = self.model.jnt_qposadr[block_joint_id]
        block_qvel_adr = self.model.jnt_dofadr[block_joint_id]
        self.data.qpos[block_qpos_adr : block_qpos_adr + 3] = block_pos
        self.data.qpos[block_qpos_adr + 3 : block_qpos_adr + 7] = [1, 0, 0, 0]
        self.data.qvel[block_qvel_adr : block_qvel_adr + 6] = 0.0
        self.obj_init_pos = block_pos

        # --- Lid (free joint "lid_joint") ---
        lid_pos = np.array([lid_xy[0], lid_xy[1], self.LID_INIT_Z])
        lid_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "lid_joint"
        )
        lid_qpos_adr = self.model.jnt_qposadr[lid_joint_id]
        lid_qvel_adr = self.model.jnt_dofadr[lid_joint_id]
        self.data.qpos[lid_qpos_adr : lid_qpos_adr + 3] = lid_pos
        # Upright lid orientation matching sawyer_box.xml quat="1 0 0 1"
        self.data.qpos[lid_qpos_adr + 3 : lid_qpos_adr + 7] = [
            0.7071, 0, 0, 0.7071
        ]
        self.data.qvel[lid_qvel_adr : lid_qvel_adr + 6] = 0.0
        self.lid_init_pos = lid_pos

        # Apply changes
        mujoco.mj_forward(self.model, self.data)

        # Cache initial gripper state
        self.init_tcp = self.tcp_center
        self.init_left_pad = self.get_body_com("leftpad")
        self.init_right_pad = self.get_body_com("rightpad")

        # Phase 1 target: block inside the box
        # Box floor top ~box_height+0.01, block half-height 0.02
        self._target_pos = np.array(
            [box_xy[0], box_xy[1], box_height + 0.04]
        )

        # Phase 2 target: lid on top of box
        self._lid_target_pos = np.array(
            [box_xy[0], box_xy[1], box_height + 0.073]
        )

        # Height params for the pick phase
        self.objHeight = self.data.geom("objGeom").xpos[2]
        self.heightTarget = self.objHeight + 0.04

        # Visualisation
        self.model.site("goal").pos = self._target_pos

        return self._get_obs()

    def _gripper_caging_reward(
        self,
        action: npt.NDArray[np.float32],
        obj_pos: npt.NDArray[Any],
    ) -> float:
        pad_success_margin = 0.05
        x_z_success_margin = 0.005
        obj_radius = 0.015
        tcp = self.tcp_center
        left_pad = self.get_body_com("leftpad")
        right_pad = self.get_body_com("rightpad")
        delta_object_y_left_pad = left_pad[1] - obj_pos[1]
        delta_object_y_right_pad = obj_pos[1] - right_pad[1]
        right_caging_margin = abs(
            abs(obj_pos[1] - self.init_right_pad[1]) - pad_success_margin
        )
        left_caging_margin = abs(
            abs(obj_pos[1] - self.init_left_pad[1]) - pad_success_margin
        )

        right_caging = reward_utils.tolerance(
            delta_object_y_right_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=right_caging_margin,
            sigmoid="long_tail",
        )
        left_caging = reward_utils.tolerance(
            delta_object_y_left_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=left_caging_margin,
            sigmoid="long_tail",
        )

        y_caging = reward_utils.hamacher_product(left_caging, right_caging)

        tcp_xz = tcp + np.array([0.0, -tcp[1], 0.0])
        obj_position_x_z = np.copy(obj_pos) + np.array([0.0, -obj_pos[1], 0.0])
        tcp_obj_norm_x_z = float(
            np.linalg.norm(tcp_xz - obj_position_x_z, ord=2)
        )

        assert self.obj_init_pos is not None
        init_obj_x_z = self.obj_init_pos + np.array(
            [0.0, -self.obj_init_pos[1], 0.0]
        )
        init_tcp_x_z = self.init_tcp + np.array(
            [0.0, -self.init_tcp[1], 0.0]
        )
        tcp_obj_x_z_margin = (
            np.linalg.norm(init_obj_x_z - init_tcp_x_z, ord=2)
            - x_z_success_margin
        )

        x_z_caging = reward_utils.tolerance(
            tcp_obj_norm_x_z,
            bounds=(0, x_z_success_margin),
            margin=tcp_obj_x_z_margin,
            sigmoid="long_tail",
        )

        caging = reward_utils.hamacher_product(y_caging, x_z_caging)
        gripper_closed = np.clip(action[-1], 0.0, 1.0)
        grasping = reward_utils.hamacher_product(caging, gripper_closed)
        return 0.7 * caging + 0.3 * grasping

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float]:
        assert (
            self._target_pos is not None and self.obj_init_pos is not None
        )

        gripper = obs[:3]
        tcp_opened = obs[3]
        obj_pos = obs[4:7]  # block in Phase 1, lid in Phase 2

        # ---------------------------------------------------------
        # PHASE 1: PICK AND PLACE BLOCK INTO BOX
        # ---------------------------------------------------------
        if not self.pickplace_completed:
            _TARGET_RADIUS: float = 0.05
            final_target = self._target_pos
            tcp_to_obj = float(np.linalg.norm(obj_pos - gripper))

            # Intermediate waypoint: directly above the block's initial
            # position at z=0.18, so the policy lifts the block over the box
            # side walls before descending into the box.
            lift_waypoint = np.array(
                [self.obj_init_pos[0], self.obj_init_pos[1], 0.18]
            )

            # =================================================
            # SUB-PHASE 1A: LIFT BLOCK TO WAYPOINT (above init pos)
            # raw reward in [0, 10]  ->  normalised [-1, -0.5]
            # =================================================
            if not self.lift_completed:
                obj_to_waypoint = float(
                    np.linalg.norm(obj_pos - lift_waypoint)
                )

                # Distance-to-waypoint shaping (monotone in obj_to_waypoint).
                lift_margin = float(
                    np.linalg.norm(self.obj_init_pos - lift_waypoint)
                )
                in_place_lift = reward_utils.tolerance(
                    obj_to_waypoint,
                    bounds=(0, _TARGET_RADIUS),
                    margin=max(lift_margin, 1e-6),
                    sigmoid="long_tail",
                )

                # Approach the block (monotone in tcp_to_obj).
                approach = reward_utils.tolerance(
                    tcp_to_obj,
                    bounds=(0, 0.04),
                    margin=float(
                        np.linalg.norm(self.obj_init_pos - self.init_tcp)
                    ),
                    sigmoid="long_tail",
                )

                object_grasped = self._gripper_caging_reward(action, obj_pos)
                grasp_and_lift = reward_utils.hamacher_product(
                    object_grasped, in_place_lift
                )

                # Component budget (max 10), all monotonically nondecreasing
                # as the gripper approaches, grasps, and lifts the block:
                #   1.0 * approach             -> [0, 1.0]
                #   2.0 * object_grasped       -> [0, 2.0]
                #   3.0 * in_place_lift        -> [0, 3.0]
                #   4.0 * grasp_and_lift       -> [0, 4.0]
                reward = (
                    1.0 * approach
                    + 2.0 * object_grasped
                    + 3.0 * in_place_lift
                    + 4.0 * grasp_and_lift
                )
                reward = float(np.clip(reward, 0.0, 10.0))

                # Transition: block at waypoint while still grasped.
                if obj_to_waypoint < _TARGET_RADIUS:
                    self.lift_completed = True

                # Normalise [0, 10] -> [-1, -0.5]
                reward = reward / 20.0 - 1.0

                in_place = in_place_lift
                obj_to_target = obj_to_waypoint
                return (
                    reward,
                    tcp_to_obj,
                    tcp_opened,
                    obj_to_target,
                    object_grasped,
                    in_place,
                )

            # =================================================
            # SUB-PHASE 1B: PLACE BLOCK INTO BOX
            # raw reward in [0, 10]  ->  normalised [-0.5, 0.25]
            # =================================================
            obj_to_target = float(np.linalg.norm(obj_pos - final_target))
            in_place_margin = float(
                np.linalg.norm(lift_waypoint - final_target)
            )

            in_place = reward_utils.tolerance(
                obj_to_target,
                bounds=(0, _TARGET_RADIUS),
                margin=max(in_place_margin, 1e-6),
                sigmoid="long_tail",
            )

            approach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, 0.04),
                margin=float(
                    np.linalg.norm(self.obj_init_pos - self.init_tcp)
                ),
                sigmoid="long_tail",
            )

            object_grasped = self._gripper_caging_reward(action, obj_pos)
            in_place_and_grasped = reward_utils.hamacher_product(
                object_grasped, in_place
            )
            tcp_release = float(np.clip(tcp_opened, 0.0, 1.0))

            # Component budget (max 10), each term monotone in progress:
            #   0.5 * approach               -> [0, 0.5]
            #   1.5 * object_grasped         -> [0, 1.5]
            #   4.0 * in_place               -> [0, 4.0]
            #   2.0 * in_place_and_grasped   -> [0, 2.0]
            #   2.0 * in_place * tcp_release -> [0, 2.0]  (release at target)
            reward = (
                0.5 * approach
                + 1.5 * object_grasped
                + 4.0 * in_place
                + 2.0 * in_place_and_grasped
                + 2.0 * in_place * tcp_release
            )
            reward = float(np.clip(reward, 0.0, 10.0))

            # Transition to Phase 2 requires the block at target AND released.
            if obj_to_target < _TARGET_RADIUS and tcp_opened > 0.8:
                reward = 10.0
                self.pickplace_completed = True
                # Prepare state for Phase 2
                self.obj_init_pos = self.lid_init_pos.copy()
                self._target_pos = self._lid_target_pos.copy()
                self.objHeight = self.get_body_com("top_link")[2]
                self.heightTarget = self.objHeight + self.liftThresh
                self.model.site("goal").pos = self._target_pos
                self.model.site("goal").rgba = np.array(
                    [1.0, 1.0, 1.0, 1.0]
                )

            # Normalise [0, 10] -> [-0.5, 0.25]
            reward = reward * 0.075 - 0.5

            return (
                reward,
                tcp_to_obj,
                tcp_opened,
                obj_to_target,
                object_grasped,
                in_place,
            )

        # ---------------------------------------------------------
        # PHASE 2: CLOSE THE BOX (PLACE LID ON BOX)
        # ---------------------------------------------------------
        else:
            # Penalty: keep block inside the box
            block_pos = self.get_body_com("obj")
            block_to_box_xy = float(
                np.linalg.norm(block_pos[:2] - self.box_init_pos[:2])
            )
            block_penalty = (
                reward_utils.tolerance(
                    block_to_box_xy,
                    bounds=(0, 0.05),
                    margin=0.2,
                    sigmoid="long_tail",
                )
                - 1.0
            )

            _TARGET_RADIUS: float = 0.05
            target = self._target_pos  # lid target = box top
            lid_pos = obj_pos  # obs[4:7] is lid in Phase 2

            lid_to_target = float(np.linalg.norm(lid_pos - target))
            tcp_to_lid = float(np.linalg.norm(lid_pos - gripper))
            in_place_margin = float(
                np.linalg.norm(self.obj_init_pos - target)
            )

            in_place = reward_utils.tolerance(
                lid_to_target,
                bounds=(0, _TARGET_RADIUS),
                margin=max(in_place_margin, 1e-6),
                sigmoid="long_tail",
            )

            approach = reward_utils.tolerance(
                tcp_to_lid,
                bounds=(0, 0.04),
                margin=float(
                    np.linalg.norm(self.obj_init_pos - self.init_tcp)
                ),
                sigmoid="long_tail",
            )

            object_grasped = self._gripper_caging_reward(action, lid_pos)

            in_place_and_object_grasped = reward_utils.hamacher_product(
                object_grasped, in_place
            )

            # Dense "reach" reward: linear decay over the full workspace
            # distance from tcp to the lid. The existing `approach` term
            # uses a long_tail tolerance with bounds=(0, 0.04), which is
            # nearly flat until the gripper is within ~10cm of the lid —
            # after releasing the block at the box, the gripper is 30-50cm
            # away from the lid, so `approach` provides essentially no
            # gradient and the policy has nothing to guide it across the
            # gap. `reach` below is a strictly monotone signal that grows
            # with every cm of progress toward the lid, so the critic
            # always has a direction to follow during the traversal.
            #
            # Margin is derived from the initial lid/hand separation with
            # a small safety pad, so the reward saturates to ~1 only when
            # the gripper is actually at the lid rather than partway.
            reach_margin = float(
                np.linalg.norm(self.lid_init_pos - self.init_tcp)
            ) + 0.1
            reach = max(
                0.0, 1.0 - tcp_to_lid / max(reach_margin, 1e-6)
            )

            # Rebalanced component budget (sum still ~10 before offset):
            #   0.5 * block_penalty           [-0.5, 0]   (soft anti-knockout)
            #   2.0 * reach                   [0, 2.0]    (dense far-field)
            #   0.5 * approach                [0, 0.5]    (near-field precision)
            #   1.5 * object_grasped          [0, 1.5]
            #   6.0 * in_place_and_grasped    [0, 6.0]
            reward = (
                0.5 * block_penalty
                + 2.0 * reach
                + 0.5 * approach
                + 1.5 * object_grasped
                + 6.0 * in_place_and_object_grasped
            )
            if lid_to_target < _TARGET_RADIUS:
                reward = 10.0

            # Clip raw to [0, 10] then normalise to [0.25, 1.0]
            reward = float(np.clip(reward, 0.0, 10.0))
            reward = reward * 0.075 + 0.25

            # Check box-close success
            lid_xy_aligned = (
                np.linalg.norm(lid_pos[:2] - target[:2]) < 0.03
            )
            lid_z_reached = np.abs(lid_pos[2] - target[2]) < 0.03
            if lid_xy_aligned and lid_z_reached:
                self.boxclose_completed = True

            return (
                reward,
                tcp_to_lid,
                tcp_opened,
                lid_to_target,
                object_grasped,
                in_place,
            )

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

        final_success = 0.0
        if self.pickplace_completed and self.boxclose_completed:
            final_success = 1.0

        info = {
            "success": final_success,
            "pickplace_completed": float(self.pickplace_completed),
            "boxclose_completed": float(self.boxclose_completed),
            "near_object": float(tcp_to_obj <= 0.03),
            "obj_to_target": dist_to_target,
            "unscaled_reward": reward,
        }

        return reward, info
