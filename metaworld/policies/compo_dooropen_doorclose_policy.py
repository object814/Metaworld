from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class CompoDoorOpenDoorClosePolicy(Policy):
    """Expert policy for the compositional Door-Open + Door-Close task.

    The task has two phases:
      Phase 1 — Open the door by grabbing the handle and pulling
                (from SawyerDoorOpenV3Policy).
      Transition — After the door is fully open, release the handle
                and raise the gripper upward to disengage cleanly.
      Phase 2 — Close the door by pushing it back from the open side
                (from SawyerDoorCloseV3Policy).

    Transition detection: the goal position in the observation shifts
    significantly (~0.5 m) when the environment flags the door as
    opened.  This jump is a reliable signal for phase switching.

    This policy is **stateful**: call ``reset()`` whenever the environment
    is reset so that the internal phase machine restarts correctly.
    """

    def __init__(self) -> None:
        super().__init__()
        self._phase: str = "door_open"
        self._prev_goal: npt.NDArray[np.float64] | None = None

    def reset(self) -> None:
        """Reset internal state.  Must be called when the environment resets."""
        self._phase = "door_open"
        self._prev_goal = None

    # ------------------------------------------------------------------
    # Observation parsing
    # ------------------------------------------------------------------
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(
        obs: npt.NDArray[np.float64],
    ) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "gripper": obs[3],
            "door_pos": obs[4:7],
            "unused_info": obs[7:-3],
            "goal_pos": obs[-3:],
        }

    # ------------------------------------------------------------------
    # Main action selection
    # ------------------------------------------------------------------
    def get_action(
        self, obs: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)
        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        # --- Detect the environment's phase transition ----------------
        # When the door is fully opened the environment switches the
        # goal from the open-target to the close-target.  The shift
        # magnitude (~0.56 m) reliably signals the transition.
        if self._prev_goal is not None and self._phase == "door_open":
            goal_delta = float(np.linalg.norm(o_d["goal_pos"] - self._prev_goal))
            if goal_delta > 0.1:
                self._phase = "raise"
        self._prev_goal = o_d["goal_pos"].copy()

        # --- Phase 1: Open the door -----------------------------------
        if self._phase == "door_open":
            action["delta_pos"] = move(
                o_d["hand_pos"],
                self._door_open_desired_pos(o_d),
                p=25.0,
            )
            action["grab_effort"] = 1.0

        # --- Transition: Raise gripper to clear the handle ------------
        elif self._phase == "raise":
            pos_curr = o_d["hand_pos"]
            pos_door = o_d["door_pos"]
            # Move straight up, well above the handle
            target = np.array([pos_curr[0], pos_curr[1], pos_door[2] + 0.3])
            action["delta_pos"] = move(pos_curr, target, p=10.0)
            action["grab_effort"] = -1.0  # release gripper
            # Once the hand is sufficiently above the handle, proceed
            if pos_curr[2] > pos_door[2] + 0.2:
                self._phase = "door_close"

        # --- Phase 2: Close the door ----------------------------------
        elif self._phase == "door_close":
            action["delta_pos"] = move(
                o_d["hand_pos"],
                self._door_close_desired_pos(o_d),
                p=25.0,
            )
            action["grab_effort"] = 1.0

        return action.array

    # ------------------------------------------------------------------
    # Phase 1 helpers — Door Open (from SawyerDoorOpenV3Policy)
    # ------------------------------------------------------------------
    @staticmethod
    def _door_open_desired_pos(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> npt.NDArray[Any]:
        """Target position for the door-open phase.

        Sequence (mirrors SawyerDoorOpenV3Policy):
        1. Fly to a point above and slightly in front of the door handle.
        2. Drop down onto the front edge of the handle.
        3. Push inward toward the handle centroid to swing the door open.
        """
        pos_curr = o_d["hand_pos"]
        pos_door = o_d["door_pos"].copy()
        pos_door[0] -= 0.05

        # Align end-effector above the door handle
        if np.linalg.norm(pos_curr[:2] - pos_door[:2]) > 0.12:
            return pos_door + np.array([0.06, 0.02, 0.2])
        # Drop down on front edge of door handle
        elif abs(pos_curr[2] - pos_door[2]) > 0.04:
            return pos_door + np.array([0.06, 0.02, 0.0])
        # Push from front edge toward door handle's centroid
        else:
            return pos_door

    # ------------------------------------------------------------------
    # Phase 2 helpers — Door Close (from SawyerDoorCloseV3Policy)
    # ------------------------------------------------------------------
    @staticmethod
    def _door_close_desired_pos(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> npt.NDArray[Any]:
        """Target position for the door-close phase.

        Sequence (mirrors SawyerDoorCloseV3Policy):
        1. If to the right of the handle and below it, rise above it.
        2. If to the right, move left toward the handle in the XY plane.
        3. Descend onto the outer edge of the handle.
        4. Push toward the goal (closed position).
        """
        pos_curr = o_d["hand_pos"]
        pos_door = o_d["door_pos"].copy()
        pos_door += np.array([0.05, 0.12, 0.1])
        pos_goal = o_d["goal_pos"]

        # If to the right of the door handle
        if pos_curr[0] > pos_door[0]:
            # If below the door handle by more than 0.2
            if pos_curr[2] < pos_door[2] + 0.2:
                # Rise above door handle by ~0.2
                return np.array([pos_curr[0], pos_curr[1], pos_door[2] + 0.25])
            else:
                # Move toward door handle in XY plane
                return np.array([pos_door[0] - 0.02, pos_door[1], pos_curr[2]])
        # Put end effector on the outer edge of door handle (still above it)
        elif abs(pos_curr[2] - pos_door[2]) > 0.04:
            return pos_door + np.array([-0.02, 0.0, 0.0])
        # Push from outer edge toward the goal (closed position)
        else:
            return pos_goal
