from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class CompoAssemblyDisassemblyPolicy(Policy):
    """Expert policy for the compositional Disassemble + Assemble task.

    The task has two phases:
      Phase 1 — Disassemble: grasp the nut on the peg and lift it off
                (from SawyerDisassembleV3Policy).
      Phase 2 — Assemble: while still holding the nut, carry it over
                and place it back onto the peg
                (from SawyerAssemblyV3Policy).

    The agent keeps its grip on the nut across phases — no release
    is needed since the nut must be placed back on the same peg.

    Transition detection: the goal position in the observation shifts
    when the environment flags the nut as disassembled (target switches
    from the disassemble-target to the assemble-target).

    This policy is **stateful**: call ``reset()`` whenever the environment
    is reset so that the internal phase machine restarts correctly.
    """

    def __init__(self) -> None:
        super().__init__()
        self._phase: str = "disassemble"
        self._prev_goal: npt.NDArray[np.float64] | None = None

    def reset(self) -> None:
        """Reset internal state.  Must be called when the environment resets."""
        self._phase = "disassemble"
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
            "wrench_pos": obs[4:7],
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
        # When the nut is lifted off the peg, the environment switches
        # the goal from the disassemble-target to the assemble-target.
        # The shift magnitude reliably signals the transition.
        if self._prev_goal is not None and self._phase == "disassemble":
            goal_delta = float(np.linalg.norm(o_d["goal_pos"] - self._prev_goal))
            if goal_delta > 0.05:
                self._phase = "assemble"
        self._prev_goal = o_d["goal_pos"].copy()

        # --- Phase 1: Disassemble (lift nut off peg) -----------------
        if self._phase == "disassemble":
            action["delta_pos"] = move(
                o_d["hand_pos"],
                self._disassemble_desired_pos(o_d),
                p=10.0,
            )
            action["grab_effort"] = self._disassemble_grab_effort(o_d)

        # --- Phase 2: Assemble (place nut onto peg) ------------------
        # The agent keeps holding the nut from Phase 1 and goes
        # straight to placing it back on the peg.
        elif self._phase == "assemble":
            action["delta_pos"] = move(
                o_d["hand_pos"],
                self._assemble_desired_pos(o_d),
                p=10.0,
            )
            action["grab_effort"] = self._assemble_grab_effort(o_d)

        return action.array

    # ------------------------------------------------------------------
    # Phase 1 helpers — Disassemble (from SawyerDisassembleV3Policy)
    # ------------------------------------------------------------------
    @staticmethod
    def _disassemble_desired_pos(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> npt.NDArray[Any]:
        """Target position for the disassembly phase.

        Sequence (mirrors SawyerDisassembleV3Policy):
        1. Fly to a point above the wrench (nut handle).
        2. Drop down onto the wrench.
        3. Wait for gripper to close.
        4. Lift upwards to pull nut off peg.
        """
        pos_curr = o_d["hand_pos"]
        pos_wrench = o_d["wrench_pos"] + np.array([-0.02, 0.0, 0.01])
        gripper = o_d["gripper"]

        # If XY error is greater than 0.02, place end effector above the wrench
        if np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02:
            return pos_wrench + np.array([0.0, 0.0, 0.1])
        # Once XY error is low enough, drop end effector down on top of wrench
        elif abs(pos_curr[2] - pos_wrench[2]) > 0.05:
            return pos_wrench
        # Wait for gripper to close around wrench before lifting
        elif gripper > 0.7:
            return pos_wrench
        # Gripper closed, move upwards
        else:
            return pos_wrench + np.array([0.0, 0.0, 0.3])

    @staticmethod
    def _disassemble_grab_effort(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> float:
        """Grab effort for the disassembly phase."""
        pos_curr = o_d["hand_pos"]
        pos_wrench = o_d["wrench_pos"] + np.array([-0.02, 0.0, 0.01])

        if (
            np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02
            or abs(pos_curr[2] - pos_wrench[2]) > 0.1
        ):
            return 0.0
        else:
            return 0.8

    # ------------------------------------------------------------------
    # Phase 2 helpers — Assemble (from SawyerAssemblyV3Policy)
    # ------------------------------------------------------------------
    @staticmethod
    def _assemble_desired_pos(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> npt.NDArray[Any]:
        """Target position for the assembly phase.

        Sequence (mirrors SawyerAssemblyV3Policy):
        1. Fly to a point above the wrench (nut on the table).
        2. Drop down onto the wrench.
        3. Lift to peg height.
        4. Move XY to align with peg.
        5. Drop down onto the peg.
        """
        pos_curr = o_d["hand_pos"]
        pos_wrench = o_d["wrench_pos"] + np.array([-0.02, 0.0, 0.0])
        pos_peg = o_d["goal_pos"] + np.array([0.12, 0.0, 0.24])

        # If XY error is greater than 0.02, place end effector above the wrench
        if np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02:
            return pos_wrench + np.array([0.0, 0.0, 0.1])
        # (For later) if lined up with peg, drop down on top of it
        elif np.linalg.norm(pos_curr[:2] - pos_peg[:2]) <= 0.02:
            return pos_peg + np.array([0.0, 0.0, -0.2])
        # Once XY error is low enough, drop end effector down on top of wrench
        elif abs(pos_curr[2] - pos_wrench[2]) > 0.05:
            return pos_wrench + np.array([0.0, 0.0, 0.03])
        # If not at the same Z height as the goal, move up to that plane
        elif abs(pos_curr[2] - pos_peg[2]) > 0.04:
            return np.array([pos_curr[0], pos_curr[1], pos_peg[2]])
        # Move XY toward peg
        else:
            return pos_peg

    @staticmethod
    def _assemble_grab_effort(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> float:
        """Grab effort for the assembly phase."""
        pos_curr = o_d["hand_pos"]
        pos_wrench = o_d["wrench_pos"] + np.array([-0.02, 0.0, 0.0])

        if (
            np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02
            or abs(pos_curr[2] - pos_wrench[2]) > 0.12
        ):
            return 0.0
        else:
            return 0.6
