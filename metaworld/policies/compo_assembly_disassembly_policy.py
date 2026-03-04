from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class CompoAssemblyDisassemblyPolicy(Policy):
    """Expert policy for the compositional Assemble + Disassemble task.

        The task has two phases:
            Phase 1 — Assemble: grasp the nut on the table and place it on the peg
                                (from SawyerAssemblyV3Policy).
            Phase 2 — Disassemble: while still holding the nut, lift it off
                                (from SawyerDisassembleV3Policy).

    The agent keeps its grip on the nut across phases — no release
    is needed since the nut must be placed back on the same peg.

    Transition detection: the goal position in the observation shifts
    when the environment flags the nut as assembled (target switches
    from the assemble-target to the disassemble-target).

    This policy is **stateful**: call ``reset()`` whenever the environment
    is reset so that the internal phase machine restarts correctly.
    """

    def __init__(self) -> None:
        super().__init__()
        self._phase: str = "assemble"
        self._prev_goal: npt.NDArray[np.float64] | None = None
        self._seen_lifted_in_assemble: bool = False

    def reset(self) -> None:
        """Reset internal state.  Must be called when the environment resets."""
        self._phase = "assemble"
        self._prev_goal = None
        self._seen_lifted_in_assemble = False

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

        # --- Detect phase transition from object state ----------------
        # Goal observation can stay unchanged in partially-observable
        # setups, so we detect assembly completion by trajectory pattern:
        # (1) nut was lifted, then (2) nut is seated near table height
        # with the hand still close to it.
        wrench_z = float(o_d["wrench_pos"][2])
        hand_wrench_xy = float(
            np.linalg.norm(o_d["hand_pos"][:2] - o_d["wrench_pos"][:2])
        )

        if self._phase == "assemble":
            if wrench_z > 0.08:
                self._seen_lifted_in_assemble = True

            seated_after_lift = (
                self._seen_lifted_in_assemble
                and wrench_z < 0.04
                and hand_wrench_xy < 0.03
            )
            if seated_after_lift:
                self._phase = "disassemble"
        self._prev_goal = o_d["goal_pos"].copy()

        # --- Phase 1: Assemble (place nut onto peg) ------------------
        if self._phase == "assemble":
            action["delta_pos"] = move(
                o_d["hand_pos"],
                self._assemble_desired_pos(o_d),
                p=10.0,
            )
            action["grab_effort"] = self._assemble_grab_effort(o_d)

        # --- Phase 2: Disassemble (lift nut off peg) -----------------
        # The agent keeps holding the nut from Phase 1 and goes
        # straight to lifting it off the peg.
        elif self._phase == "disassemble":
            action["delta_pos"] = move(
                o_d["hand_pos"],
                self._disassemble_desired_pos(o_d),
                p=10.0,
            )
            action["grab_effort"] = self._disassemble_grab_effort(o_d)

        return action.array

    # ------------------------------------------------------------------
    # Phase 1 helpers — Assemble (from SawyerAssemblyV3Policy)
    # ------------------------------------------------------------------
    @staticmethod
    def _assemble_desired_pos(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> npt.NDArray[Any]:
        """Target position for the assembly phase.

        Sequence (mirrors SawyerAssemblyV3Policy):
        1. Fly to a point above the wrench (nut handle).
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

    # ------------------------------------------------------------------
    # Phase 2 helpers — Disassemble (from SawyerDisassembleV3Policy)
    # ------------------------------------------------------------------
    @staticmethod
    def _disassemble_desired_pos(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> npt.NDArray[Any]:
        """Target position for the disassembly phase.

        Sequence (mirrors SawyerDisassembleV3Policy):
        1. Fly to a point above the wrench (nut on the table).
        2. Drop down onto the wrench.
        3. Wait for gripper to close.
        4. Lift upwards to pull nut off peg.
        """
        pos_curr = o_d["hand_pos"]
        pos_wrench = o_d["wrench_pos"] + np.array([-0.02, 0.0, 0.01])
        lift_goal = pos_wrench + np.array([0.0, 0.0, 0.25])

        # If XY error is greater than 0.02, place end effector above the wrench
        if np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02:
            return pos_wrench + np.array([0.0, 0.0, 0.1])
        # Once XY error is low enough, drop end effector down on top of wrench
        elif abs(pos_curr[2] - pos_wrench[2]) > 0.05:
            return pos_wrench
        # Gripper on wrench, move upward past the disassemble target
        else:
            return lift_goal

    @staticmethod
    def _disassemble_grab_effort(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> float:
        """Grab effort for the disassembly phase."""
        pos_curr = o_d["hand_pos"]
        pos_wrench = o_d["wrench_pos"] + np.array([-0.02, 0.0, 0.01])

        if np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.03:
            return 0.0
        return 0.9
