from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class CompoAssemblyDisassemblyPolicy(Policy):
    """Expert policy for the compositional Assembly + Disassembly task.

    The task has two phases:
      Phase 1 — Assembly: Pick up the wrench from the table, fly it above
                the peg, and slide it all the way down to the table (z=0.02).
      Phase 2 — Disassembly: Grip the wrench firmly and lift it straight
                up to a fixed height well above the peg.

    Transition detection: the goal position in the observation shifts
    (~0.15+ m) when the environment flags assembly as complete.  After
    assembly the gripper is still holding the wrench at the base of the
    peg, so the disassembly lift begins immediately.

    This policy is **stateful**: call ``reset()`` whenever the environment
    is reset so that the internal phase machine restarts correctly.
    """

    def __init__(self) -> None:
        super().__init__()
        self._phase: str = "assembly"
        self._prev_goal: npt.NDArray[np.float64] | None = None

    def reset(self) -> None:
        """Reset internal state.  Must be called when the environment resets."""
        self._phase = "assembly"
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
            "goal_pos": obs[-3:],
            "unused_info": obs[7:-3],
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
        # When assembly succeeds the goal shifts from the peg-top to
        # peg-top + [0, 0, 0.15].  The ~0.15 m jump is a reliable signal.
        if self._prev_goal is not None and self._phase == "assembly":
            goal_delta = float(np.linalg.norm(o_d["goal_pos"] - self._prev_goal))
            if goal_delta > 0.1:
                self._phase = "disassembly"
        self._prev_goal = o_d["goal_pos"].copy()

        # --- Phase 1: Assembly ----------------------------------------
        if self._phase == "assembly":
            action["delta_pos"] = move(
                o_d["hand_pos"],
                self._assembly_desired_pos(o_d),
                p=10.0,
            )
            action["grab_effort"] = self._assembly_grab_effort(o_d)

        # --- Phase 2: Disassembly -------------------------------------
        else:
            action["delta_pos"] = move(
                o_d["hand_pos"],
                self._disassembly_desired_pos(o_d),
                p=10.0,
            )
            action["grab_effort"] = self._disassembly_grab_effort(o_d)

        return action.array

    # ------------------------------------------------------------------
    # Phase 1 helpers — Assembly (from SawyerAssemblyV3Policy)
    # ------------------------------------------------------------------
    @staticmethod
    def _assembly_desired_pos(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> npt.NDArray[Any]:
        """Target position for the assembly phase.

        Sequence:
        1. Align XY above the wrench.
        2. Descend onto the wrench.
        3. Lift to peg height.
        4. Move horizontally to peg XY.
        5. Drop onto the peg.
        """
        pos_curr = o_d["hand_pos"]
        pos_wrench = o_d["wrench_pos"] + np.array([-0.02, 0.0, 0.0])
        pos_peg = o_d["goal_pos"] + np.array([0.12, 0.0, 0.14])

        # 1. Align XY above the wrench
        if np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02:
            return pos_wrench + np.array([0.0, 0.0, 0.1])
        # 5. If already lined up with peg in XY, drop down onto it
        elif np.linalg.norm(pos_curr[:2] - pos_peg[:2]) <= 0.02:
            return pos_peg + np.array([0.0, 0.0, -0.2])
        # 2. Descend onto the wrench
        elif abs(pos_curr[2] - pos_wrench[2]) > 0.05:
            return pos_wrench + np.array([0.0, 0.0, 0.03])
        # 3. Lift to peg height
        elif abs(pos_curr[2] - pos_peg[2]) > 0.04:
            return np.array([pos_curr[0], pos_curr[1], pos_peg[2]])
        # 4. Move horizontally to peg XY
        else:
            return pos_peg

    @staticmethod
    def _assembly_grab_effort(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> float:
        """Gripper effort for the assembly phase."""
        pos_curr = o_d["hand_pos"]
        pos_wrench = o_d["wrench_pos"] + np.array([-0.02, 0.0, 0.0])

        if (
            np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02
            or abs(pos_curr[2] - pos_wrench[2]) > 0.12
        ):
            return 0.0
        # Hold wrench while transporting / placing on peg
        return 0.6

    # ------------------------------------------------------------------
    # Phase 2 helpers — Disassembly (from SawyerDisassembleV3Policy)
    # ------------------------------------------------------------------
    @staticmethod
    def _disassembly_desired_pos(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> npt.NDArray[Any]:
        """Target position for the disassembly phase.

        After assembly the gripper is already near the wrench at the
        bottom of the peg, so it immediately enters the "lift" branch.
        Uses a fixed height target for decisive, non-oscillating lift.
        """
        pos_curr = o_d["hand_pos"]
        pos_wrench = o_d["wrench_pos"] + np.array([-0.02, 0.0, 0.0])

        # 1. Align XY above the wrench (if somehow misaligned)
        if np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02:
            return pos_wrench + np.array([0.0, 0.0, 0.1])
        # 2. Descend onto wrench
        elif abs(pos_curr[2] - pos_wrench[2]) > 0.03:
            return pos_wrench + np.array([0.0, 0.0, 0.03])
        # 3. Lift to a fixed height well above the peg
        else:
            return np.array([pos_curr[0], pos_curr[1], 0.35])

    @staticmethod
    def _disassembly_grab_effort(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> float:
        """Gripper effort for the disassembly phase."""
        pos_curr = o_d["hand_pos"]
        pos_wrench = o_d["wrench_pos"] + np.array([-0.02, 0.0, 0.0])

        if (
            np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02
            or abs(pos_curr[2] - pos_wrench[2]) > 0.15
        ):
            return 0.0
        return 0.8
