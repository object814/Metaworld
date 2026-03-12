from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class CompoCoffeePushButtonPullPolicy(Policy):
    """Expert policy for the compositional Coffee Push + Button + Pull task.

    Phases:
      1. push_mug            — Push the mug to the coffee machine.
      2. release_settle       — Open gripper, hold position to let mug settle.
      3. push_clear_mug       — Shift in x to avoid overlapping the mug.
      4. push_retreat_to_rest — Move arm to rest/starting position.
      5. press_button         — Align and press the coffee machine button.
      6. button_retreat_settle — Hold still after pressing button.
      7. button_retreat_to_rest — Move arm back to rest/starting position.
      8. pull_mug             — Grasp and pull the mug to the target.

    Phase transitions are detected by monitoring changes in the goal position
    (obs[-3:]), which the environment updates when each sub-task completes.
    """

    # The rest / starting pose the arm returns to between tasks
    REST_POS = np.array([0.0, 0.4, 0.2])

    def __init__(self) -> None:
        super().__init__()
        self._phase: str = "push_mug"
        self._counter: int = 0
        self._prev_goal: npt.NDArray[np.float64] | None = None
        self._transition_count: int = 0
        self._sideways_target: npt.NDArray[np.float64] | None = None

    def reset(self) -> None:
        """Reset internal state. Must be called when the environment resets."""
        self._phase = "push_mug"
        self._counter = 0
        self._prev_goal = None
        self._transition_count = 0
        self._sideways_target = None

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
            "obj_pos": obs[4:7],
            "obj_rot": obs[7:11],
            "goal_pos": obs[-3:],
            "unused_info": obs[11:-3],
        }

    # ------------------------------------------------------------------
    # Main action selection
    # ------------------------------------------------------------------
    def get_action(
        self, obs: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)
        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        # --- Detect goal changes for phase transitions ----------------
        if self._prev_goal is not None:
            goal_delta = float(np.linalg.norm(o_d["goal_pos"] - self._prev_goal))
            if goal_delta > 0.1:
                self._transition_count += 1
                if self._transition_count == 1:
                    # Push completed -> start retreat sequence
                    self._phase = "release_settle"
                    self._counter = 0
                elif self._transition_count == 2:
                    # Button completed -> start retreat sequence
                    self._phase = "button_retreat_settle"
                    self._counter = 0
        self._prev_goal = o_d["goal_pos"].copy()

        # --- Phase: push mug ------------------------------------------
        if self._phase == "push_mug":
            action["delta_pos"] = move(
                o_d["hand_pos"], self._push_desired_pos(o_d), p=10.0
            )
            action["grab_effort"] = self._push_grab_effort(o_d)

        # ==============================================================
        # PUSH -> BUTTON RETREAT SEQUENCE
        # ==============================================================

        # --- Phase: release and settle --------------------------------
        elif self._phase == "release_settle":
            action["delta_pos"] = np.zeros(3)
            action["grab_effort"] = -1.0
            self._counter += 1
            if self._counter >= 8:
                self._phase = "push_clear_mug"
                self._sideways_target = o_d["hand_pos"].copy()
                self._sideways_target[0] += 0.1

        # --- Phase: move sideways to clear the mug --------------------
        elif self._phase == "push_clear_mug":
            assert self._sideways_target is not None
            action["delta_pos"] = move(
                o_d["hand_pos"], self._sideways_target, p=10.0
            )
            action["grab_effort"] = -1.0
            if np.linalg.norm(o_d["hand_pos"] - self._sideways_target) < 0.02:
                self._phase = "push_retreat_to_rest"

        # --- Phase: move to rest position -----------------------------
        elif self._phase == "push_retreat_to_rest":
            action["delta_pos"] = move(
                o_d["hand_pos"], self.REST_POS, p=10.0
            )
            action["grab_effort"] = -1.0
            if np.linalg.norm(o_d["hand_pos"] - self.REST_POS) < 0.08:
                self._phase = "press_button"

        # --- Phase: press button --------------------------------------
        elif self._phase == "press_button":
            action["delta_pos"] = move(
                o_d["hand_pos"], self._button_desired_pos(o_d), p=10.0
            )
            action["grab_effort"] = -1.0

        # ==============================================================
        # BUTTON -> PULL RETREAT SEQUENCE
        # ==============================================================

        # --- Phase: settle after button press -------------------------
        elif self._phase == "button_retreat_settle":
            action["delta_pos"] = np.zeros(3)
            action["grab_effort"] = -1.0
            self._counter += 1
            if self._counter >= 5:
                self._phase = "button_retreat_to_rest"

        # --- Phase: move to rest position before pulling --------------
        elif self._phase == "button_retreat_to_rest":
            action["delta_pos"] = move(
                o_d["hand_pos"], self.REST_POS, p=10.0
            )
            action["grab_effort"] = -1.0
            if np.linalg.norm(o_d["hand_pos"] - self.REST_POS) < 0.08:
                self._phase = "pull_mug"

        # --- Phase: pull mug ------------------------------------------
        elif self._phase == "pull_mug":
            action["delta_pos"] = move(
                o_d["hand_pos"], self._pull_desired_pos(o_d), p=10.0
            )
            action["grab_effort"] = self._pull_grab_effort(o_d)

        return action.array

    # ------------------------------------------------------------------
    # Phase 1 helpers — push mug (from SawyerCoffeePushV3Policy)
    # ------------------------------------------------------------------
    @staticmethod
    def _push_desired_pos(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> npt.NDArray[Any]:
        pos_curr = o_d["hand_pos"]
        pos_mug = o_d["obj_pos"] + np.array([0.01, 0.0, 0.05])
        pos_goal = o_d["goal_pos"]

        if np.linalg.norm(pos_curr[:2] - pos_mug[:2]) > 0.06:
            return pos_mug + np.array([0.0, 0.0, 0.2])
        elif abs(pos_curr[2] - pos_mug[2]) > 0.02:
            return pos_mug
        else:
            return np.array([pos_goal[0], pos_goal[1], 0.1])

    @staticmethod
    def _push_grab_effort(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> float:
        pos_curr = o_d["hand_pos"]
        pos_mug = o_d["obj_pos"] + np.array([0.01, 0.0, 0.05])
        if (
            np.linalg.norm(pos_curr[:2] - pos_mug[:2]) > 0.06
            or abs(pos_curr[2] - pos_mug[2]) > 0.1
        ):
            return -1.0
        return 0.5

    # ------------------------------------------------------------------
    # Phase 2 helpers — press button (from SawyerCoffeeButtonV3Policy)
    # ------------------------------------------------------------------
    @staticmethod
    def _button_desired_pos(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> npt.NDArray[Any]:
        pos_curr = o_d["hand_pos"]
        pos_button = o_d["obj_pos"] + np.array([0.0, 0.0, -0.07])

        if np.linalg.norm(pos_curr[[0, 2]] - pos_button[[0, 2]]) > 0.02:
            return np.array([pos_button[0], pos_curr[1], pos_button[2]])
        else:
            return pos_button + np.array([0.0, 0.2, 0.0])

    # ------------------------------------------------------------------
    # Phase 3 helpers — pull mug (from SawyerCoffeePullV3Policy)
    # ------------------------------------------------------------------
    @staticmethod
    def _pull_desired_pos(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> npt.NDArray[Any]:
        pos_curr = o_d["hand_pos"]
        pos_mug = o_d["obj_pos"] + np.array([-0.005, 0.0, 0.05])

        if np.linalg.norm(pos_curr[:2] - pos_mug[:2]) > 0.06:
            return pos_mug + np.array([0.0, 0.0, 0.15])
        elif abs(pos_curr[2] - pos_mug[2]) > 0.02:
            return pos_mug
        else:
            return o_d["goal_pos"]

    @staticmethod
    def _pull_grab_effort(
        o_d: dict[str, npt.NDArray[np.float64]],
    ) -> float:
        pos_curr = o_d["hand_pos"]
        pos_mug = o_d["obj_pos"] + np.array([0.01, 0.0, 0.05])
        if (
            np.linalg.norm(pos_curr[:2] - pos_mug[:2]) > 0.06
            or abs(pos_curr[2] - pos_mug[2]) > 0.1
        ):
            return -1.0
        return 0.7
