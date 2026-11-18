"""Conventional rule-based safety shield (Task 3 baseline). Onboard information only.

Wraps any base policy. For the proposed action it checks three explicit, testable rules; if one fails it overrides with
the best remaining action (highest base Q-value if the base exposes `q_values`, otherwise closest to the goal):

  invalid_action   the action would leave the map
  obstacle         the target cell is believed blocked: detected in the current 5x5 window, believed so in the explored
                   map (majority of readings), or proven by an earlier collision
  battery          even the best case cannot work out: entry energy of the target (nominal table, from its reading)
                   exceeds the battery, or battery after the move < Manhattan distance to goal (>= 1 energy per step)

Deliberately NO uncertainty or OOD signal: that is what the Task-4 runtime-assurance system adds. Per-decision
counters (`stats`) give the intervention rate and the rules that fired. If every action violates a rule, the base action
is kept and counted in `no_safe_action`.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_WIN0, _W, _R = 9, 5, 2   # window block of the 59-d observation (see docs/OBSERVATION_SCHEMA_V2.md)
RULES = ("invalid_action", "obstacle", "battery")


class RuleShield:
    name = "rule_shield"

    def __init__(self, base, nominal_energy=(1.0, 3.0, 5.0)):
        self.base, self.nominal = base, tuple(float(v) for v in nominal_energy)
        self._get_map, self._capacity = None, None
        self.stats = {"decisions": 0, "interventions": 0, "no_safe_action": 0, "by_rule": Counter()}
        self.last_decision: dict = {}

    def attach(self, get_map, battery_capacity: float) -> None:
        self._get_map, self._capacity = get_map, float(battery_capacity)

    def _entry_energy(self, reading: float) -> float:
        return self.nominal[0] if reading < 0.15 else self.nominal[1] if reading < 0.45 else self.nominal[2]

    def violations(self, obs, action: int, m) -> list[str]:
        n = m.seen.shape[0]
        x, y = int(round(obs[0] * (n - 1))), int(round(obs[1] * (n - 1)))
        gx, gy = x + int(round(obs[2] * (n - 1))), y + int(round(obs[3] * (n - 1)))
        dr, dc = _DELTAS[action]
        tx, ty = x + dr, y + dc
        if not (0 <= tx < n and 0 <= ty < n):
            return ["invalid_action"]
        out = []
        reading = float(obs[_WIN0 + (_R + dr) * _W + (_R + dc)])
        belief = m.obstacle_frac[tx, ty]
        if reading >= 0.999 or m.bumped[tx, ty] or (not np.isnan(belief) and belief > 0.5):
            out.append("obstacle")
        battery = float(obs[4]) * self._capacity
        entry = self._entry_energy(reading) if reading < 0.999 else 1.0
        if entry > battery or battery - entry < abs(tx - gx) + abs(ty - gy):
            out.append("battery")
        return out

    def act(self, obs) -> int:
        if self._get_map is None:
            raise RuntimeError("call attach(env.explored_map, env.max_battery) before act()")
        m = self._get_map()
        proposed = int(self.base.act(obs))
        self.stats["decisions"] += 1
        bad = self.violations(obs, proposed, m)
        self.last_decision = {"proposed": proposed, "executed": proposed, "violations": bad, "intervened": False}
        if not bad:
            return proposed
        self.stats["interventions"] += 1
        self.stats["by_rule"][bad[0]] += 1
        ok = [a for a in range(4) if not self.violations(obs, a, m)]
        self.last_decision["intervened"] = True
        if not ok:
            self.stats["no_safe_action"] += 1
            self.last_decision["no_safe_action"] = True
            return proposed
        if hasattr(self.base, "q_values"):
            q = np.asarray(self.base.q_values(obs))
            choice = max(ok, key=lambda a: q[a])
        else:
            n = m.seen.shape[0]
            gdx, gdy = obs[2] * (n - 1), obs[3] * (n - 1)
            choice = max(ok, key=lambda a: _DELTAS[a][0] * gdx + _DELTAS[a][1] * gdy)
        self.last_decision["executed"] = int(choice)
        return int(choice)
