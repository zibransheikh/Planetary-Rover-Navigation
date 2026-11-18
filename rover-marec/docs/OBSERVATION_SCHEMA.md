# Observation, action, reward and termination schema (audited 10×10 environment)

Source of truth: `src/environment/rover_env.py`. `tests/test_docs_sync.py` fails if this table drifts from the code.

## Policy-facing observation (`obs_mode: onboard`, 15-d, float64)

| idx | feature | definition | available to the rover? |
|---|---|---|---|
| 0 | `x_norm` | row / (size-1) | yes (localisation) |
| 1 | `y_norm` | col / (size-1) | yes |
| 2 | `goal_dx` | (goal_row − row) / (size-1) | yes (relative goal direction) |
| 3 | `goal_dy` | (goal_col − col) / (size-1) | yes |
| 4 | `battery_ratio` | battery / capacity | yes |
| 5 | `steps_ratio` | steps taken / capacity | yes |
| 6–14 | `patch_mm … patch_pp` | 3×3 terrain window around the rover, row-major (rows −1,0,+1 × cols −1,0,+1); normal 0.0, sand 0.3, rock 0.6, obstacle **or outside map** 1.0 | yes (local sensing, noise-free in Task 1) |

Names in code: `patch_mm, patch_mz, patch_mp, patch_zm, patch_zz, patch_zp, patch_pm, patch_pz, patch_pp`
(m = −1, z = 0, p = +1; first letter = row offset, second = column offset).

The 3×3 patch is the **only** place hidden terrain enters the observation. Nothing else derived from the full map is observable.

## Removed privileged features (present only in `obs_mode: legacy17`, reproduction/ablation)

| original idx | feature | why removed |
|---|---|---|
| 5 | `cost_to_goal_ratio[PRIVILEGED]` | Dijkstra over the *entire hidden map* |
| 6 | `budget_margin[PRIVILEGED]` | uses the same full-map Dijkstra |

## Actions (unchanged)
`0: row−1`, `1: row+1`, `2: col−1`, `3: col+1`.

## Energy
Entering a cell costs normal 1, sand 3, rock 5. Obstacles are impassable.

## Reward (per step)
* Move: `0.5·(Δ Manhattan distance to goal) − 0.05`, additionally `−0.05` on sand, `−0.1` on rock.
* Optional (`reward_privileged_shaping: true`, off by default): full-map budget-margin penalty — **reward only, never observed**.
* Collision (obstacle **or map boundary**): `−2`, rover does not move, costs 1 battery.
* Goal: `100 + 50·min(1, battery_left / optimal_cost)` (`optimal_cost` is evaluator-computed; terminal reward only).
* Battery depleted: `−20` (`−10` if depleted by a collision).

## Termination (explicit `info["reason"]`)
| reason | rule | `terminated` | `truncated` |
|---|---|---|---|
| `goal` | rover enters the goal cell with battery ≥ 0 after paying the cost (arriving on exactly 0 counts) | yes | no |
| `battery_depleted` | battery reaches 0 away from the goal, or the next move costs more than the remaining battery (rover stalls in place, battery set to 0) | yes | no |
| `timeout` | `steps ≥ max_steps` (default = capacity; cannot trigger because every action costs ≥ 1) | no | yes |

**Success ≡ `reason == "goal"`.** Never inferred from reward size. Battery is never negative. `steps` counts every action including collisions.

## `info` dict (policy-visible, non-privileged only)
`event` (`move | obstacle_collision | boundary_collision | stalled`), `reason`, `terminated`, `truncated`, `energy_used`, `steps`, `battery`.
