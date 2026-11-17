# Task 2 — 32×32 partially observable simulator: observation, action and dynamics specification

Enabled with `env.kind: advanced` (presets: `configs/sim32_*.yaml`). The Task-1 10×10 env is unchanged
(`env.kind: basic`, the default). Both expose the same protocol: `reset(seed)`, `step(a) -> (obs, reward, done, info)`,
`obs_dim`, `n_actions`, `position`, `battery`, `steps`, `map_id`, `max_battery`.

## 1. Information boundary

| Hidden — evaluator only | Onboard — policy, monitor, fallback planner may use |
|---|---|
| true grid, start/goal terrain, optimal cost | position, goal offset, battery, step count |
| TRUE slip table, TRUE energy table, TRUE sensor parameters | noisy 5×5 readings (§3), visit counts |
| the three RNG streams | slip estimate, sensor confidence, `env.explored_map()` (§4) |

Ground truth is reachable only through `evaluation.evaluator.AdvancedEvaluator` (same module as the Task-1 evaluator, so
the existing "agents never import the evaluator" guard covers it). `tests/test_sim32.py::TestLeakage` checks:
observations are invariant to terrain outside the 5×5 window; random-number consumption does not depend on terrain;
`info` keys are whitelisted; no public attribute exposes the grid, slip or energy tables; policy modules do not import
the evaluator or the generator.

## 2. Action space — `Discrete(4)`

`0` row−1 (up) · `1` row+1 (down) · `2` col−1 (left) · `3` col+1 (right). Start `(0,0)`, goal `(31,31)`.

## 3. Observation — 59-d float vector (`env.feature_names`)

| Index | Name | Definition |
|---|---|---|
| 0–1 | `x_norm`, `y_norm` | row, col / 31 |
| 2–3 | `goal_dx`, `goal_dy` | (goal − position) / 31, signed |
| 4 | `battery_ratio` | battery / capacity |
| 5 | `steps_ratio` | steps / capacity |
| 6 | `slip_estimate` | (slips + 1) / (moves + 10). Odometry-based running estimate from the rover's own history (prior mean 0.1); a single global number, not terrain-specific |
| 7 | `sensor_confidence` | EMA (α = 0.3) of the fraction of re-observed cells whose new reading agrees with explored-map memory (needs ≥ 3 overlapping cells; starts at 1.0). A heuristic self-consistency score, **not** a calibrated probability |
| 8 | `coverage` | fraction of cells that have ever been inside the sensor window |
| 9–33 | `win_r{−2..+2}_c{−2..+2}` | noisy 5×5 reading, row-major, centred on the rover (incl. its own cell) |
| 34–58 | `visits_r{−2..+2}_c{−2..+2}` | times the rover stood on each window cell, clipped at 3, /3 (own trajectory memory; helps avoid loops) |

### Sensor model (per window cell, every step)

* Outside the map: reading `1.0` (map size is known), no noise, not stored in memory.
* True obstacle: detected (`1.0`) with probability `1 − sensor_miss`; if missed, it looks like noisy normal terrain.
* Free cell: reported as obstacle (`1.0`) with probability `sensor_false`; otherwise
  `clip(v(terrain) + N(0, sensor_sigma²), 0, 0.9)` with v = 0 normal, 0.3 sand, 0.6 rock.
* A reading ≥ 0.999 therefore means "obstacle detected". Readings are independent across cells and steps.
* Exactly 2·25 random numbers are drawn per observation whatever the terrain is.

The rover does **not** know the true `sensor_miss/false/sigma`; the explored map stores raw counts, not Bayesian beliefs.

## 4. Explored map — `env.explored_map()` → `ExploredMap` (copies, onboard only)

`seen`, `obs_reads`, `terrain_sum`, `visits`, `bumped` (all 32×32) plus derived `known`, `obstacle_frac` (nan where unseen,
1.0 where `bumped`), `terrain_mean`. A collision proves an obstacle (touch is noise-free) and sets `bumped`, even if the
sensor missed it. This is what the Task-3 battery-aware A* and the Task-4 monitor must plan on: unknown cells carry no
information and must be given an explicit risk cost by the planner.

## 5. Dynamics

* **Energy** (paid on entering a cell): `energy_cost = (normal, sand, rock)`, default `(1, 3, 5)`; obstacles impassable.
  Illustrative simulation units, not Joules. A move costing more than the remaining battery **stalls** (battery → 0,
  `battery_depleted`).
* **Slip**: with probability `slip_prob[terrain of the cell the rover stands on]` (default `(0.02, 0.10, 0.05)`) the
  commanded move is replaced by a uniformly chosen **perpendicular** move. The executed move then follows the normal rules,
  so a slip next to an obstacle causes a collision the policy did not command. `info["slipped"]` is reported (odometry).
* **Collision** (obstacle or map edge, identical): no motion, 1 battery, reward −2.
* Battery is never negative.

## 6. Termination and reward

`reason ∈ {goal, battery_depleted, timeout}`; success ⇔ `reason == "goal"` (arriving with exactly 0 battery is a success).
`timeout` only occurs if `max_steps < battery` (default: `max_steps = battery`, which every action's ≥ 1 cost already bounds);
it becomes relevant when a zero-cost action such as *stop* is added in Task 4.

| Event | Reward |
|---|---|
| move | `0.5·Δ(Manhattan to goal) − 0.05 − 0.025·(energy_used − 1)` |
| collision | −2 (−10 if it empties the battery) |
| stall / battery reaches 0 | −20 |
| goal | `100 + 50·battery/capacity` |

Unlike Task 1 there is **no full-map quantity in the reward**: the goal bonus uses the battery fraction, and the terrain
term uses the energy the rover actually felt.

## 7. Maps and randomness

* Terrain: two independent noise fields, smoothed by `gen_corr` passes of a 3×3 box blur and rank-transformed, give
  clustered patches with (almost) exact composition: obstacles = top `1 − p_normal − p_sand − p_rock` of one field; the
  rest split normal:sand:rock = `p_normal : p_sand : p_rock` by the other. 3×3 corner blocks around start and goal are normal.
* A map is accepted if the goal is reachable, the **reference** (nominal `ref_energy_cost`) optimal cost is
  ≤ `max_opt_cost_frac · battery` (0.6) and the true optimal cost ≤ battery. Using the nominal table for acceptance keeps maps
  identical across the sensor and dynamics shifts (paired comparisons).
* One episode seed spawns three independent streams: **map**, **slip**, **sensor**. `reset(seed=k)` is exactly episode k.
  Validation maps are seeds 10 000+, test maps 20 000+ (as in Task 1).

## 8. Presets (battery 250 and size 32 everywhere)

| Preset | Terrain (normal/sand/rock/obstacle, `gen_corr`) | Sensor (miss / false / σ) | Dynamics (energy, slip by terrain) |
|---|---|---|---|
| `id` | .60/.15/.10/.15, 2 | .05 / .02 / .05 | (1,3,5), (.02,.10,.05) |
| `terrain_shift` | .35/.25/.22/.18, 3 | = id | = id |
| `sensor_shift` | = id | .20 / .08 / .15 | = id |
| `dynamics_shift` | = id | = id | (1,4,6), (.06,.25,.15) |
| `combined_shift` | terrain shift | sensor shift | dynamics shift |

These levels are a first definition, to be frozen with the Phase-2 protocol (Task 6). Mean optimal cost over test maps:
≈ 66 (id), ≈ 92 (terrain), ≈ 67 (dynamics), ≈ 104 (combined), against a battery of 250.
