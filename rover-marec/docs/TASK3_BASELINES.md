# Task 3 — navigation baselines and the shared evaluation protocol

## Methods (all onboard-only; all go through `src/evaluation/harness.py`)

| Key | What it is | Trained? |
|---|---|---|
| `astar` | Battery-aware A* on the **explored map** (`env.explored_map()`), replanning every step. Edge cost = expected energy of the entered cell from the *nominal* table; **unknown cells** cost `unknown_energy` + a risk premium that grows as the battery drains; believed obstacles are soft walls; collision-proven obstacles are hard walls; small penalty for revisits. | no |
| `reactive` | Window-only heuristic (sanity reference, not a baseline in the paper). | no |
| `dqn` | Double DQN (same algorithm/hyper-parameters as the audited original) on the 59-d observation, greedy at test time. | 10 seeds |
| `dqn_shield` | The same DQN checkpoints wrapped by `RuleShield`: rejects invalid moves, moves into believed obstacles, and moves that make the goal unreachable on the remaining battery; overrides with the highest-Q safe action. No uncertainty / OOD signal (that is Task 4). | no (reuses `dqn`) |
| `ppo` | Actor-critic PPO with the Task-1 fix (`reward_scale 0.02`, `entropy_coef 0.003`, carried over unchanged; config `configs/sim32_id_ppo.yaml`), greedy at test time. Secondary baseline. | 10 seeds |
| `ppo_shield` | The same PPO checkpoints behind `RuleShield` (policy logits rank the safe actions). | no (reuses `ppo`) |

## Protocol (what makes the comparison fair)

* Training: in-distribution preset only (`configs/sim32_id.yaml`), 1 500 episodes, seeds 0-9. Training maps come from the
  env's own seed stream. Two checkpoints are kept: `final.pt` and `best_val.pt` (best greedy success on the 50 validation maps
  evaluated every 100 episodes; selection uses validation maps only).
* Evaluation maps: validation = episode seeds 10 000+, test = 20 000+. Neither is ever trained on. Use `val` to look and choose;
  run `test` once per configuration. Default checkpoint for evaluation: `best_val`.
* One episode = (preset, seed). The map, start/goal, battery, slip draws and sensor noise are a function of that pair only, and
  the slip/sensor streams use a fixed number of draws per step. So every method sees the same maps and the same random numbers
  (common random numbers), which is what makes paired comparisons valid later.
* Success = terminal reason `goal`. Shielded and unshielded runs are judged by the same code path.
* Reporting: seeded methods = mean ± sd over seeds with a t-based 95% CI over seeds (CI clipped to [0,1]); unseeded methods =
  Wilson 95% CI over maps. Seeds are the unit of replication for learned methods, so do not pool episodes across seeds for a CI.

## Metrics (per episode, `results/task3/eval/<split>/<preset>/<method>[_sK].csv`)

`success, reason, steps, reward, battery_left, energy_used, collisions` (resulting events, incl. slip-caused),
`attempted_unsafe` (the COMMANDED move targets an obstacle / leaves the map, judged from ground truth by the evaluator),
`interventions / decisions / no_safe_action` (shield), `optimal_cost`, `energy_ratio` (energy / optimal cost, successes only),
`plan_ms` (decision latency). Attempted-unsafe and collisions are deliberately separate (plan, §5): a slip next to an obstacle is
a collision without an unsafe command.

## Run it

```bash
python3 scripts/run_task3.py train  --jobs 4                 # DQN seeds 0-9 (each ~25-30 min single-threaded; --jobs parallelises)
python3 scripts/run_task3.py eval   --split val  --jobs 4    # look here first
python3 scripts/run_task3.py report --split val
python3 scripts/run_task3.py curves                           # training stability across seeds
python3 scripts/run_task3.py eval   --split test --jobs 4    # once the setup is settled
python3 scripts/run_task3.py report --split test
# shifts are Task 7, but the harness already supports them:  --presets id,terrain_shift,sensor_shift,dynamics_shift,combined_shift
```
`results/task3/manifest.json` records seeds, config, git commit and software versions of the training run.

## Stability gate before adding PPO

`run_task3.py curves` prints greedy validation success at every checkpoint episode across seeds and applies the gate:
**last-checkpoint mean >= 0.80 and no seed below 0.60**, and A* >= 0.90 on validation maps. Only then is PPO added
(`ppo`, `ppo_shield`) with the reward-scale / entropy settings from the Task-1 PPO fix. The DQN gate passed on the 10-seed run
(last-checkpoint mean 0.91, min 0.88). PPO is trained and evaluated exactly like DQN: `train --algo ppo`, `curves --algo ppo`, then `eval`/`report`.

Known behaviour from a 2-seed, 700-episode probe (not the final protocol): greedy validation success dips sharply around
episodes 300-500 (0.0-0.1 at episode 400) as epsilon reaches its floor, then recovers (0.88-0.92 at episode 700). This is why
`best_val.pt` is kept and why 1 500 episodes are used; the 10-seed curves will show whether it persists.
