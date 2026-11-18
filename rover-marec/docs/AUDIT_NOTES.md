# Task 1 — Codebase audit notes

Scope: the uploaded `Battery-Aware-Mars-Rover-DQN-vs-PPO` project (10×10 grid, DQN + PPO).
Original is preserved untouched at git tag **`v0.0-original`** (branch `legacy-original`) and, byte-for-byte, in `legacy/`.
Every number below is produced by `scripts/audit_legacy.py` (raw output: `results/audit/legacy_audit.json`) or by the test suite.

## 1. Current state of the original code

| Aspect | Original behaviour |
|---|---|
| Map | 10×10, i.i.d. per cell: normal 55 %, sand 15 %, rock 15 %, obstacle 15 %; start (0,0), goal (9,9); obstacles cleared in both 2×2 corner blocks; up to 200 regeneration attempts until BFS-connected |
| Observation | 17-d: `x, y, goal_dx, goal_dy, battery_ratio, cost_to_goal_ratio, budget_margin, steps_ratio` + 3×3 terrain patch (obstacle/out-of-map 1.0, rock 0.6, sand 0.3, normal 0) |
| Actions | 4: row−1, row+1, col−1, col+1 |
| Energy | enter normal 1 / sand 3 / rock 5; capacity 100 |
| Reward | move: `0.5·ΔManhattan − 0.05` (− 0.05 sand, − 0.1 rock); budget-margin penalty (full-map Dijkstra); obstacle collision −2 (1 battery); goal `100 + 50·min(1, battery/optimal_cost)`; depleted −20 (−10 if by collision) |
| Termination | goal; battery ≤ 0; no step limit (every action costs ≥ 1 battery so ≤ 100 steps) |
| DQN | Double DQN, MLP 256-256-128, γ .99, lr 5e-4, ε 1→.05 (×.992/episode), buffer 50 k, batch 128, target sync 500, 2 000-step random warm-up; **exists only in the notebook** |
| PPO | actor-critic MLP 256-256 (tanh, orthogonal init), GAE λ .95, clip .2, 10 epochs, minibatch 64, entropy .05, rollout 512 |

## 2. Findings and what was done

Severity: **C** invalidates the research claim, **H** corrupts reported metrics, **M** correctness/maintainability, **L** minor.

| ID | Sev | Finding (evidence) | Resolution |
|---|---|---|---|
| F1 | **C** | **Hidden-map leakage.** Obs features 5 (`cost_to_goal_ratio`) and 6 (`budget_margin`) are full-map Dijkstra values. Measured: both change when a rock band is placed far outside the 3×3 patch; nothing else changes. | Removed from the policy-facing observation (15-d `onboard` mode). Full-map code lives only in `src/evaluation/evaluator.py`; hidden world is a private object. Guarded by 6 leakage tests: observation-invariance under outside-window terrain changes (150 maps), a power check proving `legacy17` *does* leak, info-key whitelist, public-API scan, AST scan of `src/agents`+`src/policies`. |
| F2 | **H** | **Negative battery.** Moves are applied before the affordability check. Saved results: min battery −4; DQN 92/1500 and PPO 250/1500 episodes end below 0. Fresh run with a local-greedy policy: 70/600. | Battery clamped at 0; a move costing more than the remaining battery makes the rover stall (battery→0, `battery_depleted`). Tests: never negative over 300 random episodes; stall test. |
| F3 | **H** | **Success inferred from `reward >= 100`.** Goal is checked before battery, so a goal arrival with negative battery yields `100 + 50·(negative)` < 100 and is *not* counted as success; arrival at exactly 0 battery is counted. Latent: 0/352 goal arrivals affected in my 600-episode sample; 12 saved "successes" (DQN 1, PPO 11) ended on exactly 0 battery. | Explicit `info["reason"]` ∈ {goal, battery_depleted, timeout}; success ≡ `reason == "goal"`; arriving on exactly 0 is a success by rule (documented in `OBSERVATION_SCHEMA.md`). |
| F4 | **M** | **Boundary bump ≠ obstacle collision.** Bumping the map edge costs the *current cell's* energy (5 when standing on rock) with reward −0.15; hitting an obstacle costs 1 with reward −2. | Both are collisions: −2 reward, 1 battery, no motion. |
| F5 | **H** | **DQN `steps` is a placeholder** (`[env.max_battery] * n` → constant 100). `compare_steps.png` / `DQN_vs_PPO_Overlapping.png` therefore compare real PPO steps to a constant. | Steps always come from the env (`info["steps"]`), logged per episode. Old DQN steps curve should not be used. |
| F6 | **M** | **`legacy/train_ppo.py` crashes at the first update:** `agent.act()` returns `value.item()` (a float), then the script calls `next_val.item()` → `AttributeError`. The notebook contains a corrected loop; the script was not updated. | `PPOAgent.sample()/value()` return plain floats; bootstrap uses `value()`; loop tested with a stand-in agent. |
| F7 | **M** | **Notebook cell order (static finding, notebook not executed).** Cell 1's `rover_env.py` calls `MarsGridGenerator(...)` which is defined in cell 12 and not imported by the written module → would raise `NameError`. The generator output is overwritten by `reset()` anyway (dead code). The executed outputs imply a different version was run. | Dead code dropped; env is now a plain module. |
| F8 | **M** | `train_dqn.py` is listed in the README but absent; DQN training exists only inside the notebook. | `scripts/train_dqn.py`. |
| F9 | **H** | **No seeding.** No seed argument, no per-episode seeds, maps come from global `random`; two un-seeded envs differ. Train/validation/test maps cannot be separated. | `RoverEnv(cfg, seed)`: reproducible map stream; `reset(seed=k)` regenerates exact map *k*. Validation maps = seeds 10 000+, test maps = 20 000+. `seed_everything()` for python/numpy/torch. Tests: same seed → same map/trajectory hash, different seed → different. |
| F10 | **H** | **Reported success rate is training-time** (ε ≥ 0.05, stochastic PPO) on freshly sampled maps; there is no held-out or greedy evaluation. | Training scripts log both training-time SR (comparable with the original) and greedy evaluation on fixed validation maps every `eval_every` episodes, plus 200 validation + 200 test maps at the end with Wilson 95 % CIs. |
| F11 | **H** | **Committed artifacts disagree with the notebook logs and with the README.** Notebook log vs `*_results.pkl` (last-100 SR): DQN 0.87 vs 0.88, PPO **0.81 vs 0.91** — the pickles come from different runs than the logged ones. README says PPO has "faster convergence"; in the committed data rolling-100 SR first reaches 0.8 at episode **160 for DQN vs 1 108 for PPO** (PPO first-100 SR 0.00 vs DQN 0.36). | Treat all legacy plots/pickles as unverified single-seed runs. The claim should not be reused until re-measured over multiple seeds (Phase 2). |
| F12 | **M** | **Cost.** Legacy env runs ≈1.9 full-map Dijkstra per step (347 µs/step in this sandbox); `random.sample` on a 50 k `deque` is O(n) per index. | Onboard env: no per-step Dijkstra (7.5 µs/step, ≈46× faster, same sandbox, random actions); numpy ring-buffer replay; `torch_threads: 1`. |
| F13 | L | Map validity: BFS hard-codes start `(0,0)`, ignores battery feasibility, silently accepts an unsolvable map after 200 tries; `optimal_cost` falls back to `max_battery` for unreachable goals. **Measured on 2 000 legacy maps: 0 unsolvable, 0 unreachable, 0 cost > battery — no effect at 10×10.** | Now guarded (energy-feasibility check, hard error if no valid map) because 32×32 will behave differently. |
| F14 | design | **Reward-side privileged information.** The budget-margin penalty (per step) and the goal bonus (`optimal_cost`) use the full map. They are not observed, but they shape learning with information the rover would not have. The goal bonus also saturates: median optimal cost is 24/100 battery, so `min(1, left/optimal)` is already 1 for any route using ≤ 76 units. | Per-step shaping is a switch (`reward_privileged_shaping`, **off** by default); goal bonus kept (terminal, evaluator-computed). **Your decision — see §5.** |

Checked and found correct: Double-DQN target, PPO GAE/clipping logic, action indexing, 3×3 patch ordering, terrain probabilities (all verified by parity/distribution tests).

## 3. Parity with the original

`tests/test_legacy_parity.py` runs the **original** `legacy/rover_env.py` and the new env (`obs_mode: legacy17`, shaping on) on identical injected maps with random actions and asserts identical observations, rewards, `done` and battery on every transition (> 3 000 compared; > 100 intentionally divergent events — boundary bumps and unaffordable moves — are stepped only in the new env, after which the original is resynchronised). `onboard` obs equals the legacy obs with indices 5, 6 deleted.

## 4. Reproduction status (honest)

| Item | Status |
|---|---|
| Original env, fixed seeds, accounting defects measured | **Done** (`make audit`) |
| New env: 27 unit/scenario/leakage/parity/loop tests | **Done, all pass** (`make test`) |
| Torch-free 10×10 reference run (random + onboard-greedy, 200 val + 200 test maps, run twice, identical trajectory hash) | **Done** (`make reference`); `results/reference_10x10/` |
| DQN/PPO agents (`src/agents/*`) — training reproduction of the original learning curves | **Not run: PyTorch is not installed in the authoring sandbox and there is no network.** Code is syntax-checked and its training loops are tested with stand-in agents, but the networks themselves have not been executed. Run `make smoke` first. |

Acceptance criteria for the torch runs (run on your machine): `make smoke` completes; `make repro-check` shows identical `episodes.csv` for two runs with the same seed; `make baselines` greedy validation success is in the same range as the original 0.8–0.9 training-time figures (if it is clearly lower with `onboard` but not with `onboard_shaped`, that is evidence the privileged shaping was doing the work → decide per §5).

## 5. Decisions that need you

1. **`reward_privileged_shaping`** default `false` (strictly onboard information end-to-end). If baselines learn clearly worse than the original, switch to `configs/baseline_10x10_onboard_shaped.yaml` and document it as "training-time privileged reward shaping" in the paper.
2. **Goal bonus** still uses evaluator-computed `optimal_cost`. Acceptable as a terminal-only term; replace by a battery-fraction bonus if you prefer zero full-map dependence.
3. **Boundary bump = collision** is a deliberate behaviour change versus the original (F4).
