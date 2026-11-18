# planetary-rover-marec — Phase 1 / Task 1 baseline

Audited, reproducible starting point for *Uncertainty-Gated Runtime Assurance for Learning-Based Planetary Rover Navigation* (MAREC).
Read **`docs/AUDIT_NOTES.md`** first, then **`docs/OBSERVATION_SCHEMA.md`**.

```
configs/      YAML: three reference configs (onboard, onboard+shaping, legacy17)
src/
  environment/  rover_env.py (policy-facing), terrain.py (pure helpers)
  evaluation/   evaluator.py (ONLY place that reads the hidden map), runner.py (metrics, Wilson CIs)
  agents/       dqn_agent.py, ppo_agent.py, replay_buffer.py   (see only observations)
  policies/     torch-free reference policies (random, onboard-greedy)
  training/     agent-agnostic training loops
  utils/        JSONL step logger, run metadata (git hash, versions, config)
scripts/      audit_legacy.py, run_reference.py, train_dqn.py, train_ppo.py
tests/        27 tests: accounting, seeding, leakage, legacy parity, loops, docs sync
legacy/       original project, byte-for-byte (also at git tag v0.0-original)
```

## Quick start
```
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt   # or: make setup
make all-checks    # tests + legacy audit + torch-free reference run   (~30 s, no torch needed)
make smoke         # first thing to run on your machine: exercises the torch agents (~1 min)
make repro-check   # same seed twice => identical logs
make baselines     # full DQN/PPO runs, 1500 episodes, both reward variants
```
Recover the untouched original at any time: `git checkout v0.0-original`.

## Rules for later tasks
* Anything an agent, monitor or fallback planner uses must come from `obs`/`info` of `RoverEnv`. `tests/test_no_leakage.py` fails the build otherwise.
* Never pick thresholds/hyper-parameters on test seeds (20 000+); use validation seeds (10 000+).
* Success is `info["reason"] == "goal"`. Never infer it from reward.
