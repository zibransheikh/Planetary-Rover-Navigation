PY ?= python3
.PHONY: setup test audit reference smoke repro-check baselines legacy17 all-checks

setup:            ## one-time: virtualenv + deps
	$(PY) -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt

test:             ## 27 tests, ~10 s, torch not required
	$(PY) -m unittest discover -s tests -t .

audit:            ## measure defects of the ORIGINAL code (torch not required)
	$(PY) scripts/audit_legacy.py

reference:        ## torch-free reproducible 10x10 reference run
	$(PY) scripts/run_reference.py

all-checks: test audit reference

smoke:            ## verify torch install + agents end-to-end (30 episodes each)
	$(PY) scripts/train_dqn.py --episodes 30 --eval-every 10 --eval-episodes 5 --out results/smoke/dqn
	$(PY) scripts/train_ppo.py --episodes 30 --eval-every 10 --eval-episodes 5 --out results/smoke/ppo

repro-check:      ## same seed twice must give byte-identical episode logs
	$(PY) scripts/train_dqn.py --episodes 30 --eval-every 10 --eval-episodes 5 --seed 1 --out results/smoke/a
	$(PY) scripts/train_dqn.py --episodes 30 --eval-every 10 --eval-episodes 5 --seed 1 --out results/smoke/b
	diff results/smoke/a/episodes.csv results/smoke/b/episodes.csv && echo "REPRODUCIBLE"

baselines:        ## full 1500-episode runs, onboard obs, both reward variants
	$(PY) scripts/train_dqn.py --config configs/baseline_10x10_onboard.yaml
	$(PY) scripts/train_ppo.py --config configs/baseline_10x10_onboard.yaml
	$(PY) scripts/train_dqn.py --config configs/baseline_10x10_onboard_shaped.yaml
	$(PY) scripts/train_ppo.py --config configs/baseline_10x10_onboard_shaped.yaml

legacy17:         ## original 17-d observation (reproduction / ablation only)
	$(PY) scripts/train_dqn.py --config configs/legacy17_reproduction.yaml
	$(PY) scripts/train_ppo.py --config configs/legacy17_reproduction.yaml
