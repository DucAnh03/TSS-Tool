# ─────────────────────────────────────────────────────────────────────────────
# MLOps Demo – Makefile
# Python 3.11 required.  Install deps first: make setup
# ─────────────────────────────────────────────────────────────────────────────

PYTHON     ?= python3.11
PIP        ?= $(PYTHON) -m pip
RUN_ID     := $(shell $(PYTHON) scripts/new_run_id.py 2>/dev/null || echo "local_$(shell date +%Y%m%d_%H%M%S)")

.PHONY: all setup voice_quick vision_quick report clean

all: setup voice_quick vision_quick report

# ── Environment ───────────────────────────────────────────────────────────────
setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# ── Voice pipeline ────────────────────────────────────────────────────────────
# Runs a quick evaluation (no full audio corpus needed for a demo).
voice_quick:
	@echo ">>> [voice] run_id=$(RUN_ID)"
	$(PYTHON) voice/eval.py \
		--wakeword hey_jarvis \
		--threshold 0.5 \
		--mode quick \
		--run-id "$(RUN_ID)"

# ── Vision pipeline ───────────────────────────────────────────────────────────
# Trains a SimpleCNN on a 2 000-sample CIFAR-10 subset (≈ 2-3 min on CPU).
vision_quick:
	@echo ">>> [vision] run_id=$(RUN_ID)"
	$(PYTHON) vision/train.py \
		--epochs 3 \
		--batch-size 64 \
		--quick \
		--run-id "$(RUN_ID)"

# ── CML / Markdown report ─────────────────────────────────────────────────────
# Generates report.md from artifacts/latest/metrics.json.
# In CI, CML picks this file up and posts it as a PR comment.
report:
	$(PYTHON) scripts/gen_report.py

# ── Housekeeping ──────────────────────────────────────────────────────────────
clean:
	rm -rf artifacts/runs/* .cifar10_cache __pycache__ */__pycache__
	find . -name "*.pyc" -delete
