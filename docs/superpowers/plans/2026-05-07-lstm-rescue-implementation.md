# LSTM Rescue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the LSTM benchmark so its 168-step input windows represent true hourly history and its training uses train-only scaling.

**Architecture:** Keep the existing global LSTM model, but replace the sequence-preparation path with continuity-safe window generation plus train-only feature and target scaling. The saved outputs, split logic, and result summaries stay compatible with the rest of the project.

**Tech Stack:** Python, pandas, NumPy, TensorFlow, unittest

---

### Task 1: Repair sequence construction

**Files:**
- Modify: `src/training/sequence.py`
- Modify: `tests/test_sequence_utils.py`

- [ ] Add a failing test showing that sequence windows must not bridge timestamp gaps within a station.
- [ ] Add a failing test showing that contiguous windows still work normally inside a clean hourly run.
- [ ] Update sequence construction to emit windows only from uninterrupted hourly runs.
- [ ] Re-run `tests.test_sequence_utils`.

### Task 2: Add train-only scaling helpers

**Files:**
- Modify: `src/training/sequence.py`
- Modify: `tests/test_sequence_utils.py`

- [ ] Add a failing test for fitting feature-scaling statistics from training data only.
- [ ] Add a failing test for applying target scaling and inverse scaling consistently.
- [ ] Implement simple reusable standardization helpers.
- [ ] Re-run `tests.test_sequence_utils`.

### Task 3: Wire the repaired pipeline into LSTM training

**Files:**
- Modify: `src/train_lstm_48h.py`
- Modify: `tests/test_train_lstm_48h.py`
- Modify: `logs/project_work_log.md`

- [ ] Add a failing test for any new small helper behavior exposed from the LSTM trainer.
- [ ] Update the training script to build contiguity-safe sequences and apply train-only scaling before fitting.
- [ ] Preserve output compatibility for metrics, predictions, and metadata.
- [ ] Log the methodological repair in the project work log.
- [ ] Re-run focused LSTM and sequence tests.
