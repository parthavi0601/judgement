# Judgement Card Game — Build, Train, Test & Run Commands

## Prerequisites

- **Python 3.10+** 
- **C++ compiler**: MSVC (Windows) or GCC 9+ (Linux)
- **pip packages**: `pybind11`, `torch`, `rlcard`, `numpy`, `pytest`

---

## 1. Build C++ Module

```bash
# Install pybind11 first (if not already)
pip install pybind11

# Build and install in development mode (editable)
pip install -e .

# Verify C++ module loads
python -c "import judgement_cpp; print('C++ backend OK')"
```

If the C++ module fails to build, the system automatically falls back to the pure Python engine. Training/evaluation still works — just slower.

---

## 2. Train NFSP Agents

```bash
# Quick test (500 episodes, ~2 min)
python main.py --nfsp-episodes 500 --evaluate-every 100

# Standard training (200K episodes, ~2-4 hours with C++)
python main.py --nfsp-episodes 200000 --evaluate-every 2000 --checkpoint-every 10000 --save-dir ./checkpoints

# Full-scale training (2M episodes, ~20-40 hours with C++)
python main.py --nfsp-episodes 2000000 --evaluate-every 10000 --checkpoint-every 50000 --save-dir ./checkpoints

# Custom learning rates
python main.py --nfsp-episodes 200000 --rl-learning-rate 0.0005 --sl-learning-rate 0.002
```

---

## 3. Resume Training from Checkpoint

```bash
# Resume from episode 200000 for 200K more episodes
python main.py --load-checkpoint 200000 --resume-training --nfsp-episodes 400000 --save-dir ./checkpoints

# Resume with reduced learning rates (for fine-tuning)
python main.py --load-checkpoint 200000 --resume-training --nfsp-episodes 400000 \
  --rl-learning-rate 0.0001 --sl-learning-rate 0.001
```

---

## 4. Evaluate Agents

```bash
# Evaluate with Hybrid MC-NFSP (MCTS + trained NFSP)
python main.py --load-checkpoint 200000 --hybrid-games 100 --mcts-depth 2 --mcts-simulations 200

# High-quality evaluation (more MCTS simulations)
python main.py --load-checkpoint 200000 --hybrid-games 50 --mcts-depth 3 --mcts-simulations 500

# Compare Hybrid vs Pure NFSP (fair arena across all seats)
python main.py --load-checkpoint 200000 --hybrid-games 50 --hybrid-vs-pure-games 50 --eval-games 200
```

---

## 5. Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_game.py -v
python -m pytest tests/test_mcts.py -v
python -m pytest tests/test_hybrid.py -v

# Run with coverage
python -m pytest tests/ -v --tb=short
```

---

## 6. Full Pipeline (Train → Evaluate → Compare)

```bash
# Complete pipeline: train 200K, then evaluate
python main.py \
  --nfsp-episodes 200000 \
  --evaluate-every 2000 \
  --checkpoint-every 10000 \
  --hybrid-games 50 \
  --mcts-depth 2 \
  --mcts-simulations 200 \
  --eval-games 100 \
  --hybrid-vs-pure-games 50 \
  --save-dir ./checkpoints
```

---

## Key Arguments Reference

| Argument | Default | Description |
|---|---|---|
| `--nfsp-episodes` | 500 | Total training episodes |
| `--evaluate-every` | auto | Print metrics every N episodes |
| `--checkpoint-every` | auto | Save model every N episodes |
| `--save-dir` | `./checkpoints` | Checkpoint directory |
| `--load-checkpoint` | — | Load from episode N |
| `--resume-training` | false | Resume training from checkpoint |
| `--hybrid-games` | 10 | Hybrid MC-NFSP eval games |
| `--mcts-depth` | 2 | MCTS depth (agent's own moves) |
| `--mcts-simulations` | 200 | MCTS sims per decision |
| `--eval-games` | 20 | Pure NFSP comparison games |
| `--hybrid-vs-pure-games` | 0 | Arena: 1 Hybrid vs 3 Pure |
| `--rl-learning-rate` | 0.001 | Q-network learning rate |
| `--sl-learning-rate` | 0.005 | Average Policy learning rate |
| `--seed` | 42 | Random seed |
