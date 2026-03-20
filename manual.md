# Judgement Bot - Training and Evaluation Manual

This document outlines the primary command-line workflows for training and evaluating the Judgement NFSP and Hybrid MCTS agents.

## 1. Train From Scratch

This command kicks off a fresh training run for a given number of episodes. Neural Network checkpoints and the `training_metrics.csv` log are saved to `--save-dir`.

```bash
uv run main.py \
    --nfsp-episodes 500000 \
    --save-dir ./checkpoints \
    --evaluate-every 500 \
    --checkpoint-every 10000 \
    --rl-learning-rate 0.01 \
    --sl-learning-rate 0.005 \
    --hybrid-games 10 \
    --eval-games 20 \
    --mcts-depth 2 \
    --mcts-simulations 50
```

**Options Breakdown:**
- `--nfsp-episodes`: Total number of training episodes. After this, it evaluates Phase 2 (Hybrid) and Phase 3 (Pure NFSP).
- `--save-dir`: The folder where model weights (`.pt` files) and the metrics CSV will be stored.
- `--evaluate-every`: Prints progress and logs the current RL/SL losses and average payoffs to the CSV every N episodes.
- `--checkpoint-every`: Dumps the `.pt` neural network weights to the hard drive every N episodes.
- `--rl-learning-rate` / `--sl-learning-rate`: Base learning rates for the Q-network (RL) and Average Policy (SL) optimizers. 
- `--hybrid-games` / `--eval-games`: How many games to evaluate at the very end.
- `--mcts-depth` / `--mcts-simulations`: Configuration for the MC-NFSP Hybrid evaluation phase.

---

## 2. Resume Training From Checkpoint

If training crashes or is manually stopped, you can securely resume it from any checkpoint step. Your target `--nfsp-episodes` must be higher than the loaded checkpoint number. It will automatically load the `.pt` models, continue the episode loop, and **append** to the existing `training_metrics.csv` instead of overwriting.

```bash
uv run main.py \
    --load-checkpoint 200000 \
    --resume-training \
    --nfsp-episodes 500000 \
    --save-dir ./checkpoints \
    --evaluate-every 500 \
    --checkpoint-every 10000 \
    --rl-learning-rate 0.001 \
    --sl-learning-rate 0.0005
```

**Options Breakdown:**
- `--load-checkpoint`: The exact episode number you wish to load (e.g., `200000` looks for `nfsp_agent_X_ep200000.pt`).
- `--resume-training`: **Critical flag.** Without this, passing a checkpoint skips training entirely!
- `--rl-learning-rate` / `--sl-learning-rate`: *(Optional)* You can dynamically drop or change the learning rates from what they were previously. If provided, the PyTorch optimizers are deeply updated on-the-fly.

---

## 3. Only Evaluate (MCTS Hybrid vs Pure NFSP)

If you have entirely finished training and just want to load a checkpoint to compare how well the Hybrid MCTS agent performs against the base NFSP agent, omit `--resume-training`. The pipeline will jump straight to Phase 2 & Phase 3.

```bash
uv run main.py \
    --load-checkpoint 500000 \
    --save-dir ./checkpoints \
    --hybrid-games 100 \
    --eval-games 100 \
    --mcts-depth 3 \
    --mcts-simulations 100 \
    --seed 42
```

**Options Breakdown:**
- `--load-checkpoint`: Loads the specified checkpoint phase to use for evaluation.
- `--hybrid-games`: Number of matches to run where players use the deep Hybrid MCTS layer over their learned policy.
- `--eval-games`: Number of matches to run where players use purely the baseline Average Policy. 
- `--mcts-depth`: Lookahead depth scaling for the hybrid system.
- `--mcts-simulations`: Leaf expansion iterations per decision tree run.
