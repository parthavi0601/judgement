# Judgement AI — Setup & Command Guide

This guide covers exactly what dependencies you need to download and the exact commands to run the project from start to finish.

---

## 1. Dependencies to Download (One-Time Setup)

You need the following installed on your system before running any code.

### System Requirements
1. **Python 3.10+** (Install from [python.org](https://www.python.org/downloads/))
2. **C++ Build Tools**:
   - **Windows:** Download [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). During installation, make sure you check the box for **"Desktop development with C++"**.
   - **Linux:** Install GCC/G++ 9 or higher (`sudo apt install build-essential`).

### Python Packages
Open your terminal (in your virtual environment) and run:
```bash
python -m pip install pybind11
python -m pip install torch torchvision torchaudio
python -m pip install rlcard
python -m pip install numpy
python -m pip install pytest
```

---

## 2. Compile the C++ Engine (One-Time Setup)

Because we moved the heavy game logic and MCTS tree search into C++ for a 50x-100x speedup, you must compile it.

1. Open your terminal in the `judgement` folder.
2. Run the compilation command:
```bash
python -m pip install -e .
```
*(Note: Do not forget the `.` at the end. It tells pip to build the current directory).*

---

## 3. Train the NFSP Models

To train the underlying Neural Networks (NFSP) from scratch, run the following command. The C++ engine handles the gameplay automatically.

```bash
# Standard training (200,000 episodes) - Saves every 10,000 games
python main.py --nfsp-episodes 200000 --evaluate-every 2000 --checkpoint-every 10000 --save-dir ./checkpoints

# Full-scale training (2,000,000 episodes) - Saves every 50,000 games
python main.py --nfsp-episodes 2000000 --evaluate-every 10000 --checkpoint-every 50000 --save-dir ./checkpoints
```

---

## 4. Resume Training (If you stopped the script)

If you stopped the training script but want to continue from a saved checkpoint, use the `--resume-training` tag:

```bash
# Example: Resuming from episode 50,000 for another 150,000 episodes
python main.py --load-checkpoint 50000 --resume-training --nfsp-episodes 150000 --save-dir ./checkpoints
```

---

## 5. Evaluate the Hybrid Agent

Once you have a trained PyTorch model saved in `./checkpoints`, you can plug those models into the **C++ Hybrid MCTS Agent** to see how much stronger it is compared to the pure neural network.

```bash
# Example: Evaluate 50 games of Hybrid MCTS vs Pure NFSP using episode 200,000 checkpoint
python main.py --load-checkpoint 200000 --hybrid-vs-pure-games 50 --mcts-depth 2 --mcts-simulations 200

# High-quality evaluation (More simulations = Smarter MCTS, but slower)
python main.py --load-checkpoint 200000 --hybrid-vs-pure-games 50 --mcts-depth 3 --mcts-simulations 500
```

---

## 6. Play Against the AI Yourself

Want to test the AI manually? You can step into the arena yourself using the `play_human.py` script.

```bash
# Play against standard NFSP from checkpoint
python play_human.py --load-checkpoint 200000 

# Play against the Hybrid MCTS Agent (much harder)
python play_human.py --load-checkpoint 200000 --use-mcts
```

---

## 7. Run Automated Tests

To double check that the C++ logic perfectly matches the game rules, run the test suite:

```bash
python -m pytest tests/ -v
```
