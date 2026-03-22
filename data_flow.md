# Judgement RL Data Flow & Transformation Pipeline
**The Comprehensive Source of Truth for Technical Contributors**

This document comprehensively maps out the exact state transformations, neural network pathways, hyperparameter impacts, and reward alignment math used to train the Deep Q-Network algorithms on the Judgement card game environment. 

---

## 1. State Extraction & Encoding ([judgement/env.py](file:///home/youhan/Documents/projects/temp/judgement/judgement/env.py))
Before the neural network can decide what to do, the current snapshot of the game is converted into a mathematical tensor. The Judgement Environment builds a fixed-size `[1, 454]` array. 

### The `454` Tensor Breakdown (Data Dictionary)
Every slice of the 454 numbers is normalized specifically so that the Neural Network inputs strictly fall into the mathematical bounding box of `[0.0, 1.0]`. If values naturally exceed standard bounds (like bidding 14 tricks), they are strictly capped using `min(val, 13)`.

| Features | Size | Encoding Type | Description & Normalization Logic |
| :--- | :--- | :--- | :--- |
| **Player Hand** | 52 | Multi-Hot | `1.0` if you hold the card natively, else `0.0`. |
| **Trump Suit** | 4 | One-Hot | `1.0` at the index of the current trump suit (Spades/Hearts/Clubs/Diamonds). |
| **Your Bid** | 14 | One-Hot | `1.0` at index representing your bid `[0 to 13]`. Caps inherently. |
| **Enemy Bids** | 42 | One-Hot | 3 distinct arrays of size 14, tracking the other 3 players' bids respectively. |
| **Tricks Won** | 56 | One-Hot | 4 arrays of size 14. Tracks tricks won so far for all 4 players `[0 to 13]`. |
| **Current Trick** | 208 | Multi-Hot | 4 arrays of size 52. Tracks exactly which specific card was played by which player in the *current* incomplete trick. |
| **Played Cards** | 52 | Multi-Hot | `1.0` for all cards that have been successfully exhausted in previous completed tricks. |
| **Round Capacity**| 14 | One-Hot | Tracks how many maximum cards are dealt this round (varies per round execution). |
| **Is Bidding?** | 1 | Boolean Float | `1.0` if the game is currently in the bidding phase. `0.0` if card play phase. |
| **Current Player** | 4 | One-Hot | `1.0` indicating which of the 4 player seats is currently acting natively. |
| **Round Progress**| 1 | Fractional | `round_index / total_rounds` mapping the progressive timeline to exactly `[0.0, 1.0]`. |
| **Heuristics** | 6 | Fractional | 6 continuous floats modeling hand-strength logic: <br>1. `% of hand that is trump`<br>2. `% of trumps that are high (10+)`<br>3. `Average rank of trumps divided by max rank 12.0`<br> *(Plus 3 identically scaled values corresponding to non-trumps)* |
| **Total Size** | 454 | | |

---

## 2. Action Selection & Hyperparameters ([rlcard/agents/nfsp_agent.py](file:///home/youhan/Documents/projects/temp/judgement/.venv/lib/python3.12/site-packages/rlcard/agents/nfsp_agent.py))
At every timestep linearly, the Network receives the `[1, 454]` array and outputs an internal mapped vector of length 66 (52 possible card plays + 14 bid options).

### Execution Pathway & Parameters
1. **Epsilon Randomness (`q_epsilon_decay_steps`):**
   * **What it does:** Epsilon represents the exact percent chance `[0.0 to 1.0]` the agent ignores the Neural Network and picks a completely random legal action to force massive game exploration mathematically.
   * **How it decays:** Every time `agent.feed()` processes a new state array, its internal `total_t` step counter increments by 1. The script uses the `q_epsilon_decay_steps` constant to calculate exactly how fast to drop the randomness from 100% logarithmically down to a permanent flat floor of exactly 10%. 
   * **Modifying Impact:** Extending decay to `2.1 million steps` (roughly 150,000 games) forces the agents to heavily populate their replay buffers with volatile, non-deterministic gameplay logs. This inherently prevents early overfitting/stunting but completely obscures any metric convergence visibility until the decay flattens out at the end of the duration.
2. **Anticipatory Parameter (`anticipatory_param=0.15`):**
   * **What it does:** The specific NFSP innovation logic: When it doesn't choose randomly, exactly 15% of the time the agent will select an action using its mathematically mapped Supervised Learning (SL) *Average Policy Network* instead of its greedily driven *Best-Response Q-Network*.
   * **Impact:** This stabilizes reinforcement training exclusively by forcing agents to play a "historic average" behavior 15% of the time, making their tactical strategies less aggressively exploitable while training continuously.
3. **Greedy Q-Response (`argmax`):**
   * If both Epsilon and Anticipatory probability checks fail, the main Q-Network calculates the Q-value mathematical estimations for all valid remaining actions and uses generic `numpy.argmax` to enact the single highest numerical pathway.

---

## 3. The Dense Reward Framework ([judgement/round.py](file:///home/youhan/Documents/projects/temp/judgement/judgement/round.py) & [judgement/judger.py](file:///home/youhan/Documents/projects/temp/judgement/judgement/judger.py))

Judgement is specifically a non-Markovian game; an isolated card played intelligently on Trick 1 might accidentally force a crippling overbid penalty 12 tricks later on Trick 13. To give the network intermediate continuous breadcrumbs linking early-game actions to ultimate final outcomes, continuous floating "Dense Rewards" are injected at every localized trick step.

### A. The Bidding Phase Component:
*   **Disabled (Commented out):** The action of actively bidding currently returns exactly `0.0` dense reward physically. In isolated architectural testing, a heuristic algorithm `bid_reward = max(-1.0, 0.5 - (0.3 * absolute_difference))` actively provided a hand-strength expectation reward. This logic was purposely hard-disabled because explicit heuristics prematurely coerce the network into playing overly safely globally, overriding the neural network’s capacity to discover its own optimal dynamic trick-paths natively using the absolute Terminal End-Of-Game penalty structure itself.

### B. The Card Play Phase Component:
Every time exactly 4 cards are accumulated physically and the [Judger](file:///home/youhan/Documents/projects/temp/judgement/judgement/judger.py#11-170) declares a trick winner, it updates intermediate mathematical rewards stringently constrained mathematically inside `[-1.0, +1.0]` to safely prevent gradient explosion.

*   **WON TRICK:** 
    *   If they still actively needed tricks remaining for their bid: `+0.5`
    *   If they had already hit their bid cap (actively overbidding penalty): `-1.0`
*   **LOST TRICK:** 
    *   If they perfectly hit their bid (actively avoiding excess): `+0.5`
    *   If they critically needed tricks: `-0.5`
    *   If they previously overbid (punish excessive bleeding natively): `-0.3`

These continuous floating breadcrumbs are tracked exclusively and persistently in the `self.dense_rewards[player_id]` accumulator array.

---

## 4. The Terminal Finish & Reorganization ([agents/nfsp_runner.py](file:///home/youhan/Documents/projects/temp/judgement/agents/nfsp_runner.py))

When all 13 tricks officially conclude mathematically, the Judger fires the **Terminal Payoffs** explicitly based on tactical bid execution success.
*   **If Exact (`bid == tricks_won`):** Terminal Payoff is permanently `+1.0`
*   **If Missed:** Terminal Payoff is permanently `-1.0`
*(Note: Win/Miss specifically uses extremely discrete binary payoffs `[-1, 1]` universally mapped into the core RLCard transition architecture, ensuring standard Bellman convergence math scales properly without extreme horizon variance).*

### Matrix Reorganization logic
The native game logger dumps raw state arrays. [reorganize_dense()](file:///home/youhan/Documents/projects/temp/judgement/agents/nfsp_runner.py#16-54) translates them into normalized final `[state, action, reward, next_state, done]` Neural Network sequential memories:
*   `Step Reward = next_dense - curr_dense` (The specific mathematical delta tracking the localized Dense Reward physically earned specifically on that transition jump alone).
*   **The Terminal Injection:** If the transition array happens to be the exact localized last play of the game (`done == True`), it explicitly fuses the isolated binary `+1.0` or `-1.0` Terminal Payoff array permanently on top of whatever sparse dense trick reward was issued: `reward += payoffs[player]`.

---

## 5. Q-Learning Target Mathematics ([rlcard/agents/dqn_agent.py](file:///home/youhan/Documents/projects/temp/judgement/.venv/lib/python3.12/site-packages/rlcard/agents/dqn_agent.py))

All formally transitioned memories are piped actively into the massive initialized **Replay Buffers**. Every `q_train_every` localized frames, the Neural Network picks a totally random unbiased batch of exactly 512 array chunks exclusively drawn from the `Reservoir Buffer` (which specifically preserves a probabilistically perfect unbiased history of gameplay regardless of timescale limitations, contrasting sharply with FIFO arrays) and systematically pushes updates into its neural weights using MSE Gradient Descent natively.

### Target Formulas & Gamma Scaling
To shift the weights, the network continuously computes the exact Mean Squared Error natively between its arbitrary current estimation vector and the mathematically aligned `Q_target` absolute truth.

`Q_target = Step_Reward + (q_discount_factor * max(Q(next_state)))`

*   **The `0.95` Flaw & The Repercussion:** When Gamma specifically was locked to `0.95`, a terminal perfect reward of `1.0` dynamically occurring exactly 52 isolated steps functionally later in a game tree actively decayed fully exponentially before tying back to Trick 1: `1.0 * (0.95^52) = 0.069`. Because `0.069` natively was such a negligible mathematical blip, the neural network could intrinsically not adequately isolate/assign specific trick credit/blame from its deep early opening hand to accurately scale the ultimate penalty/bonus scalar of physically missing the native target bid limit.
*   **The `0.995` Architectural Shock Correction:** Updating the strict scalar uniquely maps the massive 52-step decay math mathematically differently to `1.0 * (0.995^52) = 0.77`. The strict numerical mathematical gravity of a terminal `1.0` scalar survives intact cleanly mapping exponentially backward through the game nodes, completely and violently adjusting all internal optimizer tensor paths and functionally resulting continuously in significantly oversized RL-loss training metrics (`~4.6`) temporarily as the entire suite of neural node weights stretch effectively to the much higher functional magnitude spectrum heavily enforced by the new mathematical framework.

---

## 6. The [training_metrics.csv](file:///home/youhan/Documents/projects/temp/judgement/training_metrics.csv) Logger ([agents/nfsp_runner.py](file:///home/youhan/Documents/projects/temp/judgement/agents/nfsp_runner.py) & [main.py](file:///home/youhan/Documents/projects/temp/judgement/main.py))

Every precisely 500 contiguous episodes cleanly executed, an explicit continuous tracker computes exact mathematical arithmetic averages mapping directly to the CSV pipeline strings.

*   `avg_payoff`: Essential generic `np.mean()` mapping of explicitly the absolute discrete Terminal `-1` or `+1` physical binary variables uniformly scored over the entire isolated batch vector of the historical 500 games sequentially. (For instance: A continuous static `-0.8` numeric scalar equates mathematically to actively perfectly winning specifically 10% locally and completely losing exactly 90% comprehensively across that specific array execution).
*   `won_pct`, `under_pct`, `over_pct`: Uniform explicit strict integer fractions uniquely mapping stringently to concrete localized condition outcomes (`tricks == bid`, `tricks < bid`, and natively `tricks > bid`). The scalar values actively map stringently dividing native counts exclusively by exactly 500 total evaluation logic runs globally directly enforcing standardized percentages accurately contained stringently universally within precisely `[0.0, 100.0]`.

### The Training Metric Distortion Noise:
The raw CSV outputs functionally outputted identically linearly inherently during the actively executed `600k` script actively structurally suppress the visual representation array output of actual unvarnished natively trained `won_pct` visual outputs automatically exactly. 
Why? Because the static `env.run(is_training=True)` validation array algorithm universally forces explicitly the Epsilon mathematical randomness mechanics and Anticipatory fractional variables perpetually `ON` inherently active continually. Since allied and completely active enemy agents continually inject mathematical randomization noise directly into all trick environments randomly, hitting exact optimal physical integer targets explicitly relies functionally massively on uncontained chaos locally significantly. 

### The Pure Phase Verification Model (`--eval-games`):
To reliably and comprehensively evaluate the native isolated mathematically optimized neural tactical algorithmic intelligence locally independently of the actively chaotic volatile arrays uniformly tied cleanly within the native Replay Buffers sequentially universally, the parameter string natively boots exclusively using `is_training=False` boolean universally consistently globally completely overriding Epsilon functionality to integer exact `0.0` natively statically actively isolating standard network deterministic generic logic precisely.
