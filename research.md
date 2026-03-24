# Deep Architectural Analysis: Hybrid MC-NFSP Applied to the Game of Judgement

## Abstract
This document provides a rigorously detailed, recursively broken-down architectural analysis of the Hybrid Monte Carlo Neural Fictitious Self-Play (MC-NFSP) reinforcement learning agent applied to the modern trick-taking card game *Judgement* (an advanced variant of Oh Hell). It details every single decision point in the pipeline: the state tensor, the discrete action space, the mathematical justification for binary reward matrices, the exact mechanics of Prioritized Experience Replay (PER), Neural Fictitious Self-Play (NFSP), the dynamic computational barriers of IS-MCTS, and the strict Most Constrained Variable (MRV) determinization algorithm designed to simulate through imperfect information.

---

## 1. Core Game Theoretic Formulation

Judgement is defined as an $N$-player, zero-sum equivalent, imperfect-information, extensive-form game. It operates on sequential decision-making divided violently across two heterogeneous phases.

### 1.1 The Bidding Phase and The Hook Rule
The mathematical core of Judgement lies in the Bidding Phase. Players observe their private hands, the aggregate number of cards deployed this round ($C_{round}$), and the globally revealed Trump suit. Sequentially, players declare exactly how many tricks they mathematically expect to win ($Bid \in [0, 13]$).

* **The Hook (Dealer Constraint)**: To prevent universal prosperity, the rules mandate that the sum of all bids cannot exactly equal $C_{round}$. This guarantees that at least one player is mathematically forced to fail their objective.
* ***Implementation Oddity***: Our environment structurally enforces this constraint globally. Within `judger.py`, `forbidden_bid = num_cards - sum(previous_bids)`. If the current player is the last in sequence to bid, `forbidden_bid` is statically masked out of their legal action array. If the neural network hallucinates this bid via raw logit values, the framework clamps its probability exactly to `-inf`.

### 1.2 The Playing Phase and Suit Constraints
Post-bidding, players sequentially drop cards into the center trick. The cardinal rule of the phase is **Following Suit**: players *must* play a card matching the suit of the lead card on the table, if they physically possess one.
* **Void Constraint**: If a player is void (does not have a matching suit), the physics constraints lift. They may arbitrarily discard off-suit or drop a Trump card. These rules heavily dictate the structure of our subsequent Determinization algorithms inside the MCTS layer.

---

## 2. Objective Function and Exact Score Reward Architectures

### 2.1 The Critical Shift to Binary (+1.0 / -1.0) Terminal Payoffs
Traditional Judgement aggregates numerical scores linearly across multiple rounds dealing 1 to 13 cards. For example, a successful bid of 4 tricks usually nets `10 + 4 = 14` points, while failing nets `- (gap)` or `0` points.

**Why did we swap to strict +1/-1 scoring?**
This architecture runs self-play training solely over isolated 13-card rounds, rather than continuous game aggregations. Operating a DQN on aggregate scoring gradients (`+14` vs `+10` vs `-3`) creates severe topological distortion in the reward landscape. 
* *The Distortion*: A DQN targeting cumulative numbers will structurally bias toward strategies that safely bid high or chase maximum trick-taking to linearly scale raw Q-values. It prioritizes maximizing volume over precise prediction.
* *The Correction*: By swapping to a stark binary $+1.0$ (Success, $tricks\_won == bid$) and $-1.0$ (Failure, $tricks\_won \neq bid$) configuration, the objective function normalizes identically for all bids. The DQN is no longer learning "how many points can I score"; it is mathematically forced to output exactly $Q(s, a) \approx P(Success \mid s, a)$. It learns exact-bid probability, perfectly uncoupling trick volume from success evaluation. 

### 2.2 Dense Trick-Level Shaping
Because 13 sequential tricks must resolve before receiving the sparse terminal binary $+1/-1$ signal, DQN targets suffer an extreme credit-assignment block early in training. We inject immediate dense shaping feedback after *every single trick* by auditing the remaining gap: $gap = bid - tricks\_won$.

| State Context After Trick | Has the Agent Won This Trick? | Tactical Condition | Reward Emitted | Algorithmic Rationale |
|-------------------|------------|-----------|----------------|-----------------------|
| $gap > 0$      | **Yes**        | On Track  | $+0.5$         | Successfully marching toward the bid. Encourages winning when explicitly needed. |
| $gap < 0$ | **Yes**        | **Excess**    | **$-1.0$**         | Agent won a trick it did not fundamentally want. Severe penalization to implicitly teach the value of low-card "ducking" or throwing off-suit. |
| $gap == 0$        | **No**         | Ducking   | $+0.5$         | The bid is currently met; the agent successfully avoided an unwanted trick. Positively reinforces throwing non-winning cards. |
| $gap > 0$         | **No**         | Lagging   | $-0.5$         | Needed the trick, but lost it. Mild correction. |
| $gap < 0$         | **No**         | Damaged   | $-0.3$         | Lost trick, but the bid was already exceeded prior to this step. Reduced penalty to prevent total gradient collapse (giving up) in theoretically doomed states. |

---

## 3. Recursive Detail: State Representation and Action Space

To rigorously uphold the Markov property across an imperfect observation space, the historical footprint is flattened into an expansive memory vector.

### 3.1 The Action Space ($|A| = 66$)
The topological dimension of the agent's output is identically map-sized to 66 discrete elements:
* **$A \in [0, 13]$**: Pure Bid actions corresponding to $0, 1, 2... 13$.
* **$A \in [14, 65]$**: Card play actions. Card ID $i \in [0, 51]$ is mapped to network output index $i + 14$. 
* Dense positional masking (`-inf`) isolates and prevents cross-contamination. An agent physically cannot trigger a card play node while positioned in the bidding phase.

### 3.2 The Observation Vector ($S \in \mathbb{R}^{454}$)
The absolute state representation for a standard 4-player architecture transforms into a dense `float32` tensor of precisely **454 dimensions**. It decomposes recursively as follows:

1.  **Agent's Active Hand (52)**: One-hot embedding of currently held cards.
2.  **Trump Suit Indicator (4)**: One-hot array for [Spades, Hearts, Diamonds, Clubs]. All zeroes indicating purely No-Trump rounds.
3.  **Agent's Current Bid (14)**: One-hot indicating the agent's bid (0-13). Pure zeros explicitly encode an unbid state.
4.  **Opponents' Bids ($14 \times 3 = 42$)**: Three continuous 14-bit arrays encoding the visible bids of the three adversaries. 
5.  **Tricks Won Tracker ($14 \times 4 = 56$)**: Four 14-bit arrays. It logs exactly how many individual tricks (0-13) each table entity has already conquered. 
6.  **Current Trick Board Center ($52 \times 4 = 208$)**: The most structurally heavy component. Instead of a flat 52-bit space for the center pile, it separates into four distinct 52-bit slots. Slot $i$ holds the specific card thrown by player $i$. This implicitly preserves throwing sequence and trick ownership.
7.  **Played Cards Global Memory (52)**: A flat multi-hot vector accumulating *every* card discarded in preceding rounds. This functions as an explicitly computed proxy for memory gating mechanisms (LSTMs), allowing MLPs to mathematically compute card counting and deduction.
8.  **Round Deal Size (14)**: One-hot encoding the total trick volume ($1-13$) active for the entire iteration.
9.  **Phase Indicator (1)**: Binary threshold. $1.0$ activates Bidding weights, $0.0$ activates card-playing logic.
10. **Active Player Pivot (4)**: One-hot mapping representing entity priority.
11. **Continuous Game Scale (1)**: Scalar variable defined as $\frac{round\_index}{total\_rounds} \in [0,1]$.
12. **Domain-Engineered Hand Strength Features (6)**: Strategic heuristics appended directly to the end of the state to accelerate early gradient convergence before the layers discover these latent representations natively:
    * `trump_count_ratio`: Float density of Trumps in hand vs total initial cards.
    * `trump_high_ratio`: Float density of Face Trumps (10, J, Q, K, A) vs total cards.
    * `trump_avg_rank`: Scaled mean mathematical rank of held trumps.
    * `non_trump_count_ratio`, `non_trump_high_ratio`, `non_trump_avg_rank`: Symmetrical statistical density checks for non-trump suits.

---

## 4. NFSP (Neural Fictitious Self-Play) Configuration Details

NFSP is mathematically designed to approximate a strict Nash Equilibrium without exponentially modeling historical strategy trees. It converges to an optimal baseline by blending Q-learning against historical probabilities.

### 4.1 Topology and Architectures
* **Q-Network (Reinforcement Learning - The Greed Vector)**: Solves the Bellman equation across imperfect bounds. Engineered using PyTorch modules with layer proportions: `Linear(454) → Linear(1024) → ReLU → Linear(512) → ReLU → Linear(256) → ReLU → Linear(66)`.
  * *Loss*: Mean Squared Error (MSE), optimized via Adam (`lr = 0.001`). 
  * *Target Integration*: Uses a hard Double DQN Target Network, refreshing rigidly every $500$ update pulses to anchor exploding Q-value approximations.
* **Average Policy Network (Supervised Learning - The Safe Vector)**: Tracks the historical mean of agent actions. Identical layout to the Q-network but terminating in a `LogSoftmax(dim=1)` distribution.
  * *Loss*: Cross-Entropy (Negative Log Likelihood), optimized via Adam (`lr = 0.005`).

### 4.2 The Anticipatory Action Parameter ($\eta$)
During physical $N \times 13$ step self-play rollouts, the agent queries an internal dice.
* With mathematical probability $\eta = 0.15$, it leverages its DQN (Greedy Policy) and plays to maximize immediate expected prediction likelihood.
* With mathematical probability $1 - \eta = 0.85$, it samples directly against its Average Policy Network ($\Pi_{SL}$). This prevents the model from hard-coding cyclical exploits and discovering dominant pure strategies that are ultimately fragile against shifting opponents.

### 4.3 Reservoir Sampling Metrics
To prevent the Average Average Policy Network from undergoing Catastrophic Forgetting over millions of training loops, it caches behavioral samples in a rigid **Reservoir Buffer** holding up to $350,000$ independent exact $(s, a)$ frames. The sample inclusion math naturally ensures the buffer stays statically uniform relative to the infinite length of the training loop.

---

## 5. Prioritized Experience Replay (PER) Math and Structure

Standard Deep Q-Networks pull transitions linearly and continuously. Because 80% of Judgement decisions are strict uninteresting choices (forced to follow suit), uniform buffering drowns the network in useless noise (like playing an 8 of Clubs when that is the only legal Club held). PER surgically samples transitions that mathematically shock the network prediction models.

### 5.1 The Binary SumTree
PER requires sub-linear storage arrays. It is engineered physically as a flattened array modeling a continuous binary `SumTree` of size $2N$ (where $N = 350,000$).
* **Memory Footprint Overhead**: A standard DQN buffer strictly holds a list of $N$ transition tuples in RAM. A PER buffer holds those same $N$ transitions *plus* the $2N$-sized `float32` tree array. Therefore, PER inherently consumes more operational memory than uniform dictionaries of equivalent capacity.
* **Data Layer**: The second half of the array bounds (indices `[N-1]` to `[2N-2]`) securely stores the immediate scalar priorities $p_i$.
* **Routing Layer**: The first half stores the composite sums allowing mathematical threshold searches to execute via $O(\log N)$ traversal jumps.
* ***Implementational Bug/Fix Oddity***: Due to 0-indexing math, the logical `left_child = 2 * tree_idx + 1`. For the very first leaf `N-1`, the left child computes to `2N-1`, which exists *within* the Python array bounds but points functionally to an unallocated cell mapping to out of bounds data limits. Traversing algorithms must explicitly terminate if `tree_idx >= N - 1`, completely disregarding bounding checks on child arrays.

### 5.2 Probabilistic Formulas
Transitions arriving dynamically are clamped immediately to $P_{max} = \infty \approx 1.0$ to guarantee they receive at least a single evaluation pass. Post-evaluation, the priority shrinks relative to calculation differences:
$p_i = |\delta_i| + \epsilon$
where $\delta_i = R_{dense} + \gamma \cdot Q_{target}(s', \arg\max_{a'} Q_{DQN}(s', a')) - Q_{DQN}(s, a)$. We apply $ \epsilon=1e^{-5}$ to guard against dividing zero probabilities and permanently burying transitions.

Batch sampling runs probabilistically:
$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$
where we constrain the distribution sharpness using **$\alpha = 0.6$**, highly biasing the system toward pulling unexpected rewards or disastrous trick failures.

### 5.3 Importance Sampling weight debiasing
PER technically alters data distributions to look statistically distinct from the game's actual probability state. Left unadjusted, the Q-values artificially skew and the policy shatters. We correct raw calculation steps by aggressively shifting back using derived Importance Sampling (IS) weights applied straight into the `Estimator.update()` logic:
$w_i = \left( \frac{1}{N \cdot P(i)} \right)^\beta \cdot \frac{1}{\max_j w_j}$

* **Beta ($\beta$) Annealing Synchronization**: Standard architectures usually leave $\beta$ as an arbitrary constant. We initialize $\beta = 0.4$ and linearly ramp it to $1.0$ over an absolute horizon exactly measuring $2,100,000$ updates. This gradient mathematically, and cleanly, matches the exact span that the DQN randomly explores ($\epsilon$-greedy dropping from $1.0 \rightarrow 0.0$ across $2.1$ million). When training reaches stability and exploring ceases at $\epsilon=0$, $\beta=1.0$ guarantees mathematically perfectly unbiased gradients mapping strictly back to the true game value function.

---

## 6. Asymmetric Hybrid MCTS Integrations

While NFSP produces incredibly theoretically robust agents, real-time prediction without an active search branch historically encounters localized hallucinations. The `HybridMCNFSPAgent` resolves this by spinning up an Information Set Monte Carlo Tree Search (IS-MCTS) to pre-evaluate actions before committing sequentially.

### 6.1 Why IS-MCTS is Grossly Computationally Expensive
Classical "Perfect Information" MCTS logic (e.g., AlphaZero applying bounds to Chess) operates iteratively upon a shared static truth state. Nodes step via microscopic boolean checks taking nanoseconds.

IS-MCTS is constrained by the mathematical "Fog of War." The agent physically does not know the true hidden state of the opponents.
* **Determinization Physics Generation**: Before a tree simulation can run, the model must *guess* the opponents' states through "Determinization"—hallucinating the layout of hidden cards legally.
* **Full Logic Instantiation**: Every unique Determinization requires completely mirroring the absolute entire game state. The Python physics ruleset (playing algorithms, trick logic, trick winner logic) must execute fully out, continuously taking hundreds of microseconds merely to iterate node physics forward.
* **Neural Interfacing**: MCTS operations traditionally do random bit-rollouts. We instead evaluate utilizing full PyTorch tensor queries across the GPU cluster at virtually every single tree node logic bound, exponentially increasing CPU $\leftrightarrow$ GPU memory traffic delay states.

### 6.2 The MRV Void Determinization Engine (Constraint Mechanics)
Determinization is the process of manifesting a distinct, theoretically valid game universe prior to a tree walk. To stop the system from building mathematically impossible tables, logic operates functionally:
1. **Accumulate Known Elements**: The table aggregates agents' hand constraints, raw history outputs, and immediate trick board sets ($K$).
2. **Sub-Deck Remainder Tracking**: The $52$-deck array deletes intersection $K$, generating the "True Unknown Cards" list array $U$.
3. **Void Extraction Pipeline**: The parser crawls sequentially down `$game.trick_history$`. If the engine locates a case where `lead_suit = 'Spade'` and Player 2 played `'Heart'`, the engine strictly asserts `VOID[2]['Spade'] = True`.
4. **Greedy MRV Delivery Logic**: The engine randomly shuffles $U$. However, strictly sequential dealing easily encounters hard constraints near the end of the subset (e.g. forced to deal a Spade to a player tagged as strictly void in Spades, destroying the simulation fidelity and breaking downstream CNN layers). 
  * *Solution*: Opponent targets are numerically sorted using a **Most Constrained Variable (MRV)** logic descending sequence based explicitly on their sum totals of restricted void classes. The engine distributes unknown cards exclusively to the most highly restricted subjects first.

### 6.3 Mechanics of The Asymmetric MCTS Engine
1. **Asymmetric Tree Logic**: Typical MCTS builds massive recursive nodes recursively encompassing all external players. Our logic strips external players. It functions mathematically *asymmetrically*—nodes exclusively branch representing **only** the Acting Agent's valid choices. 
2. **Stochastic Opponent Sampling**: Whenever the simulator iteration runs into an opponent's physical turn bound, the tree does not split. Instead, the simulation targets that particular opponent's `policy_network` representation from `all_nfsp_agents`, grabs the LogSoftmax probabilities, pulls `np.random.choice()` against the normalized array, and blindly forces the choice step forward. Therefore, the opponents functionally blur seamlessly into the base stochastic Markov transition properties of the active background.
3. **PUCT Edge Traversals**: Action edges apply scaling formulations referencing UCB1 bounding $Value_a = Q(s,a) + c \sqrt{\frac{\ln N}{n_a}}$ where exploration explicitly sits at $c=1.414$.
4. **Dynamic Scaling and Budget Thresholds**: In the 14-width Bidding block, structural MCTS samples 200 simulation bounds. During late-game trick playing (where physical action spaces collapse from suit limits to widths of 2 or 3 choices), static determinization models crash functionally due to over-concentration of statistical errors over physical guesses. The engine measures legal array bounds and strictly multiplies the simulation budget limit (e.g., $200 \times 4 = 800$), directly applying computational force to solve specific variance "determinization chaos" zones.
5. **Shallow Depth Leaf Averaging**: The CPU cost physically mandates depth restrictions spanning only $\leq D=2$ of the primary agent action loops.
   * **The Exact Leaf Mix Equation**: At depth cut-offs, leaf state $s_{eval}$ predicts raw Q values matching $Q_{greedy} = \max_a Q_{DQN}(s, a)$.
   * To prevent the raw neural network from getting lost analytically analyzing short-term trick bounds scaling to the exclusion of achieving the overarching round structure, a separate continuous exact logical model $H(bid) \in [-1.0, 1.0]$ bounds trick-distance expectations.
   * The returned MCTS value strictly computes $V_{leaf} = 0.7(Q_{greedy}) + 0.3(H(bid))$, blending raw network pattern recognition functionally against rigid mathematical structural safety anchors.
