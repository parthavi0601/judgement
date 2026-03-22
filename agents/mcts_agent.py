"""
Custom MCTS Agent for Judgement (Oh Hell) card game.

Key feature: max_depth counts only the AGENT'S OWN MOVES, not all moves.
If max_depth=2, the tree explores up to 2 of the agent's own turns ahead.
Opponent moves in between don't count toward depth.

Uses full game cloning for proper tree search and incorporates the same
bid reward heuristic from round.py for evaluating bid actions.
"""

import math
import copy
import numpy as np
from typing import List, Optional, Dict


class MCTSNode:
    """A node in the MCTS tree."""

    def __init__(self, parent=None, action=None, player_id=None):
        self.parent: Optional[MCTSNode] = parent
        self.action: Optional[int] = action  # The action that led to this node
        self.player_id: Optional[int] = player_id  # Who acted to reach this node
        self.children: Dict[int, MCTSNode] = {}
        self.visits: int = 0
        self.total_reward: float = 0.0
        self.is_terminal: bool = False

    @property
    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def ucb1(self, exploration_constant: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        exploitation = self.q_value
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    def best_child(self, exploration_constant: float = 1.414) -> 'MCTSNode':
        return max(
            self.children.values(),
            key=lambda c: c.ucb1(exploration_constant)
        )

    def is_fully_expanded(self, legal_actions: List[int]) -> bool:
        return all(a in self.children for a in legal_actions)


class JudgementMCTSAgent:
    """
    Full MCTS agent that clones the game state for proper tree search.
    max_depth counts only the agent's own moves.

    Incorporates bid reward heuristics from the game's round logic for
    evaluating bid actions, and uses bid-alignment-aware rollout policy.
    """

    def __init__(self, env, agent_player_id, num_simulations=200, max_depth=2,
                 exploration_constant=1.414):
        self.env = env
        self.agent_player_id = agent_player_id
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.use_raw = True

    def step(self, state):
        return self._run_mcts(state)

    def eval_step(self, state):
        action = self._run_mcts(state)
        return action, {'agent': 'mcts', 'simulations': self.num_simulations}

    def _run_mcts(self, state) -> int:
        legal_actions = list(state['legal_actions'].keys())
        if len(legal_actions) <= 1:
            return legal_actions[0] if legal_actions else 0

        root = MCTSNode()

        for _ in range(self.num_simulations):
            # Clone the game for simulation
            game_clone = copy.deepcopy(self.env.game)
            self._one_simulation(root, game_clone, legal_actions)

        if not root.children:
            return np.random.choice(legal_actions)
        return max(root.children.keys(), key=lambda a: root.children[a].visits)

    def _one_simulation(self, root: MCTSNode, game, legal_actions: List[int]):
        """Run one MCTS simulation with agent-move-only depth counting."""
        node = root
        depth = 0  # counts only OUR moves
        path = [node]

        # --- Selection ---
        current_legal = legal_actions
        while node.children and node.is_fully_expanded(current_legal) and not node.is_terminal:
            node = node.best_child(self.exploration_constant)
            # Execute action in cloned game
            if not game.is_over():
                acting_player = game.get_player_id()
                game.step(node.action)
                if acting_player == self.agent_player_id:
                    depth += 1
            path.append(node)
            if game.is_over():
                node.is_terminal = True
                break
            current_legal = game._get_legal_actions()

        # --- Expansion ---
        if not node.is_terminal and not game.is_over() and depth < self.max_depth:
            current_legal = game._get_legal_actions()
            unexplored = [a for a in current_legal if a not in node.children]
            if unexplored:
                action = np.random.choice(unexplored)
                acting_player = game.get_player_id()
                child = MCTSNode(parent=node, action=action, player_id=acting_player)
                node.children[action] = child
                node = child
                path.append(node)

                if not game.is_over():
                    game.step(action)
                    if acting_player == self.agent_player_id:
                        depth += 1

        # --- Rollout (smart heuristic policy with depth limit on our moves) ---
        rollout_depth = depth
        while not game.is_over() and rollout_depth < self.max_depth:
            legal = game._get_legal_actions()
            if not legal:
                break
            acting_player = game.get_player_id()
            action = self._rollout_policy(game, legal, acting_player)
            game.step(action)
            if acting_player == self.agent_player_id:
                rollout_depth += 1

        # --- Evaluate ---
        reward = self._evaluate_state(game)

        # --- Backpropagate ---
        for n in path:
            n.visits += 1
            n.total_reward += reward

    def _rollout_policy(self, game, legal_actions: List[int], acting_player: int) -> int:
        """
        Smarter-than-random rollout policy.
        For bidding: use bid heuristic to pick a reasonable bid.
        For card play: use bid-alignment-aware card selection.
        """
        if not legal_actions:
            return 0

        # Check if we're in bidding phase
        if game.current_round and game.current_round.is_bidding:
            return self._rollout_bid_policy(game, legal_actions, acting_player)
        else:
            return self._rollout_play_policy(game, legal_actions, acting_player)

    def _rollout_bid_policy(self, game, legal_actions: List[int], acting_player: int) -> int:
        """Pick a bid based on hand strength heuristic."""
        player = game.players[acting_player]
        trump_suit = game.current_round.trump_suit if game.current_round else None

        expected_tricks = self._estimate_tricks(player.hand, trump_suit)

        # Pick the legal bid closest to expected tricks
        best_bid = min(legal_actions, key=lambda b: abs(b - expected_tricks))
        return best_bid

    def _rollout_play_policy(self, game, legal_actions: List[int], acting_player: int) -> int:
        """
        Bid-alignment-aware card play: if we need tricks, prefer strong cards;
        if we've met our bid, prefer weak cards.
        """
        player = game.players[acting_player]

        if player.bid is None:
            return int(np.random.choice(legal_actions))

        need_tricks = player.bid - player.tricks_won
        trump_suit = game.current_round.trump_suit if game.current_round else None

        # Score each legal action
        scored = []
        for action_id in legal_actions:
            card_id = action_id - 14  # action IDs 14-65 map to card IDs 0-51
            if card_id < 0 or card_id >= 52:
                scored.append((action_id, 0.0))
                continue

            rank = card_id % 13
            suit_idx = card_id // 13
            strength = rank / 12.0
            is_trump = (trump_suit and ['S', 'H', 'D', 'C'][suit_idx] == trump_suit)

            if need_tricks > 0:
                # Need to win more — prefer strong cards and trumps
                score = strength
                if is_trump:
                    score += 0.4
            elif need_tricks == 0:
                # Met bid — play weakest cards to avoid winning
                score = 1.0 - strength
                if is_trump:
                    score -= 0.4
            else:
                # Over bid — dump weakest
                score = 1.0 - strength

            scored.append((action_id, score))

        # Softmax selection (temperature=0.5 for moderate greediness)
        scores = np.array([s for _, s in scored])
        scores = scores / 0.5
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        idx = np.random.choice(len(scored), p=probs)
        return scored[idx][0]

    def _estimate_tricks(self, hand, trump_suit) -> float:
        """Estimate expected tricks based on hand strength (matches round.py heuristic)."""
        expected = 0.0
        for c in hand:
            if c.suit == trump_suit:
                if c.rank_index >= 12:
                    expected += 1.0      # Ace of trump
                elif c.rank_index >= 11:
                    expected += 0.8      # King of trump
                elif c.rank_index >= 9:
                    expected += 0.5      # 10, J, Q of trump
                else:
                    expected += 0.2      # low trumps
            else:
                if c.rank_index >= 12:
                    expected += 0.5      # Ace off-suit
                elif c.rank_index >= 10:
                    expected += 0.2      # Q, K off-suit
        return expected

    def _evaluate_state(self, game) -> float:
        """
        Evaluate the game state for our agent.
        Uses compute_round_scores for terminal states,
        and bid-alignment heuristic for non-terminal states.
        """
        if game.is_over():
            return self._score_terminal(game)
        else:
            return self._heuristic_evaluate(game)

    def _score_terminal(self, game) -> float:
        """Use the game's official scoring: +1.0 for exact bid, -1.0 for miss."""
        from judgement.judger import JudgementJudger
        scores = JudgementJudger.compute_round_scores(game.players)
        return scores[self.agent_player_id]

    def _heuristic_evaluate(self, game) -> float:
        """
        Non-terminal evaluation using bid alignment.
        Primary signal: how well is the agent tracking toward its bid?
        """
        p = game.players[self.agent_player_id]

        if p.bid is None:
            return 0.0

        tricks_remaining = 0
        if game.current_round:
            tricks_remaining = game.current_round.num_cards - game.current_round.tricks_played

        needed = p.bid - p.tricks_won

        if needed == 0:
            # Already met bid — good position, reward proportional to
            # how few tricks remain (fewer = safer)
            safety = 1.0 - (tricks_remaining / max(game.current_round.num_cards, 1)) if game.current_round else 1.0
            return 0.3 + 0.4 * safety
        elif needed > 0:
            if tricks_remaining >= needed:
                # Still achievable — mild optimism
                achievability = needed / max(tricks_remaining, 1)
                return 0.1 * (1.0 - achievability)
            else:
                # Impossible to make bid — penalize
                return -0.5
        else:
            # Over bid — penalize based on how much over
            return -0.3 * min(abs(needed), 3)
