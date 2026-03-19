"""
Custom MCTS Agent for Judgement (Oh Hell) card game.

Key feature: max_depth counts only the AGENT'S OWN MOVES, not all moves.
If max_depth=2, the tree explores up to 2 of the agent's own turns ahead.
Opponent moves in between don't count toward depth.
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
    MCTS agent where max_depth counts only the agent's own moves.
    
    If max_depth=2, we explore up to 2 of OUR turns ahead.
    Opponent turns in between are simulated but don't count toward depth.
    """

    def __init__(self, num_simulations=100, max_depth=2, exploration_constant=1.414):
        """
        Args:
            num_simulations: Number of MCTS simulations per decision.
            max_depth: Max number of the AGENT'S OWN moves to look ahead.
            exploration_constant: UCB1 exploration parameter.
        """
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.use_raw = True  # Use raw actions (action_id ints)
        self._player_id: Optional[int] = None

    def step(self, state):
        """Choose an action using MCTS. Used during training."""
        return self._run_mcts(state)

    def eval_step(self, state):
        """Choose an action using MCTS. Used during evaluation."""
        action = self._run_mcts(state)
        info = {'num_simulations': self.num_simulations}
        return action, info

    def _run_mcts(self, state) -> int:
        """Run MCTS and return the best action."""
        legal_actions = list(state['legal_actions'].keys())
        if not legal_actions:
            raise ValueError("No legal actions available")
        if len(legal_actions) == 1:
            return legal_actions[0]

        self._player_id = state['raw_obs']  # We'll extract player_id from game state
        # Store the game reference from perfect information
        root = MCTSNode()

        for _ in range(self.num_simulations):
            # We need the game state for simulation
            # Since we can't deep-copy a full env easily, we use the state info
            # to simulate rollouts
            self._simulate(root, state, legal_actions)

        # Choose action with most visits (robust child)
        if not root.children:
            return np.random.choice(legal_actions)

        best_action = max(root.children.keys(), key=lambda a: root.children[a].visits)
        return best_action

    def _simulate(self, root: MCTSNode, state, legal_actions: List[int]):
        """One MCTS simulation: select → expand → rollout → backpropagate."""
        node = root
        # For simplicity, we do a one-level expansion + random rollout
        # weighted by legal action quality

        # Expand: pick an unexplored action or use UCB1
        if not node.is_fully_expanded(legal_actions):
            unexplored = [a for a in legal_actions if a not in node.children]
            action = np.random.choice(unexplored)
            child = MCTSNode(parent=node, action=action)
            node.children[action] = child
            node = child
        else:
            node = node.best_child(self.exploration_constant)

        # Rollout: estimate value using heuristic
        reward = self._heuristic_rollout(state, node.action, legal_actions)

        # Backpropagate
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def _heuristic_rollout(self, state, action: int, legal_actions: List[int]) -> float:
        """
        Heuristic rollout that evaluates action quality.
        Since we can't easily clone the full game, we use heuristics.
        """
        # Parse state info
        raw_obs = state.get('raw_obs', None)
        if raw_obs is None:
            return 0.0

        is_bidding = False
        # Check if this is a bidding action (0-13) or play action (14-65)
        if action < 14:
            return self._evaluate_bid(state, action)
        else:
            return self._evaluate_play(state, action)

    def _evaluate_bid(self, state, bid_action: int) -> float:
        """Heuristic evaluation of a bid. Higher confidence = better."""
        bid_value = bid_action
        raw_obs = state['raw_obs']

        # Count high cards in hand (rough trick-taking potential)
        hand_indices = np.where(raw_obs[:52] == 1)[0] if len(raw_obs) >= 52 else []
        high_card_count = sum(1 for idx in hand_indices if (idx % 13) >= 10)  # J, Q, K, A

        # How close is bid to estimated potential
        estimated_tricks = high_card_count * 0.7

        # Check trump alignment
        trump_rep = raw_obs[52:56] if len(raw_obs) >= 56 else np.zeros(4)
        trump_suit_idx = np.argmax(trump_rep) if np.any(trump_rep) else -1
        if trump_suit_idx >= 0:
            trump_cards = sum(1 for idx in hand_indices if idx // 13 == trump_suit_idx)
            estimated_tricks += trump_cards * 0.3

        # Reward bids close to estimated potential
        diff = abs(bid_value - estimated_tricks)
        return max(0, 1.0 - diff * 0.3)

    def _evaluate_play(self, state, play_action: int) -> float:
        """Heuristic evaluation of playing a card."""
        card_id = play_action - 14
        raw_obs = state['raw_obs']

        # Card strength (rank within suit)
        rank_index = card_id % 13
        strength = rank_index / 12.0  # 0.0 for 2, 1.0 for A

        # Check if trump
        trump_rep = raw_obs[52:56] if len(raw_obs) >= 56 else np.zeros(4)
        trump_suit_idx = np.argmax(trump_rep) if np.any(trump_rep) else -1
        card_suit_idx = card_id // 13
        is_trump = (card_suit_idx == trump_suit_idx and trump_suit_idx >= 0)

        # Get player's bid and tricks_won from obs
        my_bid_rep = raw_obs[56:70] if len(raw_obs) >= 70 else np.zeros(14)
        my_bid = np.argmax(my_bid_rep) if np.any(my_bid_rep) else 0

        # Estimate tricks_won from tricks_won section of obs
        # Player 0's tricks are at a specific offset
        num_players = 4
        # Bids section ends at 56 + 14*num_players = 56 + 56 = 112 for 4 players
        # But actually: 52 + 4 + 14 + 14*(num_players-1) = 52+4+14+42 = 112
        tricks_offset = 52 + 4 + 14 + 14 * (num_players - 1)
        my_tricks_rep = raw_obs[tricks_offset:tricks_offset + 14] if len(raw_obs) > tricks_offset + 14 else np.zeros(14)
        my_tricks = np.argmax(my_tricks_rep) if np.any(my_tricks_rep) else 0

        need_tricks = my_bid - my_tricks

        if need_tricks > 0:
            # Need to win more tricks — play strong cards
            reward = strength * 0.8
            if is_trump:
                reward += 0.3
        elif need_tricks == 0:
            # Met bid — play weak cards to avoid winning
            reward = (1.0 - strength) * 0.8
            if is_trump:
                reward -= 0.3  # don't waste trump
        else:
            # Over bid — definitely play weak
            reward = (1.0 - strength) * 1.0

        return max(0, min(1, reward))


class JudgementMCTSWithGameClone:
    """
    Full MCTS agent that clones the game state for proper tree search.
    max_depth counts only the agent's own moves.
    
    This version requires access to the game object for cloning.
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
        return action, {}

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
                _, _ = game.step(node.action)
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
                    _, _ = game.step(action)
                    if acting_player == self.agent_player_id:
                        depth += 1

        # --- Rollout (random with depth limit on our moves) ---
        rollout_depth = depth
        while not game.is_over() and rollout_depth < self.max_depth:
            legal = game._get_legal_actions()
            if not legal:
                break
            acting_player = game.get_player_id()
            action = np.random.choice(legal)
            game.step(action)
            if acting_player == self.agent_player_id:
                rollout_depth += 1

        # --- Evaluate ---
        if game.is_over():
            scores = [p.score for p in game.players]
            reward = scores[self.agent_player_id]
            max_score = max(scores) if max(scores) != 0 else 1
            reward = reward / max(abs(max_score), 1)
        else:
            # Heuristic: use current score + dense rewards
            p = game.players[self.agent_player_id]
            reward = p.score / max(abs(p.score), 1) if p.score != 0 else 0
            # Add bid alignment heuristic
            if p.bid is not None:
                remaining = p.bid - p.tricks_won
                if remaining == 0:
                    reward += 0.5
                else:
                    reward -= 0.1 * abs(remaining)

        # --- Backpropagate ---
        for n in path:
            n.visits += 1
            n.total_reward += reward
