"""
Hybrid MC-NFSP Agent for Judgement (Oh Hell) card game.

Combines MCTS tree search with NFSP policy evaluation:
- MCTS explores up to max_depth of the agent's OWN moves
- Beyond max_depth, the trained NFSP average-policy network evaluates the
  leaf state instead of a random/heuristic rollout

This gives the planning strength of MCTS with the learned evaluation of NFSP.
"""

import copy
import math
import numpy as np
import torch
from typing import List, Optional, Dict
from collections import OrderedDict


class _MCTSNode:
    """A node in the MCTS search tree."""

    def __init__(self, parent=None, action=None, player_id=None):
        self.parent: Optional['_MCTSNode'] = parent
        self.action: Optional[int] = action
        self.player_id: Optional[int] = player_id
        self.children: Dict[int, '_MCTSNode'] = {}
        self.visits: int = 0
        self.total_reward: float = 0.0
        self.is_terminal: bool = False

    @property
    def q_value(self) -> float:
        return self.total_reward / self.visits if self.visits else 0.0

    def ucb1(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        return self.q_value + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self, c: float = 1.414) -> '_MCTSNode':
        return max(self.children.values(), key=lambda n: n.ucb1(c))

    def is_fully_expanded(self, legal_actions: List[int]) -> bool:
        return all(a in self.children for a in legal_actions)


class HybridMCNFSPAgent:
    """
    Hybrid Monte Carlo + NFSP agent.

    - Uses MCTS tree search (with game cloning via deepcopy)
    - Depth counts only the agent's own moves
    - At leaf nodes (depth >= max_depth), evaluates using the
      trained NFSP average-policy network
    - Falls back to score-based heuristic if no NFSP agent is set

    Usage:
        # After NFSP training:
        hybrid = HybridMCNFSPAgent(env, player_id=0, nfsp_agent=trained_nfsp)
        action = hybrid.step(state)
    """

    def __init__(self, env, agent_player_id: int, nfsp_agent=None,
                 num_simulations: int = 200, max_depth: int = 2,
                 exploration_constant: float = 1.414):
        """
        Args:
            env: JudgementEnv instance (needed for game cloning and state extraction)
            agent_player_id: Which player this agent controls
            nfsp_agent: A trained NFSPAgent whose average-policy network is used
                        for leaf evaluation. If None, falls back to heuristic.
            num_simulations: Number of MCTS simulations per decision
            max_depth: Max number of the AGENT'S OWN moves to search ahead
            exploration_constant: UCB1 exploration parameter
        """
        self.env = env
        self.agent_player_id = agent_player_id
        self.nfsp_agent = nfsp_agent
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.use_raw = True

    # ── RLCard agent interface ───────────────────────────────────────────

    def step(self, state) -> int:
        """Choose action during training."""
        return self._run_mcts(state)

    def eval_step(self, state):
        """Choose action during evaluation."""
        action = self._run_mcts(state)
        return action, {'agent': 'hybrid_mc_nfsp'}

    # ── MCTS core ────────────────────────────────────────────────────────

    def _run_mcts(self, state) -> int:
        legal_actions = list(state['legal_actions'].keys())
        if len(legal_actions) <= 1:
            return legal_actions[0] if legal_actions else 0

        root = _MCTSNode()

        for _ in range(self.num_simulations):
            game_clone = copy.deepcopy(self.env.game)
            self._simulate(root, game_clone, legal_actions)

        if not root.children:
            return int(np.random.choice(legal_actions))

        # Select action with most visits (robust child selection)
        return max(root.children.keys(), key=lambda a: root.children[a].visits)

    def _simulate(self, root: _MCTSNode, game, legal_actions: List[int]):
        """One full MCTS simulation: select → expand → evaluate → backprop."""
        node = root
        depth = 0        # counts only OUR moves
        path = [node]

        # ── Selection ──
        current_legal = legal_actions
        while (node.children
               and node.is_fully_expanded(current_legal)
               and not node.is_terminal):
            node = node.best_child(self.exploration_constant)
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

        # ── Expansion ──
        if not node.is_terminal and not game.is_over() and depth < self.max_depth:
            current_legal = game._get_legal_actions()
            unexplored = [a for a in current_legal if a not in node.children]
            if unexplored:
                action = int(np.random.choice(unexplored))
                acting_player = game.get_player_id()
                child = _MCTSNode(parent=node, action=action, player_id=acting_player)
                node.children[action] = child
                node = child
                path.append(node)

                if not game.is_over():
                    game.step(action)
                    if acting_player == self.agent_player_id:
                        depth += 1

        # ── Leaf Evaluation ──
        if game.is_over():
            reward = self._score_terminal(game)
        elif depth >= self.max_depth and self.nfsp_agent is not None:
            # NFSP policy evaluation at the frontier
            reward = self._nfsp_evaluate(game)
        else:
            # Fallback: heuristic evaluation
            reward = self._heuristic_evaluate(game)

        # ── Backpropagation ──
        for n in path:
            n.visits += 1
            n.total_reward += reward

    # ── Evaluation functions ─────────────────────────────────────────────

    def _score_terminal(self, game) -> float:
        """Evaluate a finished game by comparing our score to best."""
        scores = [p.score for p in game.players]
        our_score = scores[self.agent_player_id]
        max_abs = max(abs(s) for s in scores) or 1.0
        return our_score / max_abs

    def _nfsp_evaluate(self, game) -> float:
        """
        Use the NFSP average-policy network to evaluate a leaf state.

        Extracts the observation from the cloned game, feeds it through the
        NFSP policy network, and returns an estimated value:
          - Higher probability on the best legal action → higher value
          - Uses the max action probability among legal actions as the value
            estimate, scaled to [-1, 1]
        """
        # Build the raw state from the cloned game
        pid = game.get_player_id()
        raw_state = game.get_state(pid)

        # Extract numeric observation using the env's method
        extracted = self.env._extract_state(raw_state)
        obs = extracted['obs']
        legal_actions = list(extracted['legal_actions'].keys())

        if not legal_actions:
            return 0.0

        # Get action probabilities from the NFSP average policy
        obs_tensor = np.expand_dims(obs, axis=0)
        obs_tensor = torch.from_numpy(obs_tensor).float()

        if hasattr(self.nfsp_agent, 'policy_network'):
            device = next(self.nfsp_agent.policy_network.parameters()).device
            obs_tensor = obs_tensor.to(device)
            with torch.no_grad():
                log_probs = self.nfsp_agent.policy_network(obs_tensor).cpu().numpy()[0]
            probs = np.exp(log_probs)
        else:
            # Fallback if nfsp_agent doesn't have policy_network
            return self._heuristic_evaluate(game)

        # Value estimate: confidence of the best legal action
        # High confidence → agent has a clear best play → good state
        # Low/uniform confidence → uncertain → neutral state
        legal_probs = probs[legal_actions]
        legal_probs = legal_probs / (legal_probs.sum() + 1e-8)

        # Combine: max probability (exploitation signal) + entropy (uncertainty signal)
        max_prob = np.max(legal_probs)
        entropy = -np.sum(legal_probs * np.log(legal_probs + 1e-8))
        max_entropy = np.log(len(legal_actions) + 1e-8)
        normalized_entropy = entropy / (max_entropy + 1e-8)

        # Higher max_prob and lower entropy → better state for us
        # Also factor in our current score alignment with bid
        p = game.players[self.agent_player_id]
        bid_alignment = 0.0
        if p.bid is not None:
            remaining = p.bid - p.tricks_won
            if remaining == 0:
                bid_alignment = 0.3  # met bid, good position
            elif remaining > 0:
                bid_alignment = -0.05 * remaining  # still need tricks
            else:
                bid_alignment = -0.1 * abs(remaining)  # over bid

        # Final value: weighted combination
        value = (max_prob * 0.5) + ((1.0 - normalized_entropy) * 0.2) + bid_alignment
        return np.clip(value, -1.0, 1.0)

    def _heuristic_evaluate(self, game) -> float:
        """Fallback heuristic when NFSP agent is not available."""
        p = game.players[self.agent_player_id]
        reward = p.score / max(abs(p.score), 1) if p.score != 0 else 0
        if p.bid is not None:
            remaining = p.bid - p.tricks_won
            if remaining == 0:
                reward += 0.5
            else:
                reward -= 0.1 * abs(remaining)
        return np.clip(reward, -1.0, 1.0)
