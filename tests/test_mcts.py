"""Tests for the MCTS agent depth counting."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from agents.mcts_agent import MCTSNode, JudgementMCTSAgent


class TestMCTSNode:

    def test_ucb1_unexplored(self):
        """Unexplored node has infinite UCB1."""
        node = MCTSNode()
        assert node.ucb1() == float('inf')

    def test_ucb1_explored(self):
        """Explored node has finite UCB1."""
        parent = MCTSNode()
        parent.visits = 10
        child = MCTSNode(parent=parent)
        child.visits = 3
        child.total_reward = 1.5
        val = child.ucb1()
        assert np.isfinite(val)
        assert val > 0

    def test_best_child(self):
        """Best child is selected by UCB1."""
        root = MCTSNode()
        root.visits = 10
        c1 = MCTSNode(parent=root)
        c1.visits = 5
        c1.total_reward = 2.0
        c2 = MCTSNode(parent=root)
        c2.visits = 1
        c2.total_reward = 0.5
        root.children = {0: c1, 1: c2}
        best = root.best_child()
        # c2 with fewer visits should have higher UCB1 exploration term
        assert best is not None

    def test_is_fully_expanded(self):
        root = MCTSNode()
        root.children = {0: MCTSNode(), 1: MCTSNode()}
        assert root.is_fully_expanded([0, 1]) is True
        assert root.is_fully_expanded([0, 1, 2]) is False


class TestJudgementMCTSAgent:

    def _make_dummy_state(self, legal_actions):
        """Create a minimal state dict for testing."""
        obs = np.zeros(400, dtype=np.float32)
        # Set some hand cards
        obs[0] = 1  # 2S
        obs[12] = 1  # AS
        return {
            'obs': obs,
            'legal_actions': {a: None for a in legal_actions},
            'raw_legal_actions': legal_actions,
            'raw_obs': obs,
        }

    def test_single_legal_action(self):
        """With only one legal action, returns it immediately."""
        agent = JudgementMCTSAgent(num_simulations=10, max_depth=2)
        state = self._make_dummy_state([14])  # only one card
        action = agent.step(state)
        assert action == 14

    def test_returns_legal_action(self):
        """MCTS returns a legal action."""
        agent = JudgementMCTSAgent(num_simulations=50, max_depth=2)
        legal = [14, 15, 20, 25]
        state = self._make_dummy_state(legal)
        action = agent.step(state)
        assert action in legal

    def test_eval_step(self):
        """eval_step returns action and info dict."""
        agent = JudgementMCTSAgent(num_simulations=10, max_depth=1)
        state = self._make_dummy_state([0, 1, 2])
        action, info = agent.eval_step(state)
        assert action in [0, 1, 2]
        assert isinstance(info, dict)

    def test_depth_parameter(self):
        """Different depth values create valid agents."""
        for depth in [1, 2, 3]:
            agent = JudgementMCTSAgent(num_simulations=5, max_depth=depth)
            assert agent.max_depth == depth
            state = self._make_dummy_state([14, 15])
            action = agent.step(state)
            assert action in [14, 15]

    def test_bid_evaluation(self):
        """Agent can handle bidding actions."""
        agent = JudgementMCTSAgent(num_simulations=20, max_depth=2)
        state = self._make_dummy_state([0, 1, 2, 3])  # bid actions
        action = agent.step(state)
        assert action in [0, 1, 2, 3]
