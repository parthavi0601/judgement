"""Tests for the MCTS agent with game cloning and bid heuristics."""

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

    def _make_env_and_agent(self, num_simulations=20, max_depth=1):
        """Create a test env and MCTS agent."""
        from rlcard.envs.registration import register, make
        try:
            register(env_id='judgement', entry_point='judgement.env:JudgementEnv')
        except ValueError:
            pass
        env = make('judgement', config={
            'seed': 42,
            'allow_step_back': False,
            'game_num_players': 4,
        })
        agent = JudgementMCTSAgent(
            env=env,
            agent_player_id=0,
            num_simulations=num_simulations,
            max_depth=max_depth,
        )
        return env, agent

    def test_returns_legal_action(self):
        """MCTS returns a legal action."""
        env, agent = self._make_env_and_agent(num_simulations=30, max_depth=1)
        state, _ = env.reset()
        legal = list(state['legal_actions'].keys())
        action = agent.step(state)
        assert action in legal

    def test_eval_step(self):
        """eval_step returns action and info dict."""
        env, agent = self._make_env_and_agent(num_simulations=10, max_depth=1)
        state, _ = env.reset()
        action, info = agent.eval_step(state)
        legal = list(state['legal_actions'].keys())
        assert action in legal
        assert isinstance(info, dict)
        assert info['agent'] == 'mcts'

    def test_single_legal_action(self):
        """With one legal action, returns it immediately."""
        env, agent = self._make_env_and_agent()
        state = {
            'legal_actions': {5: None},
            'raw_legal_actions': [5],
            'obs': np.zeros(454, dtype=np.float32),
            'raw_obs': np.zeros(454, dtype=np.float32),
        }
        action = agent.step(state)
        assert action == 5

    def test_depth_parameter(self):
        """Different depth values create valid agents."""
        for depth in [1, 2, 3]:
            env, agent = self._make_env_and_agent(num_simulations=10, max_depth=depth)
            assert agent.max_depth == depth
            state, _ = env.reset()
            legal = list(state['legal_actions'].keys())
            action = agent.step(state)
            assert action in legal

    def test_bid_heuristic(self):
        """The bid heuristic produces reasonable estimates."""
        env, agent = self._make_env_and_agent()
        # The agent should have an _estimate_tricks method
        assert hasattr(agent, '_estimate_tricks')
        # With an empty hand, estimated tricks should be 0
        assert agent._estimate_tricks([], 'S') == 0.0

    def test_terminal_scoring(self):
        """Terminal scoring uses compute_round_scores (returns +1 or -1)."""
        env, agent = self._make_env_and_agent()
        # Run a full game to get a terminal state
        env.reset()
        while not env.game.is_over():
            legal = env.game._get_legal_actions()
            if legal:
                action = np.random.choice(legal)
                env.game.step(action)
        score = agent._score_terminal(env.game)
        assert score in [1.0, -1.0, 0.0]
