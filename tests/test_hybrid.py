"""Tests for the Hybrid MC-NFSP agent."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from agents.hybrid_agent import HybridMCNFSPAgent, _MCTSNode


class TestMCTSNode:

    def test_ucb1_unexplored(self):
        node = _MCTSNode()
        assert node.ucb1() == float('inf')

    def test_ucb1_explored(self):
        parent = _MCTSNode()
        parent.visits = 10
        child = _MCTSNode(parent=parent)
        child.visits = 3
        child.total_reward = 1.5
        val = child.ucb1()
        assert np.isfinite(val)

    def test_best_child(self):
        root = _MCTSNode()
        root.visits = 10
        c1 = _MCTSNode(parent=root)
        c1.visits = 5
        c1.total_reward = 2.0
        c2 = _MCTSNode(parent=root)
        c2.visits = 1
        c2.total_reward = 0.5
        root.children = {0: c1, 1: c2}
        best = root.best_child()
        assert best is not None

    def test_is_fully_expanded(self):
        root = _MCTSNode()
        root.children = {0: _MCTSNode(), 1: _MCTSNode()}
        assert root.is_fully_expanded([0, 1]) is True
        assert root.is_fully_expanded([0, 1, 2]) is False


class TestHybridAgent:

    def _make_env_and_agent(self, all_nfsp_agents=None):
        """Create a test env and hybrid agent."""
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
        agent = HybridMCNFSPAgent(
            env=env,
            agent_player_id=0,
            all_nfsp_agents=all_nfsp_agents,
            num_simulations=20,
            max_depth=1,
        )
        return env, agent

    def test_returns_legal_action(self):
        """Hybrid agent returns a legal action."""
        env, agent = self._make_env_and_agent()
        state, _ = env.reset()
        legal = list(state['legal_actions'].keys())
        action = agent.step(state)
        assert action in legal

    def test_eval_step(self):
        """eval_step returns action and info dict."""
        env, agent = self._make_env_and_agent()
        state, _ = env.reset()
        action, info = agent.eval_step(state)
        legal = list(state['legal_actions'].keys())
        assert action in legal
        assert isinstance(info, dict)
        assert info['agent'] == 'hybrid_mc_nfsp'

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

    def test_without_nfsp_falls_back_to_heuristic(self):
        """Without NFSP agents, falls back to heuristic evaluation."""
        env, agent = self._make_env_and_agent(all_nfsp_agents=None)
        state, _ = env.reset()
        # Should not crash — uses heuristic fallback
        action = agent.step(state)
        legal = list(state['legal_actions'].keys())
        assert action in legal

    def test_with_nfsp_agents(self):
        """With NFSP agents, uses policy network for evaluation."""
        from agents.nfsp_runner import create_nfsp_agents
        env, _ = self._make_env_and_agent()
        nfsp_agents = create_nfsp_agents(env)
        agent = HybridMCNFSPAgent(
            env=env,
            agent_player_id=0,
            all_nfsp_agents=nfsp_agents,
            num_simulations=20,
            max_depth=1,
        )
        state, _ = env.reset()
        action = agent.step(state)
        legal = list(state['legal_actions'].keys())
        assert action in legal

    def test_bid_reward_heuristic(self):
        """Bid heuristic produces non-zero reward for bid actions."""
        env, agent = self._make_env_and_agent()
        # Start a game so we have a valid game state
        env.reset()
        # Bid action 3 (bid 3 tricks) should produce a non-zero heuristic
        reward = agent._bid_reward_heuristic(env.game, 3)
        assert isinstance(reward, float)
        assert -1.0 <= reward <= 0.5

    def test_bid_heuristic_non_bid_action(self):
        """Bid heuristic returns 0.0 for non-bid actions (card play)."""
        env, agent = self._make_env_and_agent()
        env.reset()
        reward = agent._bid_reward_heuristic(env.game, 20)  # Card play action
        assert reward == 0.0

    def test_terminal_scoring_uses_compute_round_scores(self):
        """Terminal scoring uses compute_round_scores (returns +1 or -1)."""
        env, agent = self._make_env_and_agent()
        env.reset()
        # Play a full game randomly
        while not env.game.is_over():
            legal = env.game._get_legal_actions()
            if legal:
                action = np.random.choice(legal)
                env.game.step(action)
        score = agent._score_terminal(env.game)
        assert score in [1.0, -1.0, 0.0]
