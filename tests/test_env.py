"""Tests for the Judgement RLCard environment."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from rlcard.envs.registration import register, make
from rlcard.agents.random_agent import RandomAgent


def _register():
    try:
        register(env_id='judgement', entry_point='judgement.env:JudgementEnv')
    except ValueError:
        pass


class TestJudgementEnv:

    def setup_method(self):
        _register()
        self.env = make('judgement', config={
            'seed': 42,
            'allow_step_back': False,
            'game_num_players': 4,
        })

    def test_env_init(self):
        """Environment initializes correctly."""
        assert self.env.num_players == 4
        assert self.env.num_actions == 66

    def test_state_shape(self):
        """Observation shape matches declared state_shape."""
        state, pid = self.env.reset()
        obs = state['obs']
        expected_size = self.env.state_shape[0][1]
        assert obs.shape[0] == expected_size, f'{obs.shape[0]} != {expected_size}'

    def test_legal_actions_in_state(self):
        """State contains legal actions."""
        state, pid = self.env.reset()
        assert 'legal_actions' in state
        assert len(state['legal_actions']) > 0

    def test_step_returns_valid_state(self):
        """Stepping with a legal action returns a valid next state."""
        state, pid = self.env.reset()
        legal = list(state['legal_actions'].keys())
        action = legal[0]
        next_state, next_pid = self.env.step(action)
        assert 'obs' in next_state
        assert 0 <= next_pid < 4

    def test_run_with_random_agents(self):
        """A full game with random agents completes."""
        agents = [RandomAgent(num_actions=self.env.num_actions) for _ in range(4)]
        self.env.set_agents(agents)
        trajectories, payoffs = self.env.run()
        assert len(payoffs) == 4
        # Payoffs should be finite
        for p in payoffs:
            assert np.isfinite(p)

    def test_payoffs_normalized(self):
        """Payoffs are in a reasonable range."""
        agents = [RandomAgent(num_actions=self.env.num_actions) for _ in range(4)]
        self.env.set_agents(agents)
        _, payoffs = self.env.run()
        for p in payoffs:
            assert -2.0 <= p <= 2.0, f'Payoff {p} out of expected range'

    def test_multiple_runs(self):
        """Multiple game runs work without error."""
        agents = [RandomAgent(num_actions=self.env.num_actions) for _ in range(4)]
        self.env.set_agents(agents)
        for _ in range(3):
            trajectories, payoffs = self.env.run()
            assert len(payoffs) == 4
