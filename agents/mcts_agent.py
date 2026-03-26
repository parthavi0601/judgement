"""
MCTS Agent for Judgement (Oh Hell) card game.
Uses C++ backend (judgement_cpp) for fast tree search.
"""

import numpy as np
import pybind11
import judgement_cpp

class JudgementMCTSAgent:
    """
    MCTS agent utilizing the C++ backend for ~50-100x speedup.
    """

    def __init__(self, env, agent_player_id, num_simulations=200, max_depth=2,
                 exploration_constant=1.414):
        self.env = env
        self.agent_player_id = agent_player_id
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.use_raw = True

        self._cpp_agent = judgement_cpp.MCTSAgent(
            agent_player_id, num_simulations, max_depth, exploration_constant)

    def step(self, state):
        return self._run_mcts(state)

    def eval_step(self, state):
        action = self._run_mcts(state)
        return action, {'agent': 'mcts', 'simulations': self.num_simulations}

    def _run_mcts(self, state) -> int:
        legal_actions = list(state['legal_actions'].keys())
        if len(legal_actions) <= 1:
            return legal_actions[0] if legal_actions else 0

        game = self.env.game
        return self._cpp_agent.step(game._cpp_game, legal_actions)

