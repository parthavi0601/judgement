"""
Hybrid MC-NFSP Agent for Judgement (Oh Hell) card game.
Uses C++ backend (judgement_cpp) for fast MCTS tree search.
Python callbacks handle NFSP network queries for opponent modeling + leaf evaluation.
"""

import copy
import math
import numpy as np
import torch
from typing import List, Optional, Dict

import numpy as np
import torch
from typing import List, Optional, Dict

import pybind11
import judgement_cpp


class HybridMCNFSPAgent:
    """
    Hybrid MC-NFSP agent. Uses C++ MCTS when available (~50-100x speedup).
    Python callbacks provide NFSP network queries for opponent modeling + leaf evaluation.
    """

    def __init__(self, env, agent_player_id: int, all_nfsp_agents=None,
                 num_simulations: int = 200, max_depth: int = 2,
                 exploration_constant: float = 1.414):
        self.env = env
        self.agent_player_id = agent_player_id
        self.all_nfsp_agents = all_nfsp_agents
        self.nfsp_agent = all_nfsp_agents[agent_player_id] if all_nfsp_agents else None
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.use_raw = True

        self._cpp_agent = judgement_cpp.HybridMCNFSPAgent(
            agent_player_id, num_simulations, max_depth, exploration_constant)
        self._setup_cpp_callbacks()

    def _setup_cpp_callbacks(self):
        """Wire Python NFSP network callbacks into C++ agent."""
        if self._cpp_agent is None:
            return

        # Opponent action callback: use NFSP Average Policy network
        def opponent_action_fn(acting_pid, legal_actions, cpp_game):
            return self._sample_opponent_action_cpp(acting_pid, legal_actions, cpp_game)

        self._cpp_agent.set_opponent_action_fn(opponent_action_fn)

        # Leaf evaluation callback: use NFSP DQN Q-values
        if self.nfsp_agent is not None:
            def leaf_eval_fn(agent_pid, cpp_game):
                return self._nfsp_evaluate_cpp(agent_pid, cpp_game)
            self._cpp_agent.set_leaf_eval_fn(leaf_eval_fn)

    def _sample_opponent_action_cpp(self, acting_pid, legal_actions, cpp_game):
        """Called from C++ to sample opponent action using NFSP policy network."""
        if not self.all_nfsp_agents or acting_pid >= len(self.all_nfsp_agents):
            return int(np.random.choice(legal_actions))

        opp_agent = self.all_nfsp_agents[acting_pid]
        if opp_agent is None or not hasattr(opp_agent, 'policy_network'):
            return int(np.random.choice(legal_actions))

        # Get state from C++ game and extract
        raw_state = cpp_game.get_state_dict(acting_pid)
        # Need to convert to format extract_state expects
        from judgement.env import _dict_to_card, _PlayerLike, _CardLike
        hand = [_dict_to_card(c) for c in raw_state['hand']]
        current_trick = []
        for t in raw_state['current_trick']:
            pid, cid, sidx, ridx = t
            current_trick.append((pid, _CardLike(cid, sidx, ridx)))
        played_cards = []
        for t in raw_state['played_cards']:
            pid, cid = t
            played_cards.append((pid, _CardLike(card_id=cid)))

        all_players = []
        for i in range(cpp_game.num_players):
            p = _PlayerLike(i, raw_state['all_bids'][i], raw_state['all_tricks_won'][i],
                            raw_state['all_scores'][i])
            if i == acting_pid:
                p.hand = hand
            all_players.append(p)

        state = {
            'player_id': acting_pid,
            'current_player_id': raw_state['current_player_id'],
            'hand': hand,
            'all_players': all_players,
            'is_bidding': raw_state['is_bidding'],
            'trump_card': _dict_to_card(raw_state['trump_card']) if raw_state['trump_card'] is not None else None,
            'trump_suit': raw_state['trump_suit'],
            'num_cards_this_round': raw_state['num_cards_this_round'],
            'current_trick': current_trick,
            'tricks_played': raw_state['tricks_played'],
            'round_index': raw_state['round_index'],
            'total_rounds': raw_state['total_rounds'],
            'played_cards': played_cards,
            'legal_actions': legal_actions,
            'game_over': raw_state['game_over'],
            'dense_rewards': raw_state['dense_rewards'],
        }

        extracted = self.env._extract_state(state)
        obs = extracted['obs']

        obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).float()
        device = next(opp_agent.policy_network.parameters()).device
        obs_tensor = obs_tensor.to(device)

        with torch.no_grad():
            log_probs = opp_agent.policy_network(obs_tensor).cpu().numpy()[0]

        probs = np.exp(log_probs)
        legal_probs = probs[legal_actions]
        sum_probs = legal_probs.sum()

        if sum_probs < 1e-8:
            return int(np.random.choice(legal_actions))

        legal_probs /= sum_probs
        return int(np.random.choice(legal_actions, p=legal_probs))

    def _nfsp_evaluate_cpp(self, agent_pid, cpp_game):
        """Called from C++ to evaluate leaf state using NFSP DQN."""
        raw_state = cpp_game.get_state_dict(agent_pid)
        from judgement.env import _dict_to_card, _PlayerLike, _CardLike
        hand = [_dict_to_card(c) for c in raw_state['hand']]
        current_trick = []
        for t in raw_state['current_trick']:
            pid, cid, sidx, ridx = t
            current_trick.append((pid, _CardLike(cid, sidx, ridx)))
        played_cards = []
        for t in raw_state['played_cards']:
            pid, cid = t
            played_cards.append((pid, _CardLike(card_id=cid)))

        all_players = []
        for i in range(cpp_game.num_players):
            p = _PlayerLike(i, raw_state['all_bids'][i], raw_state['all_tricks_won'][i],
                            raw_state['all_scores'][i])
            if i == agent_pid:
                p.hand = hand
            all_players.append(p)

        state = {
            'player_id': agent_pid,
            'current_player_id': raw_state['current_player_id'],
            'hand': hand,
            'all_players': all_players,
            'is_bidding': raw_state['is_bidding'],
            'trump_card': _dict_to_card(raw_state['trump_card']) if raw_state['trump_card'] is not None else None,
            'trump_suit': raw_state['trump_suit'],
            'num_cards_this_round': raw_state['num_cards_this_round'],
            'current_trick': current_trick,
            'tricks_played': raw_state['tricks_played'],
            'round_index': raw_state['round_index'],
            'total_rounds': raw_state['total_rounds'],
            'played_cards': played_cards,
            'legal_actions': raw_state['legal_actions'],
            'game_over': raw_state['game_over'],
            'dense_rewards': raw_state['dense_rewards'],
        }

        extracted = self.env._extract_state(state)
        obs = extracted['obs']
        legal_actions = list(extracted['legal_actions'].keys())

        if not legal_actions:
            return self._heuristic_from_raw(raw_state)

        obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).float()

        q_value = None
        if hasattr(self.nfsp_agent, '_rl_agent') and hasattr(self.nfsp_agent._rl_agent, 'q_estimator'):
            q_estimator = self.nfsp_agent._rl_agent.q_estimator
            device = next(q_estimator.qnet.parameters()).device
            obs_tensor = obs_tensor.to(device)
            with torch.no_grad():
                q_values = q_estimator.qnet(obs_tensor).cpu().numpy()[0]
            legal_q_values = q_values[legal_actions]
            q_value = float(np.max(legal_q_values))
            q_value = np.clip(q_value, -1.0, 1.0)

        if q_value is not None:
            heuristic = self._heuristic_from_raw(raw_state)
            return 0.7 * q_value + 0.3 * heuristic
        return self._heuristic_from_raw(raw_state)

    def _heuristic_from_raw(self, raw_state):
        """Heuristic evaluation from raw C++ game state dict."""
        bid = raw_state['all_bids'][self.agent_player_id]
        tricks_won = raw_state['all_tricks_won'][self.agent_player_id]
        if bid is None:
            return 0.0
        tricks_remaining = raw_state['num_cards_this_round'] - raw_state['tricks_played']
        needed = bid - tricks_won
        if needed == 0:
            total = max(raw_state['num_cards_this_round'], 1)
            safety = 1.0 - tricks_remaining / total
            return 0.3 + 0.4 * safety
        elif needed > 0:
            if tricks_remaining >= needed:
                return 0.1 * (1.0 - needed / max(tricks_remaining, 1))
            return -0.5
        else:
            return float(np.clip(-0.3 * abs(needed), -1.0, 0.0))

    def step(self, state) -> int:
        return self._run_mcts(state)

    def eval_step(self, state):
        action = self._run_mcts(state)
        return action, {'agent': 'hybrid_mc_nfsp'}

    def _run_mcts(self, state) -> int:
        legal_actions = list(state['legal_actions'].keys())
        if len(legal_actions) <= 1:
            return legal_actions[0] if legal_actions else 0

        if self._cpp_agent and hasattr(self.env, 'game') and hasattr(self.env.game, '_cpp_game'):
            # C++ fast path: MCTS runs in C++, callbacks to Python for NN
            return self._cpp_agent.step(self.env.game._cpp_game, legal_actions)
        
        return legal_actions[0] if legal_actions else 0


