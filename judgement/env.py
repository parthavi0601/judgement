"""
RLCard Environment wrapper for Judgement (Oh Hell) card game.
Provides state extraction, action decoding, payoffs, and dense rewards.
"""

import numpy as np
from collections import OrderedDict

from rlcard.envs import Env
from .game import JudgementGame
from .judger import JudgementJudger


class JudgementEnv(Env):
    """Judgement / Oh Hell RLCard Environment."""

    def __init__(self, config):
        self.name = 'judgement'
        num_players = config.get('game_num_players', 4)
        self.game = JudgementGame(
            allow_step_back=config.get('allow_step_back', False),
            num_players=num_players,
        )
        super().__init__(config=config)
        self.state_shape = [[1, self._get_state_shape_size()] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

        # Dense reward tracking
        self._step_dense_rewards = [0.0] * self.num_players

    def _get_state_shape_size(self) -> int:
        """Calculate the observation vector size."""
        size = 0
        size += 52       # hand (one-hot)
        size += 4        # trump suit (one-hot, or zeros if no trump)
        size += 14       # my bid (one-hot over 0-13)
        size += 14 * (self.num_players - 1)  # other players' bids
        size += 14 * self.num_players  # tricks_won per player (one-hot 0-13)
        size += 52 * self.num_players  # cards in current trick (one slot per player)
        size += 52       # cards already played (multi-hot)
        size += 14       # num_cards this round (one-hot 1-13 + padding)
        size += 1        # is_bidding flag
        size += self.num_players  # current_player (one-hot)
        size += 1        # round progress (round_index / total_rounds)
        return size

    def _extract_state(self, state):
        """Convert raw game state to numeric observation."""
        extracted = {}

        player_id = state['player_id']
        players = state['all_players']
        current_player = players[player_id]
        is_bidding = state['is_bidding']
        trump_suit = state['trump_suit']
        num_cards = state['num_cards_this_round']
        current_trick = state['current_trick']
        played_cards = state['played_cards']

        # --- Build observation vector ---
        obs_parts = []

        # 1. Hand (52-bit one-hot)
        hand_rep = np.zeros(52, dtype=np.float32)
        for card in state['hand']:
            hand_rep[card.card_id] = 1
        obs_parts.append(hand_rep)

        # 2. Trump suit (4-bit one-hot)
        trump_rep = np.zeros(4, dtype=np.float32)
        if trump_suit:
            from .card import JudgementCard
            trump_rep[JudgementCard.suits.index(trump_suit)] = 1
        obs_parts.append(trump_rep)

        # 3. My bid (14-bit one-hot, 0-13)
        my_bid_rep = np.zeros(14, dtype=np.float32)
        if current_player.bid is not None:
            my_bid_rep[current_player.bid] = 1
        obs_parts.append(my_bid_rep)

        # 4. Other players' bids (14-bit each)
        for i in range(self.num_players):
            if i != player_id:
                bid_rep = np.zeros(14, dtype=np.float32)
                if players[i].bid is not None:
                    bid_rep[players[i].bid] = 1
                obs_parts.append(bid_rep)

        # 5. Tricks won per player (14-bit one-hot each)
        for i in range(self.num_players):
            tricks_rep = np.zeros(14, dtype=np.float32)
            tricks_rep[min(players[i].tricks_won, 13)] = 1
            obs_parts.append(tricks_rep)

        # 6. Current trick cards (52-bit per player slot)
        for i in range(self.num_players):
            trick_slot = np.zeros(52, dtype=np.float32)
            for pid, card in current_trick:
                if pid == i:
                    trick_slot[card.card_id] = 1
            obs_parts.append(trick_slot)

        # 7. Cards already played (52-bit multi-hot)
        played_rep = np.zeros(52, dtype=np.float32)
        for _, card in played_cards:
            played_rep[card.card_id] = 1
        obs_parts.append(played_rep)

        # 8. Num cards this round (14-bit one-hot)
        num_cards_rep = np.zeros(14, dtype=np.float32)
        num_cards_rep[min(num_cards, 13)] = 1
        obs_parts.append(num_cards_rep)

        # 9. Is bidding flag
        obs_parts.append(np.array([1.0 if is_bidding else 0.0], dtype=np.float32))

        # 10. Current player (one-hot)
        cp_rep = np.zeros(self.num_players, dtype=np.float32)
        cp_rep[state['current_player_id']] = 1
        obs_parts.append(cp_rep)

        # 11. Round progress
        total = state['total_rounds'] if state['total_rounds'] > 0 else 1
        obs_parts.append(np.array([state['round_index'] / total], dtype=np.float32))

        obs = np.concatenate(obs_parts)
        legal_actions = state['legal_actions']
        legal_actions_dict = OrderedDict({a: None for a in legal_actions})

        extracted['obs'] = obs
        extracted['legal_actions'] = legal_actions_dict
        extracted['raw_legal_actions'] = list(legal_actions_dict.keys())
        extracted['raw_obs'] = obs
        return extracted

    def get_payoffs(self):
        """
        Get final payoffs for all players.
        Normalized to roughly [-1, 1] based on score range.
        """
        scores = np.array([p.score for p in self.game.players])
        # Normalize: theoretical max per round is ~23 (10+13), 13 rounds → ~299
        max_possible = 13 * 23
        payoffs = scores / max_possible
        return payoffs

    def get_dense_rewards(self):
        """Get accumulated dense rewards for all players since last call."""
        if self.game.current_round:
            rewards = list(self.game.current_round.dense_rewards)
        else:
            rewards = [0.0] * self.num_players
        return rewards

    def _decode_action(self, action_id):
        """Decode action ID to raw action (the ID itself is used directly by the game)."""
        return action_id

    def _get_legal_actions(self):
        """Get legal actions for current state."""
        return self.game._get_legal_actions()

    def get_perfect_information(self):
        """Get full game state (for debugging / MCTS)."""
        return self.game.get_state(self.game.get_player_id())
