"""
RLCard Environment wrapper for Judgement (Oh Hell) card game.
Uses C++ backend (judgement_cpp) for fast game simulation with fallback to pure Python.
"""

import numpy as np
from collections import OrderedDict

import rlcard.envs
from rlcard.envs import Env

import judgement_cpp
from judgement_cpp import Card as CppCard
from judgement_cpp import Game as CppGame


class _CardLike:
    """Utility to make CppCard behave exactly like the old Python Card object"""
    __slots__ = ('card_id', 'suit_index', 'rank_index', 'suit', 'rank')
    _suits = ['S', 'H', 'D', 'C']
    _ranks = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']

    def __init__(self, card_id=None, suit_index=None, rank_index=None, suit=None, rank=None):
        if card_id is not None:
            self.card_id = card_id
            self.suit_index = card_id // 13
            self.rank_index = card_id % 13
        else:
            self.suit_index = suit_index or 0
            self.rank_index = rank_index or 0
            self.card_id = self.suit_index * 13 + self.rank_index
        self.suit = suit if suit else self._suits[self.suit_index]
        self.rank = rank if rank else self._ranks[self.rank_index]

    def __repr__(self):
        return f'{self.rank}{self.suit}'


class _PlayerLike:
    """Lightweight player proxy for state extraction."""
    __slots__ = ('player_id', 'bid', 'tricks_won', 'score', 'hand')

    def __init__(self, pid, bid=None, tricks_won=0, score=0.0):
        self.player_id = pid
        self.bid = bid
        self.tricks_won = tricks_won
        self.score = score
        self.hand = []


def _dict_to_card(d):
    """Convert C++ card dict to _CardLike."""
    if isinstance(d, dict):
        return _CardLike(card_id=d['card_id'], suit_index=d['suit_index'],
                         rank_index=d['rank_index'], suit=d.get('suit'), rank=d.get('rank'))
    return d


class _RoundProxy:
    def __init__(self, state_dict):
        self.is_bidding = state_dict['is_bidding']
        self.trump_suit = state_dict['trump_suit']
        self.num_cards = state_dict['num_cards_this_round']
        self.tricks_played = state_dict['tricks_played']
        self._trump_card = state_dict['trump_card']
        
        # In Python, current_trick is list of (pid, Card)
        self.current_trick = []
        for t in state_dict['current_trick']:
            pid, cid, sidx, ridx = t
            self.current_trick.append((pid, _CardLike(cid, sidx, ridx)))
            
        # We don't have trick_history in the state_dict directly, but we can reconstruct it from played_cards if needed, or simply return an empty list for the heuristic's determinize. Actually trick_history is used in determinize.
        # It's better to fetch it if we can, but since state_dict doesn't export trick_history, we can just return empty and let determinize use played_cards, or we can export trick_history from C++. Let's just return what we have for now, the MCTS runs in C++ anyway so this is only for the Python fallback and the leaf evaluator which doesn't need trick_history.
        self.trick_history = []
        
    def get_trump_card(self):
        if self._trump_card:
            return _dict_to_card(self._trump_card)
        return None

class _CppGameWrapper:
    """Wraps C++ Game to provide the same interface as Python JudgementGame."""

    def __init__(self, allow_step_back=False, num_players=4):
        self.num_players = num_players
        self._cpp_game = CppGame(num_players, allow_step_back)
        self.players = [_PlayerLike(i) for i in range(num_players)]
        self._round_schedule = [52 // num_players]
        self.pending_dense_rewards = [0.0] * num_players

    @property
    def current_round(self):
        if not self._cpp_game.current_round_ptr:
            return None
        state = self._cpp_game.get_state_dict(0)
        return _RoundProxy(state)

    def init_game(self):
        pid = self._cpp_game.init_game()
        self._sync_players()
        state = self._get_state_dict(pid)
        return state, pid

    def step(self, action):
        pid = self._cpp_game.step(action)
        self._sync_players()
        state = self._get_state_dict(pid)
        return state, pid

    def is_over(self):
        return self._cpp_game.is_over()

    def get_player_id(self):
        return self._cpp_game.get_player_id()

    def get_num_players(self):
        return self.num_players

    @staticmethod
    def get_num_actions():
        return 66  # 14 bids + 52 cards

    def _get_legal_actions(self):
        return self._cpp_game.get_legal_actions()

    def save_checkpoint(self):
        return self._cpp_game.save_checkpoint()

    def restore_checkpoint(self, cp):
        self._cpp_game.restore_checkpoint(cp)
        self._sync_players()

    def _sync_players(self):
        """Sync Python player proxies from C++ state."""
        cpp_players = self._cpp_game.players
        for i in range(self.num_players):
            cp = cpp_players[i]
            self.players[i].bid = cp.bid
            self.players[i].tricks_won = cp.tricks_won
            self.players[i].score = cp.score
            self.players[i].hand = [_CardLike(c.card_id, c.suit_index, c.rank_index)
                                    for c in cp.hand]
        self.pending_dense_rewards = list(self._cpp_game.pending_dense_rewards)

    def get_state(self, player_id):
        return self._get_state_dict(player_id)

    def _get_state_dict(self, player_id):
        """Get state dict from C++ and convert to Python format."""
        d = self._cpp_game.get_state_dict(player_id)

        # Convert raw dicts/tuples to card-like objects
        hand = [_dict_to_card(c) for c in d['hand']]
        trump_card = _dict_to_card(d['trump_card']) if d['trump_card'] is not None else None

        current_trick = []
        for t in d['current_trick']:
            pid, cid, sidx, ridx = t
            current_trick.append((pid, _CardLike(cid, sidx, ridx)))

        played_cards = []
        for t in d['played_cards']:
            pid, cid = t
            played_cards.append((pid, _CardLike(card_id=cid)))

        # Build player proxies
        all_players = []
        for i in range(self.num_players):
            p = _PlayerLike(i, d['all_bids'][i], d['all_tricks_won'][i], d['all_scores'][i])
            if i == player_id:
                p.hand = hand
            all_players.append(p)

        return {
            'player_id': player_id,
            'current_player_id': d['current_player_id'],
            'hand': hand,
            'all_players': all_players,
            'is_bidding': d['is_bidding'],
            'trump_card': trump_card,
            'trump_suit': d['trump_suit'],
            'num_cards_this_round': d['num_cards_this_round'],
            'current_trick': current_trick,
            'tricks_played': d['tricks_played'],
            'round_index': d['round_index'],
            'total_rounds': d['total_rounds'],
            'played_cards': played_cards,
            'legal_actions': d['legal_actions'],
            'game_over': d['game_over'],
            'dense_rewards': d['dense_rewards'],
        }


class JudgementEnv(rlcard.envs.Env):
    """Judgement / Oh Hell RLCard Environment."""

    def __init__(self, config):
        self.name = 'judgement'
        self.allow_step_back = config.get('allow_step_back', False)
        num_players = config.get('game_num_players', 4)

        self.game = _CppGameWrapper(
            allow_step_back=self.allow_step_back,
            num_players=num_players,
        )
        
        self.actions = ['S-0', 'S-1', 'S-2', 'S-3', 'S-4', 'S-5', 'S-6', 'S-7', 'S-8', 'S-9', 'S-T', 'S-J', 'S-Q', 'S-K', 'S-A', 'D-0', 'D-1', 'D-2', 'D-3', 'D-4', 'D-5', 'D-6', 'D-7', 'D-8', 'D-9', 'D-T', 'D-J', 'D-Q', 'D-K', 'D-A', 'C-0', 'C-1', 'C-2', 'C-3', 'C-4', 'C-5', 'C-6', 'C-7', 'C-8', 'C-9', 'C-T', 'C-J', 'C-Q', 'C-K', 'C-A', 'H-0', 'H-1', 'H-2', 'H-3', 'H-4', 'H-5', 'H-6', 'H-7', 'H-8', 'H-9', 'H-T', 'H-J', 'H-Q', 'H-K', 'H-A']
        super().__init__(config=config)
        self.state_shape = [[1, self._get_state_shape_size()] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]
        self._step_dense_rewards = [0.0] * self.num_players

    def _get_state_shape_size(self) -> int:
        size = 0
        size += 52       # hand
        size += 4        # trump suit
        size += 14       # my bid
        size += 14 * (self.num_players - 1)  # other bids
        size += 14 * self.num_players  # tricks_won
        size += 52 * self.num_players  # current trick
        size += 52       # played cards
        size += 14       # num_cards
        size += 1        # is_bidding
        size += self.num_players  # current_player
        size += 1        # round progress
        size += 6        # hand strength
        return size

    def _extract_state(self, state):
        extracted = {}
        player_id = state['player_id']
        players = state['all_players']
        current_player = players[player_id]
        is_bidding = state['is_bidding']
        trump_suit = state['trump_suit']
        num_cards = state['num_cards_this_round']
        current_trick = state['current_trick']
        played_cards = state['played_cards']

        obs_parts = []

        # 1. Hand (52-bit)
        hand_rep = np.zeros(52, dtype=np.float32)
        for card in state['hand']:
            hand_rep[card.card_id] = 1
        obs_parts.append(hand_rep)

        # 2. Trump suit (4-bit)
        trump_rep = np.zeros(4, dtype=np.float32)
        if trump_suit:
            suit_map = {'S': 0, 'H': 1, 'D': 2, 'C': 3}
            trump_rep[suit_map.get(trump_suit, 0)] = 1
        obs_parts.append(trump_rep)

        # 3. My bid (14-bit)
        my_bid_rep = np.zeros(14, dtype=np.float32)
        if current_player.bid is not None:
            my_bid_rep[current_player.bid] = 1
        obs_parts.append(my_bid_rep)

        # 4. Other bids
        for i in range(self.num_players):
            if i != player_id:
                bid_rep = np.zeros(14, dtype=np.float32)
                if players[i].bid is not None:
                    bid_rep[players[i].bid] = 1
                obs_parts.append(bid_rep)

        # 5. Tricks won
        for i in range(self.num_players):
            tricks_rep = np.zeros(14, dtype=np.float32)
            tricks_rep[min(players[i].tricks_won, 13)] = 1
            obs_parts.append(tricks_rep)

        # 6. Current trick
        for i in range(self.num_players):
            trick_slot = np.zeros(52, dtype=np.float32)
            for pid, card in current_trick:
                if pid == i:
                    trick_slot[card.card_id] = 1
            obs_parts.append(trick_slot)

        # 7. Played cards
        played_rep = np.zeros(52, dtype=np.float32)
        for _, card in played_cards:
            played_rep[card.card_id] = 1
        obs_parts.append(played_rep)

        # 8. Num cards
        num_cards_rep = np.zeros(14, dtype=np.float32)
        num_cards_rep[min(num_cards, 13)] = 1
        obs_parts.append(num_cards_rep)

        # 9. Is bidding
        obs_parts.append(np.array([1.0 if is_bidding else 0.0], dtype=np.float32))

        # 10. Current player
        cp_rep = np.zeros(self.num_players, dtype=np.float32)
        cp_rep[state['current_player_id']] = 1
        obs_parts.append(cp_rep)

        # 11. Round progress
        total = state['total_rounds'] if state['total_rounds'] > 0 else 1
        obs_parts.append(np.array([state['round_index'] / total], dtype=np.float32))

        # 12. Hand strength features
        hand = state['hand']
        if hand and num_cards > 0:
            trump_cards = [c for c in hand if c.suit == trump_suit]
            non_trump_cards = [c for c in hand if c.suit != trump_suit]

            trump_count = len(trump_cards) / num_cards
            trump_high = sum(1 for c in trump_cards if c.rank_index >= 10) / num_cards
            trump_avg = sum(c.rank_index for c in trump_cards) / (len(trump_cards) * 12.0) if trump_cards else 0.0

            non_trump_count = len(non_trump_cards) / num_cards
            non_trump_high = sum(1 for c in non_trump_cards if c.rank_index >= 10) / num_cards
            non_trump_avg = sum(c.rank_index for c in non_trump_cards) / (len(non_trump_cards) * 12.0) if non_trump_cards else 0.0
        else:
            trump_count = trump_high = trump_avg = 0.0
            non_trump_count = non_trump_high = non_trump_avg = 0.0

        obs_parts.append(np.array([
            trump_count, trump_high, trump_avg,
            non_trump_count, non_trump_high, non_trump_avg
        ], dtype=np.float32))

        obs = np.concatenate(obs_parts)
        legal_actions = state['legal_actions']
        legal_actions_dict = OrderedDict({a: None for a in legal_actions})

        extracted['obs'] = obs
        extracted['legal_actions'] = legal_actions_dict
        extracted['raw_legal_actions'] = list(legal_actions_dict.keys())
        extracted['raw_obs'] = obs
        extracted['dense_rewards'] = state.get('dense_rewards', [0.0] * self.num_players)
        return extracted

    def get_payoffs(self):
        scores = np.array([p.score for p in self.game.players])
        return scores

    def get_dense_rewards(self):
        if hasattr(self.game, '_cpp_game') and self.game._cpp_game.current_round_ptr:
            # C++ path: dense_rewards are synced
            state = self.game._get_state_dict(0)
            return state['dense_rewards']
        elif hasattr(self.game, 'current_round') and self.game.current_round:
            return list(self.game.current_round.dense_rewards)
        return [0.0] * self.num_players

    def _decode_action(self, action_id):
        return action_id

    def _get_legal_actions(self):
        return self.game._get_legal_actions()

    def get_perfect_information(self):
        return self.game.get_state(self.game.get_player_id())
