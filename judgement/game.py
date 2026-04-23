"""
Game orchestrator for Judgement (Oh Hell) card game.
Manages multiple sub-rounds with card counts 1→2→...→13→12→...→1.
Compatible with RLCard's Game interface.
"""

from typing import List, Optional, Tuple

import numpy as np

from .player import JudgementPlayer
from .round import JudgementRound
from .judger import JudgementJudger


class JudgementGame:
    """Top-level game: orchestrates sub-rounds and provides RLCard-compatible API."""

    def __init__(self, allow_step_back=False, num_players=4):
        self.allow_step_back = allow_step_back
        self.num_players = num_players
        self.np_random = np.random.RandomState()

        # Sub-round schedule: just one round of max_cards
        max_cards = 52 // num_players  # 13 for 4 players
        self._round_schedule = [max_cards]

        # State
        self.players: List[JudgementPlayer] = []
        self.current_round: Optional[JudgementRound] = None
        self.round_index: int = 0
        self.dealer_index: int = 0  # rotates each sub-round
        self._game_over: bool = False

        # Dense reward buffer: accumulated per-step rewards for each player
        self.pending_dense_rewards: List[float] = [0.0] * num_players

    def init_game(self):
        """Start a new full game. Returns (state, current_player_id)."""
        self.players = [JudgementPlayer(i) for i in range(self.num_players)]
        self.round_index = 0
        self.dealer_index = 0
        self._game_over = False
        self.pending_dense_rewards = [0.0] * self.num_players

        self._start_new_round()

        current_player_id = self.current_round.current_player_id
        state = self.get_state(current_player_id)
        return state, current_player_id

    def _start_new_round(self):
        """Initialize the next sub-round."""
        num_cards = self._round_schedule[self.round_index]
        self.current_round = JudgementRound(
            players=self.players,
            num_cards=num_cards,
            dealer_player_id=self.dealer_index,
            np_random=self.np_random,
            round_index=self.round_index,
        )

    def step(self, action):
        """
        Take a game step. action is an int action_id.
        Returns (next_state, next_player_id).
        """
        if self._game_over:
            raise ValueError("Game is already over")

        # Execute action in current round
        dense_reward = self.current_round.step(action)

        if dense_reward is not None:
            acting_player = self.current_round.current_player_id

        if self.current_round.is_over():
            self._finalize_round()

            if self.round_index < len(self._round_schedule):
                self._start_new_round()
            else:
                self._game_over = True

        if self._game_over:
            state = self.get_state(0)
            return state, 0

        current_player_id = self.current_round.current_player_id
        state = self.get_state(current_player_id)
        return state, current_player_id

    def _finalize_round(self):
        """Score the completed sub-round and advance to next."""
        scores = JudgementJudger.compute_round_scores(self.players)
        for i, s in enumerate(scores):
            self.players[i].score += s
            self.pending_dense_rewards[i] += s

        self.round_index += 1
        self.dealer_index = (self.dealer_index + 1) % self.num_players

    def is_over(self) -> bool:
        return self._game_over

    def save_checkpoint(self):
        """
        Save a lightweight snapshot of the full game state.
        Cards are immutable so we only copy list references and scalars.
        Used by MCTS to avoid expensive copy.deepcopy per simulation.
        """
        cp = {
            'round_index': self.round_index,
            'dealer_index': self.dealer_index,
            '_game_over': self._game_over,
            'pending_dense_rewards': list(self.pending_dense_rewards),
            'player_states': [
                {
                    'hand': list(p.hand),
                    'bid': p.bid,
                    'tricks_won': p.tricks_won,
                    'score': p.score,
                }
                for p in self.players
            ],
        }
        if self.current_round:
            rnd = self.current_round
            cp['round'] = {
                'is_bidding': rnd.is_bidding,
                'bids_made': rnd.bids_made,
                'current_player_id': rnd.current_player_id,
                'lead_player_id': rnd.lead_player_id,
                'current_trick': list(rnd.current_trick),
                'tricks_played': rnd.tricks_played,
                'trick_history': [list(t) for t in rnd.trick_history],
                'played_cards': list(rnd.played_cards),
                'dense_rewards': list(rnd.dense_rewards),
            }
        return cp

    def restore_checkpoint(self, cp):
        """Restore game state from a saved checkpoint."""
        self.round_index = cp['round_index']
        self.dealer_index = cp['dealer_index']
        self._game_over = cp['_game_over']
        self.pending_dense_rewards = list(cp['pending_dense_rewards'])

        for i, ps in enumerate(cp['player_states']):
            self.players[i].hand = list(ps['hand'])
            self.players[i].bid = ps['bid']
            self.players[i].tricks_won = ps['tricks_won']
            self.players[i].score = ps['score']

        if 'round' in cp and self.current_round:
            rnd = self.current_round
            r = cp['round']
            rnd.is_bidding = r['is_bidding']
            rnd.bids_made = r['bids_made']
            rnd.current_player_id = r['current_player_id']
            rnd.lead_player_id = r['lead_player_id']
            rnd.current_trick = list(r['current_trick'])
            rnd.tricks_played = r['tricks_played']
            rnd.trick_history = [list(t) for t in r['trick_history']]
            rnd.played_cards = list(r['played_cards'])
            rnd.dense_rewards = list(r['dense_rewards'])

    def get_player_id(self) -> int:
        if self._game_over:
            return 0
        return self.current_round.current_player_id

    def get_num_players(self) -> int:
        return self.num_players

    @staticmethod
    def get_num_actions() -> int:
        return JudgementJudger.NUM_ACTIONS  # 66

    def get_state(self, player_id: int) -> dict:
        """
        Get the raw state for a player. The Env wrapper will convert this to
        a numeric observation.
        """
        state = {
            'player_id': player_id,
            'current_player_id': self.get_player_id(),
            'hand': list(self.players[player_id].hand),
            'all_players': self.players,
            'is_bidding': self.current_round.is_bidding if self.current_round else False,
            'trump_card': self.current_round.get_trump_card() if self.current_round else None,
            'trump_suit': self.current_round.trump_suit if self.current_round else None,
            'num_cards_this_round': (
                self.current_round.num_cards if self.current_round else 0
            ),
            'current_trick': (
                list(self.current_round.current_trick) if self.current_round else []
            ),
            'tricks_played': (
                self.current_round.tricks_played if self.current_round else 0
            ),
            'round_index': self.round_index,
            'total_rounds': len(self._round_schedule),
            'played_cards': (
                list(self.current_round.played_cards) if self.current_round else []
            ),
            'legal_actions': self._get_legal_actions(),
            'game_over': self._game_over,
            'dense_rewards': list(self.current_round.dense_rewards) if self.current_round else [0.0] * self.num_players,
        }
        return state

    def _get_legal_actions(self) -> List[int]:
        """Get legal action IDs for current player."""
        if self._game_over or self.current_round is None:
            return []
        return self.current_round.get_legal_actions()
