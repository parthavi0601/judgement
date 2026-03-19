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

        # Sub-round schedule: 13→12→...→1
        max_cards = 52 // num_players  # 13 for 4 players
        self._round_schedule = list(range(max_cards, 0, -1))

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

        # Track dense rewards
        if dense_reward is not None:
            acting_player = self.current_round.current_player_id
            # Note: current_player_id already advanced, so we use the trick context

        # Check if current sub-round is over
        if self.current_round.is_over():
            self._finalize_round()

            if self.round_index < len(self._round_schedule):
                self._start_new_round()
            else:
                self._game_over = True

        if self._game_over:
            # Return state for any player
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
            # Add round-end dense reward
            self.pending_dense_rewards[i] += s

        # Advance round
        self.round_index += 1
        self.dealer_index = (self.dealer_index + 1) % self.num_players

    def is_over(self) -> bool:
        return self._game_over

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
