"""
Round logic for Judgement (Oh Hell) card game.
Handles: bidding phase → trick-taking phase for a single sub-round.
"""

from typing import List, Optional, Tuple
from .card import JudgementCard
from .player import JudgementPlayer
from .dealer import JudgementDealer
from .judger import JudgementJudger


class JudgementRound:
    """A single sub-round: deal, bid, play tricks."""

    def __init__(self, players: List[JudgementPlayer], num_cards: int,
                 dealer_player_id: int, np_random):
        self.players = players
        self.num_players = len(players)
        self.num_cards = num_cards
        self.dealer_player_id = dealer_player_id
        self.np_random = np_random

        # Deal cards and reveal trump
        self.dealer = JudgementDealer(np_random)
        self.dealer.new_round(players, num_cards)
        self.trump_suit: Optional[str] = (
            self.dealer.trump_card.suit if self.dealer.trump_card else None
        )

        # Phase tracking
        self.is_bidding: bool = True
        self.bids_made: int = 0

        # The bidding order starts left of dealer
        self.current_player_id: int = (dealer_player_id + 1) % self.num_players

        # Trick-taking state
        self.current_trick: List[Tuple[int, JudgementCard]] = []  # (player_id, card)
        self.tricks_played: int = 0
        self.lead_player_id: int = 0  # set when trick-taking begins
        self.trick_history: List[List[Tuple[int, JudgementCard]]] = []

        # Dense reward accumulator per player (for current round)
        self.dense_rewards: List[float] = [0.0] * self.num_players

        # History for state extraction
        self.played_cards: List[Tuple[int, JudgementCard]] = []  # all cards played so far

    def get_trump_card(self) -> Optional[JudgementCard]:
        return self.dealer.trump_card

    def is_over(self) -> bool:
        """Round is over when all tricks are played."""
        if self.is_bidding:
            return False
        return self.tricks_played >= self.num_cards

    def get_legal_actions(self) -> List[int]:
        """Get legal action IDs for the current player."""
        player = self.players[self.current_player_id]

        if self.is_bidding:
            is_dealer = (self.current_player_id == self.dealer_player_id)
            return JudgementJudger.get_legal_bid_actions(
                player, self.players, self.num_cards, is_dealer
            )
        else:
            lead_suit = self.current_trick[0][1].suit if self.current_trick else None
            return JudgementJudger.get_legal_play_actions(player, lead_suit)

    def step(self, action_id: int) -> Optional[int]:
        """
        Execute action. Returns dense reward for the acting player (or None during bidding).
        Updates current_player_id for next turn.
        """
        player = self.players[self.current_player_id]

        if self.is_bidding:
            return self._step_bid(action_id, player)
        else:
            return self._step_play(action_id, player)

    def _step_bid(self, action_id: int, player: JudgementPlayer) -> None:
        """Process a bid action."""
        bid_value = JudgementJudger.action_id_to_bid(action_id)
        player.bid = bid_value
        self.bids_made += 1

        if self.bids_made >= self.num_players:
            # All players have bid — transition to trick-taking
            self.is_bidding = False
            # First lead: player left of dealer
            self.lead_player_id = (self.dealer_player_id + 1) % self.num_players
            self.current_player_id = self.lead_player_id
        else:
            self.current_player_id = (self.current_player_id + 1) % self.num_players

        return None  # No dense reward during bidding

    def _step_play(self, action_id: int, player: JudgementPlayer) -> float:
        """Process a play card action. Returns dense reward."""
        card_id = JudgementJudger.action_id_to_card_id(action_id)

        # Find the card in hand
        card = None
        for c in player.hand:
            if c.card_id == card_id:
                card = c
                break

        if card is None:
            raise ValueError(f"Card id {card_id} not in {player}'s hand")

        player.remove_card_from_hand(card)
        self.current_trick.append((player.player_id, card))
        self.played_cards.append((player.player_id, card))

        dense_reward = 0.0

        if len(self.current_trick) >= self.num_players:
            # Trick complete — determine winner
            winner_id = JudgementJudger.judge_trick(self.current_trick, self.trump_suit)
            self.players[winner_id].tricks_won += 1
            self.trick_history.append(self.current_trick.copy())
            self.tricks_played += 1

            # Compute dense rewards for ALL players in this trick
            for pid in range(self.num_players):
                p = self.players[pid]
                won = (pid == winner_id)
                reward = JudgementJudger.compute_dense_trick_reward(p, won)
                self.dense_rewards[pid] += reward

            # Dense reward for the player who just played (completing the trick)
            dense_reward = JudgementJudger.compute_dense_trick_reward(
                player, player.player_id == winner_id
            )

            # Start new trick
            self.current_trick = []
            self.lead_player_id = winner_id
            self.current_player_id = winner_id
        else:
            # Next player in trick
            self.current_player_id = (self.current_player_id + 1) % self.num_players

        return dense_reward
