"""
Dealer for Judgement (Oh Hell) card game.
Handles shuffling, dealing, and trump card reveal.
"""
import numpy as np
from typing import List, Optional
from .card import JudgementCard
from .player import JudgementPlayer


class JudgementDealer:
    """Deals cards and reveals trump."""

    def __init__(self, np_random):
        self.np_random = np.random.default_rng()
        self.deck: List[JudgementCard] = []
        self.trump_card: Optional[JudgementCard] = None

    def new_round(self, players: List[JudgementPlayer], num_cards: int):
        """Shuffle deck, deal num_cards to each player, reveal trump from remainder."""
        self.deck = JudgementCard.get_deck()
        self.np_random.shuffle(self.deck)

        # Deal cards
        for player in players:
            player.reset_for_round()
            for _ in range(num_cards):
                player.hand.append(self.deck.pop())

        # Reveal trump card from remaining deck (if any cards remain)
        if self.deck:
            self.trump_card = self.deck[0]
        else:
            # When all 52 cards are dealt (13 cards × 4 players), no trump
            self.trump_card = None
