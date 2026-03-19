"""
Judger for Judgement (Oh Hell) card game.
Determines legal actions, trick winners, and scoring.
"""

from typing import List, Optional
from .card import JudgementCard
from .player import JudgementPlayer


class JudgementJudger:
    """Determines legal actions, trick winners, and per-round scores."""

    # --- Action space ---
    # IDs 0-13: bid values (0 through 13)
    # IDs 14-65: play card (card_id 0-51 mapped to action_id 14-65)
    NUM_BID_ACTIONS = 14   # bids 0..13
    NUM_PLAY_ACTIONS = 52  # one per card
    NUM_ACTIONS = NUM_BID_ACTIONS + NUM_PLAY_ACTIONS  # 66

    @staticmethod
    def card_to_action_id(card: JudgementCard) -> int:
        return card.card_id + JudgementJudger.NUM_BID_ACTIONS  # 14 + card_id

    @staticmethod
    def action_id_to_bid(action_id: int) -> int:
        assert 0 <= action_id < JudgementJudger.NUM_BID_ACTIONS
        return action_id

    @staticmethod
    def action_id_to_card_id(action_id: int) -> int:
        assert JudgementJudger.NUM_BID_ACTIONS <= action_id < JudgementJudger.NUM_ACTIONS
        return action_id - JudgementJudger.NUM_BID_ACTIONS

    @staticmethod
    def get_legal_bid_actions(
        player: JudgementPlayer,
        players: List[JudgementPlayer],
        num_cards: int,
        is_dealer: bool,
    ) -> List[int]:
        """
        Return legal bid action IDs.
        - Any player can bid 0..num_cards.
        - Hook rule: dealer (last bidder) cannot make total bids == num_cards.
        """
        all_bids = list(range(num_cards + 1))  # 0..num_cards

        if is_dealer:
            # Sum of bids already made by other players
            other_bids_sum = sum(p.bid for p in players if p.bid is not None)
            forbidden_bid = num_cards - other_bids_sum
            if 0 <= forbidden_bid <= num_cards:
                all_bids = [b for b in all_bids if b != forbidden_bid]

        return all_bids  # these are also the action IDs (0..num_cards)

    @staticmethod
    def get_legal_play_actions(
        player: JudgementPlayer,
        lead_suit: Optional[str],
    ) -> List[int]:
        """
        Return legal play action IDs.
        - Must follow lead suit if possible.
        - If can't follow suit, any card is legal.
        """
        hand = player.hand
        if not hand:
            return []

        if lead_suit is not None:
            suited_cards = [c for c in hand if c.suit == lead_suit]
            if suited_cards:
                return [JudgementJudger.card_to_action_id(c) for c in suited_cards]

        # No lead suit constraint or can't follow suit
        return [JudgementJudger.card_to_action_id(c) for c in hand]

    @staticmethod
    def judge_trick(
        trick_cards: List[tuple],  # [(player_id, JudgementCard), ...]
        trump_suit: Optional[str],
    ) -> int:
        """
        Determine the winner of a trick. Returns the winning player_id.
        - Highest trump wins if any trump played.
        - Otherwise highest card of lead suit wins.
        """
        if not trick_cards:
            raise ValueError("Empty trick")

        lead_suit = trick_cards[0][1].suit
        winner_id = trick_cards[0][0]
        winning_card = trick_cards[0][1]

        for pid, card in trick_cards[1:]:
            if trump_suit and card.suit == trump_suit:
                if winning_card.suit != trump_suit:
                    # First trump beats any non-trump
                    winning_card = card
                    winner_id = pid
                elif card.rank_index > winning_card.rank_index:
                    # Higher trump
                    winning_card = card
                    winner_id = pid
            elif card.suit == lead_suit and winning_card.suit != trump_suit:
                # Same suit as lead, and current winner is not trump
                if card.suit == winning_card.suit:
                    if card.rank_index > winning_card.rank_index:
                        winning_card = card
                        winner_id = pid
                else:
                    # winning_card is off-suit (not trump, not lead) — shouldn't happen
                    # since lead would match if winning_card.suit == lead_suit
                    winning_card = card
                    winner_id = pid

        return winner_id

    @staticmethod
    def compute_round_scores(players: List[JudgementPlayer]) -> List[float]:
        """
        Compute scores for a completed round.
        Exact bid: +10 + tricks_won
        Miss: -|bid - tricks_won|
        """
        scores = []
        for p in players:
            if p.bid is not None and p.bid == p.tricks_won:
                scores.append(10.0 + p.tricks_won)
            elif p.bid is not None:
                scores.append(-abs(p.bid - p.tricks_won))
            else:
                scores.append(0.0)
        return scores

    @staticmethod
    def compute_dense_trick_reward(player: JudgementPlayer, won_trick: bool) -> float:
        """
        Dense per-trick reward based on alignment with bid.
        Call AFTER updating tricks_won.
        """
        if player.bid is None:
            return 0.0

        remaining_needed = player.bid - player.tricks_won
        # After this trick, remaining_needed reflects the gap

        if won_trick:
            if remaining_needed >= 0:
                # Won and still need more or exactly met — on track
                return 1.0
            else:
                # Won but already exceeded bid
                return -0.5
        else:
            if remaining_needed <= 0:
                # Lost and already met/exceeded bid — good to lose
                return 0.5
            else:
                # Lost but still need tricks — bad
                return -0.3
