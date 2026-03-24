"""Tests for Judgement game logic."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from judgement.game import JudgementGame
from judgement.judger import JudgementJudger
from judgement.card import JudgementCard
from judgement.player import JudgementPlayer


class TestJudgementJudger:

    def test_trick_winner_lead_suit(self):
        """Highest card of lead suit wins when no trump."""
        cards = [
            (0, JudgementCard('S', '5')),
            (1, JudgementCard('S', 'K')),
            (2, JudgementCard('S', '3')),
            (3, JudgementCard('H', 'A')),  # off-suit, doesn't win
        ]
        winner = JudgementJudger.judge_trick(cards, trump_suit=None)
        assert winner == 1  # King of Spades

    def test_trick_winner_trump(self):
        """Trump card beats lead suit."""
        cards = [
            (0, JudgementCard('S', 'A')),  # lead: Ace of Spades
            (1, JudgementCard('H', '2')),  # trump: 2 of Hearts
            (2, JudgementCard('S', 'K')),
            (3, JudgementCard('D', 'Q')),
        ]
        winner = JudgementJudger.judge_trick(cards, trump_suit='H')
        assert winner == 1  # Even 2 of trumps beats Ace of lead

    def test_trick_winner_highest_trump(self):
        """Multiple trumps: highest trump wins."""
        cards = [
            (0, JudgementCard('S', 'A')),
            (1, JudgementCard('H', '5')),  # trump
            (2, JudgementCard('H', 'K')),  # trump, higher
            (3, JudgementCard('S', 'Q')),
        ]
        winner = JudgementJudger.judge_trick(cards, trump_suit='H')
        assert winner == 2  # King of Hearts beats 5 of Hearts

    def test_hook_rule(self):
        """Dealer cannot make total bids == num_cards."""
        players = [JudgementPlayer(i) for i in range(4)]
        players[0].bid = 1
        players[1].bid = 1
        players[2].bid = 1
        # num_cards=5, other bids sum=3, forbidden=5-3=2

        legal = JudgementJudger.get_legal_bid_actions(
            players[3], players, num_cards=5, is_dealer=True
        )
        assert 2 not in legal  # forbidden
        assert 0 in legal
        assert 1 in legal
        assert 3 in legal

    def test_no_hook_for_non_dealer(self):
        """Non-dealer has no bid restriction."""
        players = [JudgementPlayer(i) for i in range(4)]
        legal = JudgementJudger.get_legal_bid_actions(
            players[0], players, num_cards=5, is_dealer=False
        )
        assert len(legal) == 6  # 0,1,2,3,4,5

    def test_follow_suit(self):
        """Must follow lead suit if possible."""
        player = JudgementPlayer(0)
        player.hand = [
            JudgementCard('S', 'A'),
            JudgementCard('S', '5'),
            JudgementCard('H', 'K'),
        ]
        actions = JudgementJudger.get_legal_play_actions(player, lead_suit='S')
        # Only spades should be legal
        card_ids = [a - JudgementJudger.NUM_BID_ACTIONS for a in actions]
        for cid in card_ids:
            assert cid // 13 == 0  # suit_index 0 is Spades

    def test_any_card_when_void(self):
        """If void in lead suit, any card is legal."""
        player = JudgementPlayer(0)
        player.hand = [
            JudgementCard('H', 'A'),
            JudgementCard('D', '5'),
        ]
        actions = JudgementJudger.get_legal_play_actions(player, lead_suit='S')
        assert len(actions) == 2  # both cards

    def test_scoring_exact_bid(self):
        """Exact bid: +1.0 (binary scoring)."""
        players = [JudgementPlayer(0)]
        players[0].bid = 3
        players[0].tricks_won = 3
        scores = JudgementJudger.compute_round_scores(players)
        assert scores[0] == 1.0

    def test_scoring_miss_bid(self):
        """Missed bid: -1.0 (binary scoring)."""
        players = [JudgementPlayer(0)]
        players[0].bid = 3
        players[0].tricks_won = 1
        scores = JudgementJudger.compute_round_scores(players)
        assert scores[0] == -1.0

    def test_scoring_zero_bid_exact(self):
        """Bid 0, take 0: +1.0 (binary scoring)."""
        players = [JudgementPlayer(0)]
        players[0].bid = 0
        players[0].tricks_won = 0
        scores = JudgementJudger.compute_round_scores(players)
        assert scores[0] == 1.0


class TestJudgementGame:

    def test_init_game(self):
        """Game initializes without error."""
        game = JudgementGame(num_players=4)
        game.np_random = np.random.RandomState(42)
        state, pid = game.init_game()
        assert 0 <= pid < 4
        assert 'hand' in state
        assert 'legal_actions' in state

    def test_round_schedule(self):
        """Card counts follow 13 only pattern."""
        game = JudgementGame(num_players=4)
        schedule = game._round_schedule
        assert schedule[0] == 13
        assert len(schedule) == 1

    def test_full_game_plays_to_completion(self):
        """A full game with random actions completes without error."""
        game = JudgementGame(num_players=4)
        game.np_random = np.random.RandomState(42)
        state, pid = game.init_game()

        max_steps = 50000  # safety limit
        steps = 0
        while not game.is_over() and steps < max_steps:
            legal = state['legal_actions']
            if not legal:
                break
            action = np.random.choice(legal)
            state, pid = game.step(action)
            steps += 1

        assert game.is_over(), f"Game didn't finish after {steps} steps"

    def test_bidding_phase_transitions(self):
        """After all 4 bids, round transitions to playing phase."""
        game = JudgementGame(num_players=4)
        game.np_random = np.random.RandomState(42)
        state, pid = game.init_game()

        # First round has 1 card, so bids should be 0 or 1
        assert state['is_bidding'] is True

        # Make 4 bids
        for _ in range(4):
            legal = state['legal_actions']
            action = legal[0]
            state, pid = game.step(action)

        # Should be in playing phase now
        assert state['is_bidding'] is False

    def test_num_actions(self):
        assert JudgementGame.get_num_actions() == 66

    def test_num_players(self):
        game = JudgementGame(num_players=4)
        assert game.get_num_players() == 4

    def test_checkpoint_save_restore(self):
        """save_checkpoint → step → restore_checkpoint returns game to saved state."""
        game = JudgementGame(num_players=4)
        game.np_random = np.random.RandomState(42)
        state, pid = game.init_game()

        # Save checkpoint after init
        cp = game.save_checkpoint()
        orig_bids = [p.bid for p in game.players]
        orig_tricks = [p.tricks_won for p in game.players]
        orig_hands = [list(p.hand) for p in game.players]

        # Take enough steps to pass bidding (4 bids) + a few card plays
        for _ in range(8):
            if game.is_over():
                break
            legal = state['legal_actions']
            if not legal:
                break
            action = np.random.choice(legal)
            state, pid = game.step(action)

        # Something should have changed (bids set, cards played)
        bids_changed = any(p.bid != ob for p, ob in zip(game.players, orig_bids))
        hands_changed = any(list(p.hand) != oh for p, oh in zip(game.players, orig_hands))
        assert bids_changed or hands_changed, "State should have changed after steps"

        # Restore checkpoint
        game.restore_checkpoint(cp)

        # Everything should be back to original
        for i, p in enumerate(game.players):
            assert p.bid == orig_bids[i], f"Player {i} bid not restored"
            assert p.tricks_won == orig_tricks[i], f"Player {i} tricks not restored"
            assert list(p.hand) == orig_hands[i], f"Player {i} hand not restored"
