"""Tests for dense reward computation — matches current judger.py values."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from judgement.judger import JudgementJudger
from judgement.player import JudgementPlayer


class TestDenseRewards:

    def test_won_trick_on_track(self):
        """Winning a trick when still needing tricks → +0.5."""
        player = JudgementPlayer(0)
        player.bid = 3
        player.tricks_won = 2  # remaining_needed = 1 >= 0
        reward = JudgementJudger.compute_dense_trick_reward(player, won_trick=True)
        assert reward == 0.5

    def test_won_trick_over_bid(self):
        """Winning a trick when already exceeded bid → -1.0."""
        player = JudgementPlayer(0)
        player.bid = 2
        player.tricks_won = 3  # remaining_needed = -1 < 0
        reward = JudgementJudger.compute_dense_trick_reward(player, won_trick=True)
        assert reward == -1.0

    def test_lost_trick_met_bid(self):
        """Losing a trick when already met bid → +0.5 (good to lose)."""
        player = JudgementPlayer(0)
        player.bid = 2
        player.tricks_won = 2  # remaining_needed = 0
        reward = JudgementJudger.compute_dense_trick_reward(player, won_trick=False)
        assert reward == 0.5

    def test_lost_trick_still_need(self):
        """Losing a trick when still needing tricks → -0.5."""
        player = JudgementPlayer(0)
        player.bid = 3
        player.tricks_won = 1  # remaining_needed = 2 > 0
        reward = JudgementJudger.compute_dense_trick_reward(player, won_trick=False)
        assert reward == -0.5

    def test_no_bid_no_reward(self):
        """No bid set → zero reward."""
        player = JudgementPlayer(0)
        player.bid = None
        reward = JudgementJudger.compute_dense_trick_reward(player, won_trick=True)
        assert reward == 0.0

    def test_won_trick_exactly_met(self):
        """Winning the exact trick that meets bid → remaining_needed = 0, +0.5."""
        player = JudgementPlayer(0)
        player.bid = 2
        player.tricks_won = 2  # remaining_needed = 0 >= 0
        reward = JudgementJudger.compute_dense_trick_reward(player, won_trick=True)
        assert reward == 0.5

    def test_zero_bid_win(self):
        """Bid 0, win a trick → over bid, -1.0."""
        player = JudgementPlayer(0)
        player.bid = 0
        player.tricks_won = 1  # remaining_needed = -1
        reward = JudgementJudger.compute_dense_trick_reward(player, won_trick=True)
        assert reward == -1.0

    def test_zero_bid_lose(self):
        """Bid 0, lose a trick → met bid, +0.5."""
        player = JudgementPlayer(0)
        player.bid = 0
        player.tricks_won = 0  # remaining_needed = 0
        reward = JudgementJudger.compute_dense_trick_reward(player, won_trick=False)
        assert reward == 0.5

    def test_lost_trick_already_over(self):
        """Losing a trick when already over bid → -0.3."""
        player = JudgementPlayer(0)
        player.bid = 1
        player.tricks_won = 3  # remaining_needed = -2 < 0
        reward = JudgementJudger.compute_dense_trick_reward(player, won_trick=False)
        assert reward == -0.3

    def test_round_scoring_exact(self):
        """Round scoring: exact bid → +1.0 (binary)."""
        players = [JudgementPlayer(0), JudgementPlayer(1)]
        players[0].bid = 3
        players[0].tricks_won = 3
        players[1].bid = 2
        players[1].tricks_won = 2
        scores = JudgementJudger.compute_round_scores(players)
        assert scores[0] == 1.0
        assert scores[1] == 1.0

    def test_round_scoring_mixed(self):
        """Round scoring: one exact, one miss."""
        players = [JudgementPlayer(0), JudgementPlayer(1)]
        players[0].bid = 3
        players[0].tricks_won = 3
        players[1].bid = 2
        players[1].tricks_won = 4
        scores = JudgementJudger.compute_round_scores(players)
        assert scores[0] == 1.0
        assert scores[1] == -1.0
