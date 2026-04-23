"""
Per-game CSV logger for evaluation phases.

Two CSV files are produced:
  1. hybrid_vs_pure_log.csv  — Paired games (same deal played by hybrid then pure)
  2. eval_games_log.csv      — Phase 2 (all-hybrid) and Phase 3 (all-pure) games
"""

import os
import csv


class GameLogger:
    """Logs individual game outcomes to CSV files."""

    def __init__(self, log_dir='./logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.hvp_path = os.path.join(log_dir, 'hybrid_vs_pure_log.csv')
        self._hvp_game_id = 0
        with open(self.hvp_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'game_id', 'seat', 'agent_type', 'player_id',
                'bid', 'tricks_won', 'bid_hit', 'payoff', 'win',
            ])

        self.eval_path = os.path.join(log_dir, 'eval_games_log.csv')
        self._eval_game_id = 0
        with open(self.eval_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'game_id', 'eval_type', 'player_id',
                'bid', 'tricks_won', 'bid_hit', 'payoff', 'win',
            ])



    def log_hvp_game(self, seat, agent_type, players, payoffs, num_players):
        """
        Log one game from Phase 4 (hybrid-vs-pure arena).

        Args:
            seat:        which seat the hybrid agent occupied (0-3)
            agent_type:  'hybrid' or 'pure'
            players:     list of player objects (have .bid, .tricks_won)
            payoffs:     array/list of payoffs per player
            num_players: number of players
        """
        self._hvp_game_id += 1
        rows = []
        for pid in range(num_players):
            p = players[pid]
            bid = p.bid if p.bid is not None else -1
            tricks = p.tricks_won
            bid_hit = 1 if (p.bid is not None and tricks == p.bid) else 0
            payoff = float(payoffs[pid])
            win = 1 if payoff > 0 else 0
            rows.append([
                self._hvp_game_id, seat, agent_type, pid,
                bid, tricks, bid_hit, payoff, win,
            ])

        with open(self.hvp_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)


    def log_eval_game(self, eval_type, players, payoffs, num_players):
        """
        Log one game from Phase 2 or Phase 3.

        Args:
            eval_type:   'hybrid_all' or 'pure_all'
            players:     list of player objects
            payoffs:     array/list of payoffs per player
            num_players: number of players
        """
        self._eval_game_id += 1
        rows = []
        for pid in range(num_players):
            p = players[pid]
            bid = p.bid if p.bid is not None else -1
            tricks = p.tricks_won
            bid_hit = 1 if (p.bid is not None and tricks == p.bid) else 0
            payoff = float(payoffs[pid])
            win = 1 if payoff > 0 else 0
            rows.append([
                self._eval_game_id, eval_type, pid,
                bid, tricks, bid_hit, payoff, win,
            ])

        with open(self.eval_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

