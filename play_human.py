"""
Human-playable Judgement (Oh Hell) card game.

You play as Player 0 against 3 random-action bots.
This lets you test that the game rules, bidding, trick-taking,
and trump order all work correctly.

Usage:
    python play_human.py
    python play_human.py --rounds 3    # play only 3 sub-rounds instead of all 25
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from judgement.game import JudgementGame
from judgement.card import JudgementCard
from judgement.judger import JudgementJudger
from judgement.round import JudgementRound

# ── Pretty-printing helpers ──────────────────────────────────────────────

SUIT_SYMBOLS = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}
SUIT_NAMES   = {'S': 'Spades', 'H': 'Hearts', 'D': 'Diamonds', 'C': 'Clubs'}


def card_str(card: JudgementCard) -> str:
    """Pretty card string like A♠ or 10♥."""
    rank = card.rank if card.rank != 'T' else '10'
    return f"{rank}{SUIT_SYMBOLS[card.suit]}"


def hand_str(cards) -> str:
    """Pretty-print a list of cards sorted by suit then rank."""
    sorted_cards = sorted(cards, key=lambda c: (c.suit_index, c.rank_index))
    return '  '.join(card_str(c) for c in sorted_cards)


def print_divider():
    print('─' * 60)


# ── Main game loop ──────────────────────────────────────────────────────

def play_game(max_rounds=None):
    game = JudgementGame(allow_step_back=False, num_players=4)
    game.np_random = np.random.RandomState(42)

    state, current_pid = game.init_game()

    human_id = 0
    round_count = 0
    total_rounds = len(game._round_schedule)

    print('\n' + '═' * 60)
    print('       ♠ ♥ ♦ ♣  JUDGEMENT (Oh Hell) CARD GAME  ♣ ♦ ♥ ♠')
    print('═' * 60)
    print(f'  You are Player {human_id}.')
    print(f'  Trump order rotates: Spades → Diamonds → Clubs → Hearts')
    print(f'  Total sub-rounds: {total_rounds}')
    print(f'  Card schedule: 13→12→...→1')
    if max_rounds:
        print(f'  (Playing only first {max_rounds} sub-rounds)')
    print('═' * 60)

    prev_round_index = -1

    while not game.is_over():
        ri = game.round_index
        if max_rounds and ri >= max_rounds:
            print(f'\n  Stopped after {max_rounds} sub-rounds.')
            break

        rnd = game.current_round

        # ── Sub-round banner ──
        if ri != prev_round_index:
            prev_round_index = ri
            round_count += 1
            trump_sym = SUIT_SYMBOLS.get(rnd.trump_suit, '?')
            trump_name = SUIT_NAMES.get(rnd.trump_suit, '?')
            trump_card_display = card_str(rnd.dealer.trump_card) if rnd.dealer.trump_card else 'None (all cards dealt)'
            print_divider()
            print(f'  SUB-ROUND {ri+1}/{total_rounds}  |  Cards: {rnd.num_cards}  |  '
                  f'Trump: {trump_sym} {trump_name}  |  '
                  f'Dealer: Player {rnd.dealer_player_id}')
            print(f'  Trump card revealed: {trump_card_display}')
            print_divider()

            # Show your hand
            your_hand = game.players[human_id].hand
            print(f'\n  Your hand: {hand_str(your_hand)}\n')

        legal_actions = state['legal_actions']

        if current_pid == human_id:
            # ── Human turn ──
            if rnd.is_bidding:
                # Bidding
                print(f'  📣 BIDDING — Your turn (Player {human_id})')
                bids_so_far = [(f'P{i}: {game.players[i].bid}' if game.players[i].bid is not None else f'P{i}: ?')
                               for i in range(4)]
                print(f'     Bids so far: {", ".join(bids_so_far)}')
                print(f'     Legal bids: {sorted(legal_actions)}')

                while True:
                    try:
                        bid = int(input('     Enter your bid: '))
                        if bid in legal_actions:
                            action = bid
                            break
                        else:
                            print(f'     ❌ Invalid! Choose from {sorted(legal_actions)}')
                    except (ValueError, EOFError):
                        print(f'     ❌ Enter a number from {sorted(legal_actions)}')
            else:
                # Playing a card
                print(f'  🃏 TRICK {rnd.tricks_played + 1}/{rnd.num_cards} — Your turn (Player {human_id})')
                if rnd.current_trick:
                    trick_display = ', '.join(f'P{pid}: {card_str(c)}' for pid, c in rnd.current_trick)
                    print(f'     Cards on table: {trick_display}')
                else:
                    print(f'     You lead this trick!')

                # Show legal cards
                legal_cards = []
                for action_id in sorted(legal_actions):
                    cid = action_id - JudgementJudger.NUM_BID_ACTIONS
                    for c in game.players[human_id].hand:
                        if c.card_id == cid:
                            legal_cards.append((action_id, c))
                            break

                print(f'     Your hand: {hand_str(game.players[human_id].hand)}')
                print(f'     Legal plays:')
                for i, (aid, c) in enumerate(legal_cards):
                    print(f'       [{i+1}] {card_str(c)}')

                while True:
                    try:
                        choice = int(input(f'     Pick card (1-{len(legal_cards)}): '))
                        if 1 <= choice <= len(legal_cards):
                            action = legal_cards[choice - 1][0]
                            break
                        else:
                            print(f'     ❌ Choose 1 to {len(legal_cards)}')
                    except (ValueError, EOFError):
                        print(f'     ❌ Enter a number.')

            state, current_pid = game.step(action)
            print()

        else:
            # ── Bot turn ──
            action = np.random.choice(list(legal_actions))

            if rnd.is_bidding:
                bid_val = action
                print(f'  🤖 Player {current_pid} bids {bid_val}')
            else:
                cid = action - JudgementJudger.NUM_BID_ACTIONS
                played_card = None
                for c in game.players[current_pid].hand:
                    if c.card_id == cid:
                        played_card = c
                        break
                if played_card:
                    print(f'  🤖 Player {current_pid} plays {card_str(played_card)}')
                else:
                    print(f'  🤖 Player {current_pid} plays action {action}')

            state, current_pid = game.step(action)

            # After a trick completes, show the winner
            if not rnd.is_bidding and len(rnd.current_trick) == 0 and rnd.trick_history:
                last_trick = rnd.trick_history[-1]
                winner_id = JudgementJudger.judge_trick(last_trick, rnd.trump_suit)
                trick_display = ', '.join(f'P{pid}: {card_str(c)}' for pid, c in last_trick)
                print(f'     → Trick won by Player {winner_id}  ({trick_display})')
                print()

        # After round ends, show scoreboard
        if game.current_round and game.current_round.round_index != prev_round_index:
            # Round just changed
            _print_scoreboard(game)

    # ── Final scoreboard ──
    _print_scoreboard(game, final=True)


def _print_scoreboard(game, final=False):
    print_divider()
    label = '🏆 FINAL SCORES' if final else '📊 SCOREBOARD'
    print(f'  {label}')
    for p in game.players:
        bid_info = f'(bid {p.bid}, won {p.tricks_won})' if p.bid is not None else ''
        marker = ' ← YOU' if p.player_id == 0 else ''
        print(f'    Player {p.player_id}: {p.score:+.1f} pts  {bid_info}{marker}')
    print_divider()
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play Judgement interactively')
    parser.add_argument('--rounds', type=int, default=None,
                        help='Number of sub-rounds to play (default: all 25)')
    args = parser.parse_args()
    try:
        play_game(max_rounds=args.rounds)
    except KeyboardInterrupt:
        print('\n\n  Game cancelled. Goodbye!')
