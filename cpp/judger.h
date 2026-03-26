#pragma once
#include "card.h"
#include "player.h"
#include <vector>
#include <optional>
#include <cmath>
#include <algorithm>
#include <cassert>

namespace judgement {

class Judger {
public:
    static constexpr int NUM_BID_ACTIONS = 14;   // bids 0..13
    static constexpr int NUM_PLAY_ACTIONS = 52;   // one per card
    static constexpr int NUM_ACTIONS = NUM_BID_ACTIONS + NUM_PLAY_ACTIONS; // 66

    static int card_to_action_id(const Card& card) {
        return card.card_id + NUM_BID_ACTIONS;
    }

    static int action_id_to_bid(int action_id) {
        assert(action_id >= 0 && action_id < NUM_BID_ACTIONS);
        return action_id;
    }

    static int action_id_to_card_id(int action_id) {
        assert(action_id >= NUM_BID_ACTIONS && action_id < NUM_ACTIONS);
        return action_id - NUM_BID_ACTIONS;
    }

    static std::vector<int> get_legal_bid_actions(
        const Player& player,
        const std::vector<Player>& players,
        int num_cards,
        bool is_dealer)
    {
        std::vector<int> bids;
        bids.reserve(num_cards + 1);
        for (int b = 0; b <= num_cards; ++b)
            bids.push_back(b);

        if (is_dealer) {
            int other_bids_sum = 0;
            for (const auto& p : players) {
                if (p.bid.has_value())
                    other_bids_sum += p.bid.value();
            }
            int forbidden = num_cards - other_bids_sum;
            if (forbidden >= 0 && forbidden <= num_cards) {
                bids.erase(std::remove(bids.begin(), bids.end(), forbidden), bids.end());
            }
        }
        return bids;
    }

    static std::vector<int> get_legal_play_actions(
        const Player& player,
        std::optional<char> lead_suit)
    {
        const auto& hand = player.hand;
        if (hand.empty()) return {};

        if (lead_suit.has_value()) {
            int lead_idx = Card::suit_char_to_index(lead_suit.value());
            std::vector<int> suited;
            for (const auto& c : hand) {
                if (c.suit_index == lead_idx)
                    suited.push_back(card_to_action_id(c));
            }
            if (!suited.empty()) return suited;
        }

        std::vector<int> all;
        all.reserve(hand.size());
        for (const auto& c : hand)
            all.push_back(card_to_action_id(c));
        return all;
    }

    // Trick = vector of (player_id, Card)
    using TrickEntry = std::pair<int, Card>;

    static int judge_trick(
        const std::vector<TrickEntry>& trick,
        std::optional<char> trump_suit)
    {
        assert(!trick.empty());

        char lead_suit = trick[0].second.suit();
        int lead_suit_idx = trick[0].second.suit_index;
        int winner_id = trick[0].first;
        Card winning_card = trick[0].second;

        int trump_idx = trump_suit.has_value() ? Card::suit_char_to_index(trump_suit.value()) : -1;

        for (size_t i = 1; i < trick.size(); ++i) {
            int pid = trick[i].first;
            const Card& card = trick[i].second;

            if (trump_idx >= 0 && card.suit_index == trump_idx) {
                if (winning_card.suit_index != trump_idx) {
                    winning_card = card;
                    winner_id = pid;
                } else if (card.rank_index > winning_card.rank_index) {
                    winning_card = card;
                    winner_id = pid;
                }
            } else if (card.suit_index == lead_suit_idx && winning_card.suit_index != trump_idx) {
                if (card.suit_index == winning_card.suit_index) {
                    if (card.rank_index > winning_card.rank_index) {
                        winning_card = card;
                        winner_id = pid;
                    }
                } else {
                    winning_card = card;
                    winner_id = pid;
                }
            }
        }
        return winner_id;
    }

    static std::vector<double> compute_round_scores(const std::vector<Player>& players) {
        std::vector<double> scores;
        scores.reserve(players.size());
        for (const auto& p : players) {
            if (p.bid.has_value() && p.bid.value() == p.tricks_won) {
                scores.push_back(1.0);
            } else if (p.bid.has_value()) {
                scores.push_back(-1.0);
            } else {
                scores.push_back(0.0);
            }
        }
        return scores;
    }

    static double compute_dense_trick_reward(const Player& player, bool won_trick) {
        if (!player.bid.has_value()) return 0.0;

        int remaining_needed = player.bid.value() - player.tricks_won;

        if (won_trick) {
            if (remaining_needed >= 0) return 0.5;
            else return -1.0;
        } else {
            if (remaining_needed == 0) return 0.5;
            else if (remaining_needed > 0) return -0.5;
            else return -0.3;
        }
    }
};

} // namespace judgement
