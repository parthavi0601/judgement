#pragma once
#include "card.h"
#include "player.h"
#include "dealer.h"
#include "judger.h"
#include <vector>
#include <optional>
#include <random>

namespace judgement {

class Round {
public:
    static constexpr char TRUMP_ORDER[] = {'S', 'D', 'C', 'H'};

    std::vector<Player>* players;
    int num_players;
    int num_cards;
    int dealer_player_id;
    int round_index;

    Dealer dealer;
    std::optional<char> trump_suit;

    bool is_bidding = true;
    int bids_made = 0;
    int current_player_id;

    // Trick state
    std::vector<Judger::TrickEntry> current_trick;
    int tricks_played = 0;
    int lead_player_id = 0;
    std::vector<std::vector<Judger::TrickEntry>> trick_history;

    // Dense reward accumulator
    std::vector<double> dense_rewards;

    // History
    std::vector<Judger::TrickEntry> played_cards;

    Round(std::vector<Player>* players_, int num_cards_, int dealer_pid,
          std::mt19937& rng, int round_idx = 0)
        : players(players_), num_players(static_cast<int>(players_->size())),
          num_cards(num_cards_), dealer_player_id(dealer_pid),
          round_index(round_idx),
          dense_rewards(players_->size(), 0.0)
    {
        dealer.new_round(*players, num_cards, rng);

        trump_suit = TRUMP_ORDER[round_index % 4];
        current_player_id = (dealer_player_id + 1) % num_players;
    }

    std::optional<Card> get_trump_card() const {
        return dealer.trump_card;
    }

    bool is_over() const {
        if (is_bidding) return false;
        return tricks_played >= num_cards;
    }

    std::vector<int> get_legal_actions() const {
        const Player& player = (*players)[current_player_id];

        if (is_bidding) {
            bool is_dlr = (current_player_id == dealer_player_id);
            return Judger::get_legal_bid_actions(player, *players, num_cards, is_dlr);
        } else {
            std::optional<char> lead;
            if (!current_trick.empty())
                lead = current_trick[0].second.suit();
            return Judger::get_legal_play_actions(player, lead);
        }
    }

    // Returns dense reward for acting player (0.0 during bidding)
    double step(int action_id) {
        Player& player = (*players)[current_player_id];

        if (is_bidding) {
            return step_bid(action_id, player);
        } else {
            return step_play(action_id, player);
        }
    }

private:
    double step_bid(int action_id, Player& player) {
        int bid_value = Judger::action_id_to_bid(action_id);
        player.bid = bid_value;
        bids_made++;

        // Bid reward heuristic
        double expected_tricks = 0.0;
        for (const auto& c : player.hand) {
            if (trump_suit.has_value() && c.suit() == trump_suit.value()) {
                if (c.rank_index >= 12) expected_tricks += 1.0;
                else if (c.rank_index >= 11) expected_tricks += 0.8;
                else if (c.rank_index >= 9) expected_tricks += 0.5;
                else expected_tricks += 0.2;
            } else {
                if (c.rank_index >= 12) expected_tricks += 0.5;
                else if (c.rank_index >= 10) expected_tricks += 0.2;
            }
        }

        double diff = std::abs(bid_value - expected_tricks);
        double bid_reward = std::max(-1.0, 0.5 - 0.3 * diff);
        dense_rewards[player.player_id] += bid_reward;

        if (bids_made >= num_players) {
            is_bidding = false;
            lead_player_id = (dealer_player_id + 1) % num_players;
            current_player_id = lead_player_id;
        } else {
            current_player_id = (current_player_id + 1) % num_players;
        }

        return 0.0;
    }

    double step_play(int action_id, Player& player) {
        int card_id = Judger::action_id_to_card_id(action_id);

        // Find card in hand
        Card card;
        bool found = false;
        for (const auto& c : player.hand) {
            if (c.card_id == card_id) {
                card = c;
                found = true;
                break;
            }
        }
        assert(found);

        player.remove_card_from_hand(card);
        current_trick.emplace_back(player.player_id, card);
        played_cards.emplace_back(player.player_id, card);

        double dense_reward = 0.0;

        if (static_cast<int>(current_trick.size()) >= num_players) {
            // Trick complete
            int winner_id = Judger::judge_trick(current_trick, trump_suit);
            (*players)[winner_id].tricks_won++;
            trick_history.push_back(current_trick);
            tricks_played++;

            // Dense rewards for all players
            for (int pid = 0; pid < num_players; ++pid) {
                bool won = (pid == winner_id);
                double r = Judger::compute_dense_trick_reward((*players)[pid], won);
                dense_rewards[pid] += r;
            }

            dense_reward = Judger::compute_dense_trick_reward(
                player, player.player_id == winner_id);

            current_trick.clear();
            lead_player_id = winner_id;
            current_player_id = winner_id;
        } else {
            current_player_id = (current_player_id + 1) % num_players;
        }

        return dense_reward;
    }
};

} // namespace judgement
