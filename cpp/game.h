#pragma once
#include "card.h"
#include "player.h"
#include "round.h"
#include "judger.h"
#include <vector>
#include <optional>
#include <random>
#include <memory>

namespace judgement {

// Lightweight checkpoint for save/restore (avoids deep copies in MCTS)
struct GameCheckpoint {
    int round_index;
    int dealer_index;
    bool game_over;
    std::vector<double> pending_dense_rewards;

    struct PlayerState {
        std::vector<Card> hand;
        std::optional<int> bid;
        int tricks_won;
        double score;
    };
    std::vector<PlayerState> player_states;

    struct RoundState {
        bool is_bidding;
        int bids_made;
        int current_player_id;
        int lead_player_id;
        std::vector<Judger::TrickEntry> current_trick;
        int tricks_played;
        std::vector<std::vector<Judger::TrickEntry>> trick_history;
        std::vector<Judger::TrickEntry> played_cards;
        std::vector<double> dense_rewards;
    };
    std::optional<RoundState> round_state;
};


class Game {
public:
    int num_players;
    std::mt19937 rng;

    std::vector<int> round_schedule;
    std::vector<Player> players;
    std::unique_ptr<Round> current_round;
    int round_index = 0;
    int dealer_index = 0;
    bool game_over_ = false;
    std::vector<double> pending_dense_rewards;

    explicit Game(int num_players_ = 4, bool /*allow_step_back*/ = false)
        : num_players(num_players_),
          pending_dense_rewards(num_players_, 0.0)
    {
        int max_cards = 52 / num_players;
        round_schedule = {max_cards};
    }

    Game(const Game& other)
        : num_players(other.num_players),
          rng(other.rng),
          round_schedule(other.round_schedule),
          players(other.players),
          round_index(other.round_index),
          dealer_index(other.dealer_index),
          game_over_(other.game_over_),
          pending_dense_rewards(other.pending_dense_rewards)
    {
        if (other.current_round) {
            current_round = std::make_unique<Round>(*other.current_round);
            current_round->players = &players;
        }
    }
    
    Game& operator=(const Game& other) {
        if (this != &other) {
            num_players = other.num_players;
            rng = other.rng;
            round_schedule = other.round_schedule;
            players = other.players;
            round_index = other.round_index;
            dealer_index = other.dealer_index;
            game_over_ = other.game_over_;
            pending_dense_rewards = other.pending_dense_rewards;
            if (other.current_round) {
                current_round = std::make_unique<Round>(*other.current_round);
                current_round->players = &players;
            } else {
                current_round.reset();
            }
        }
        return *this;
    }

    void seed(int s) { rng.seed(s); }

    // Returns (current_player_id). Caller gets state via get_state().
    int init_game() {
        players.clear();
        for (int i = 0; i < num_players; ++i)
            players.emplace_back(i);
        round_index = 0;
        dealer_index = 0;
        game_over_ = false;
        pending_dense_rewards.assign(num_players, 0.0);

        start_new_round();
        return current_round->current_player_id;
    }

    // Returns current_player_id after step
    int step(int action) {
        if (game_over_) return 0;

        current_round->step(action);

        if (current_round->is_over()) {
            finalize_round();
            if (round_index < static_cast<int>(round_schedule.size())) {
                start_new_round();
            } else {
                game_over_ = true;
            }
        }

        if (game_over_) return 0;
        return current_round->current_player_id;
    }

    bool is_over() const { return game_over_; }

    int get_player_id() const {
        if (game_over_) return 0;
        return current_round->current_player_id;
    }

    int get_num_players() const { return num_players; }
    static int get_num_actions() { return Judger::NUM_ACTIONS; }

    std::vector<int> get_legal_actions() const {
        if (game_over_ || !current_round) return {};
        return current_round->get_legal_actions();
    }

    // --- Checkpoint / Restore (lightweight, no deep copy) ---
    GameCheckpoint save_checkpoint() const {
        GameCheckpoint cp;
        cp.round_index = round_index;
        cp.dealer_index = dealer_index;
        cp.game_over = game_over_;
        cp.pending_dense_rewards = pending_dense_rewards;

        cp.player_states.reserve(players.size());
        for (const auto& p : players) {
            cp.player_states.push_back({p.hand, p.bid, p.tricks_won, p.score});
        }

        if (current_round) {
            GameCheckpoint::RoundState rs;
            rs.is_bidding = current_round->is_bidding;
            rs.bids_made = current_round->bids_made;
            rs.current_player_id = current_round->current_player_id;
            rs.lead_player_id = current_round->lead_player_id;
            rs.current_trick = current_round->current_trick;
            rs.tricks_played = current_round->tricks_played;
            rs.trick_history = current_round->trick_history;
            rs.played_cards = current_round->played_cards;
            rs.dense_rewards = current_round->dense_rewards;
            cp.round_state = std::move(rs);
        }
        return cp;
    }

    void restore_checkpoint(const GameCheckpoint& cp) {
        round_index = cp.round_index;
        dealer_index = cp.dealer_index;
        game_over_ = cp.game_over;
        pending_dense_rewards = cp.pending_dense_rewards;

        for (size_t i = 0; i < players.size(); ++i) {
            players[i].hand = cp.player_states[i].hand;
            players[i].bid = cp.player_states[i].bid;
            players[i].tricks_won = cp.player_states[i].tricks_won;
            players[i].score = cp.player_states[i].score;
        }

        if (cp.round_state.has_value() && current_round) {
            const auto& rs = cp.round_state.value();
            current_round->is_bidding = rs.is_bidding;
            current_round->bids_made = rs.bids_made;
            current_round->current_player_id = rs.current_player_id;
            current_round->lead_player_id = rs.lead_player_id;
            current_round->current_trick = rs.current_trick;
            current_round->tricks_played = rs.tricks_played;
            current_round->trick_history = rs.trick_history;
            current_round->played_cards = rs.played_cards;
            current_round->dense_rewards = rs.dense_rewards;
        }
    }

    // --- State extraction (returns struct, bindings convert to Python dict) ---
    struct GameState {
        int player_id;
        int current_player_id;
        std::vector<Card> hand;
        bool is_bidding;
        std::optional<Card> trump_card;
        std::optional<char> trump_suit;
        int num_cards_this_round;
        std::vector<Judger::TrickEntry> current_trick;
        int tricks_played;
        int round_idx;
        int total_rounds;
        std::vector<Judger::TrickEntry> played_cards;
        std::vector<int> legal_actions;
        bool game_is_over;
        std::vector<double> dense_rewards;
        // All players info for state extraction
        std::vector<std::optional<int>> all_bids;
        std::vector<int> all_tricks_won;
        std::vector<double> all_scores;
    };

    GameState get_state(int player_id) const {
        GameState s;
        s.player_id = player_id;
        s.current_player_id = get_player_id();
        s.hand = players[player_id].hand;
        s.is_bidding = current_round ? current_round->is_bidding : false;
        s.trump_card = current_round ? current_round->get_trump_card() : std::nullopt;
        s.trump_suit = current_round ? current_round->trump_suit : std::nullopt;
        s.num_cards_this_round = current_round ? current_round->num_cards : 0;
        s.current_trick = current_round ? current_round->current_trick : std::vector<Judger::TrickEntry>{};
        s.tricks_played = current_round ? current_round->tricks_played : 0;
        s.round_idx = round_index;
        s.total_rounds = static_cast<int>(round_schedule.size());
        s.played_cards = current_round ? current_round->played_cards : std::vector<Judger::TrickEntry>{};
        s.legal_actions = get_legal_actions();
        s.game_is_over = game_over_;
        s.dense_rewards = current_round ? current_round->dense_rewards
                                        : std::vector<double>(num_players, 0.0);
        // All players info
        for (const auto& p : players) {
            s.all_bids.push_back(p.bid);
            s.all_tricks_won.push_back(p.tricks_won);
            s.all_scores.push_back(p.score);
        }
        return s;
    }

private:
    void start_new_round() {
        int num_cards = round_schedule[round_index];
        current_round = std::make_unique<Round>(
            &players, num_cards, dealer_index, rng, round_index);
    }

    void finalize_round() {
        auto scores = Judger::compute_round_scores(players);
        for (int i = 0; i < num_players; ++i) {
            players[i].score += scores[i];
            pending_dense_rewards[i] += scores[i];
        }
        round_index++;
        dealer_index = (dealer_index + 1) % num_players;
    }
};

} // namespace judgement
