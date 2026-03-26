#pragma once
#include "game.h"
#include "judger.h"
#include <vector>
#include <cmath>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <memory>
#include <cassert>

namespace judgement {

struct HybridMCTSNode {
    HybridMCTSNode* parent = nullptr;
    int action = -1;
    int player_id = -1;
    std::unordered_map<int, std::unique_ptr<HybridMCTSNode>> children;
    int visits = 0;
    double total_reward = 0.0;
    bool is_terminal = false;

    double q_value() const { return visits == 0 ? 0.0 : total_reward / visits; }

    double ucb1(double c) const {
        if (visits == 0) return 1e18;
        return q_value() + c * std::sqrt(std::log(parent->visits) / visits);
    }

    HybridMCTSNode* best_child(double c) {
        HybridMCTSNode* best = nullptr;
        double best_val = -1e18;
        for (auto& [a, ch] : children) {
            double v = ch->ucb1(c);
            if (v > best_val) { best_val = v; best = ch.get(); }
        }
        return best;
    }

    bool is_fully_expanded(const std::vector<int>& legal) const {
        for (int a : legal)
            if (children.find(a) == children.end()) return false;
        return true;
    }
};

// Callback types: Python provides these
using OpponentActionFn = std::function<int(int /*player_id*/, const std::vector<int>& /*legal_actions*/,
                                           const Game& /*game*/)>;
using LeafEvalFn = std::function<double(int /*agent_pid*/, const Game& /*game*/)>;


class HybridMCNFSPAgent {
public:
    int agent_player_id;
    int num_simulations;
    int max_depth;
    double exploration_constant;
    std::mt19937 rng;

    // Python callbacks
    OpponentActionFn opponent_action_fn;
    LeafEvalFn leaf_eval_fn;
    bool has_leaf_eval = false;

    HybridMCNFSPAgent(int pid, int sims = 200, int depth = 2, double c = 1.414)
        : agent_player_id(pid), num_simulations(sims), max_depth(depth),
          exploration_constant(c), rng(std::random_device{}()) {}

    void set_opponent_action_fn(OpponentActionFn fn) { opponent_action_fn = fn; }
    void set_leaf_eval_fn(LeafEvalFn fn) { leaf_eval_fn = fn; has_leaf_eval = true; }

    int step(const Game& env_game, const std::vector<int>& legal_actions) {
        if (legal_actions.size() <= 1)
            return legal_actions.empty() ? 0 : legal_actions[0];

        HybridMCTSNode root;
        root.player_id = agent_player_id;

        Game game = env_game; // copy
        auto checkpoint = game.save_checkpoint();

        // Dynamic sim scaling for narrow choice points
        int sims = num_simulations;
        if (legal_actions.size() <= 3) sims *= 4;
        else if (legal_actions.size() <= 5) sims *= 2;

        for (int s = 0; s < sims; ++s) {
            determinize(game);
            simulate(&root, game, legal_actions);
            game.restore_checkpoint(checkpoint);
        }

        if (root.children.empty())
            return legal_actions[rng() % legal_actions.size()];

        int best_act = -1;
        int best_visits = -1;
        for (auto& [a, ch] : root.children) {
            if (ch->visits > best_visits) { best_visits = ch->visits; best_act = a; }
        }
        return best_act;
    }

private:
    int sample_opponent(Game& game, int acting_pid, const std::vector<int>& legal) {
        if (opponent_action_fn)
            return opponent_action_fn(acting_pid, legal, game);
        return legal[rng() % legal.size()];
    }

    void determinize(Game& game) {
        if (!game.current_round) return;

        // 1. Collect known card IDs
        std::unordered_set<int> known;
        for (const auto& c : game.players[agent_player_id].hand)
            known.insert(c.card_id);
        if (game.current_round->get_trump_card())
            known.insert(game.current_round->get_trump_card()->card_id);
        for (const auto& trick : game.current_round->trick_history)
            for (const auto& [pid, c] : trick) known.insert(c.card_id);
        for (const auto& [pid, c] : game.current_round->current_trick)
            known.insert(c.card_id);

        // 2. True unknown cards
        auto full_deck = Card::get_deck();
        std::vector<Card> unknown;
        for (const auto& c : full_deck)
            if (known.find(c.card_id) == known.end())
                unknown.push_back(c);
        std::shuffle(unknown.begin(), unknown.end(), rng);

        // 3. Record opponent hand sizes, clear hands
        std::vector<std::pair<int, int>> opp_sizes; // (pid, size)
        for (int i = 0; i < game.num_players; ++i) {
            if (i != agent_player_id) {
                opp_sizes.emplace_back(i, static_cast<int>(game.players[i].hand.size()));
                game.players[i].hand.clear();
            }
        }

        // 4. Deduce voids from trick history
        std::vector<std::unordered_set<int>> voids(game.num_players); // suit indices
        auto process_trick = [&](const std::vector<Judger::TrickEntry>& trick) {
            if (trick.empty()) return;
            int lead_sidx = trick[0].second.suit_index;
            for (const auto& [pid, c] : trick)
                if (c.suit_index != lead_sidx) voids[pid].insert(lead_sidx);
        };
        for (const auto& t : game.current_round->trick_history) process_trick(t);
        process_trick(game.current_round->current_trick);

        // 5. MRV-sorted deal
        std::sort(opp_sizes.begin(), opp_sizes.end(),
            [&](auto& a, auto& b) { return voids[a.first].size() > voids[b.first].size(); });

        for (auto& [pid, needed] : opp_sizes) {
            while (needed > 0 && !unknown.empty()) {
                int found = -1;
                for (int i = 0; i < static_cast<int>(unknown.size()); ++i) {
                    if (voids[pid].find(unknown[i].suit_index) == voids[pid].end()) {
                        found = i;
                        break;
                    }
                }
                if (found >= 0) {
                    game.players[pid].hand.push_back(unknown[found]);
                    unknown.erase(unknown.begin() + found);
                } else {
                    game.players[pid].hand.push_back(unknown[0]);
                    unknown.erase(unknown.begin());
                }
                needed--;
            }
        }
    }

    void simulate(HybridMCTSNode* root, Game& game, const std::vector<int>& legal_actions) {
        HybridMCTSNode* node = root;
        int depth = 0;
        std::vector<HybridMCTSNode*> path = {node};
        double bid_bonus = 0.0;

        // Selection
        auto current_legal = legal_actions;
        while (!node->children.empty() &&
               node->is_fully_expanded(current_legal) &&
               !node->is_terminal) {
            int acting = game.get_player_id();
            int action;
            if (acting == agent_player_id) {
                node = node->best_child(exploration_constant);
                action = node->action;
                depth++;
            } else {
                action = sample_opponent(game, acting, current_legal);
                auto it = node->children.find(action);
                if (it == node->children.end()) break;
                node = it->second.get();
            }

            if (acting == agent_player_id &&
                game.current_round && game.current_round->is_bidding) {
                bid_bonus += bid_reward_heuristic(game, action);
            }

            game.step(action);
            path.push_back(node);

            if (game.is_over()) { node->is_terminal = true; break; }
            current_legal = game.get_legal_actions();
        }

        // Expansion
        if (!node->is_terminal && !game.is_over() && depth < max_depth) {
            current_legal = game.get_legal_actions();
            int acting = game.get_player_id();
            int action = -1;

            if (acting == agent_player_id) {
                std::vector<int> unexplored;
                for (int a : current_legal)
                    if (node->children.find(a) == node->children.end())
                        unexplored.push_back(a);
                if (!unexplored.empty())
                    action = unexplored[rng() % unexplored.size()];
            } else {
                action = sample_opponent(game, acting, current_legal);
            }

            if (action >= 0 && node->children.find(action) == node->children.end()) {
                auto child = std::make_unique<HybridMCTSNode>();
                child->parent = node;
                child->action = action;
                child->player_id = acting;
                auto* child_ptr = child.get();
                node->children[action] = std::move(child);
                node = child_ptr;
                path.push_back(node);

                if (acting == agent_player_id &&
                    game.current_round && game.current_round->is_bidding) {
                    bid_bonus += bid_reward_heuristic(game, action);
                }

                if (!game.is_over()) {
                    game.step(action);
                    if (acting == agent_player_id) depth++;
                }
            }
        }

        // Fast-forward opponents to reach our next turn
        while (!game.is_over() && game.get_player_id() != agent_player_id) {
            int acting = game.get_player_id();
            auto cl = game.get_legal_actions();
            if (cl.empty()) break;
            int action = sample_opponent(game, acting, cl);
            game.step(action);
        }

        // Evaluate leaf
        double reward;
        if (game.is_over()) {
            reward = score_terminal(game);
        } else if (has_leaf_eval) {
            reward = leaf_eval_fn(agent_player_id, game);
        } else {
            reward = heuristic_evaluate(game);
        }

        reward += 0.15 * bid_bonus;
        reward = std::max(-1.0, std::min(1.0, reward));

        // Backprop
        for (auto* n : path) {
            n->visits++;
            n->total_reward += reward;
        }
    }

    double bid_reward_heuristic(const Game& game, int bid_action) const {
        if (bid_action >= 14) return 0.0;
        const auto& player = game.players[agent_player_id];
        std::optional<char> trump = game.current_round ? game.current_round->trump_suit : std::nullopt;
        double expected = estimate_tricks(player.hand, trump);
        double diff = std::abs(bid_action - expected);
        return std::max(-1.0, 0.5 - 0.3 * diff);
    }

    double estimate_tricks(const std::vector<Card>& hand, std::optional<char> trump) const {
        double expected = 0.0;
        for (const auto& c : hand) {
            bool is_trump = trump.has_value() && c.suit() == trump.value();
            if (is_trump) {
                if (c.rank_index >= 12) expected += 1.0;
                else if (c.rank_index >= 11) expected += 0.8;
                else if (c.rank_index >= 9) expected += 0.5;
                else expected += 0.2;
            } else {
                if (c.rank_index >= 12) expected += 0.5;
                else if (c.rank_index >= 10) expected += 0.2;
            }
        }
        return expected;
    }

    double score_terminal(const Game& game) const {
        auto scores = Judger::compute_round_scores(game.players);
        return scores[agent_player_id];
    }

    double heuristic_evaluate(const Game& game) const {
        const auto& p = game.players[agent_player_id];
        if (!p.bid.has_value()) return 0.0;

        int tricks_remaining = 0;
        if (game.current_round)
            tricks_remaining = game.current_round->num_cards - game.current_round->tricks_played;

        int needed = p.bid.value() - p.tricks_won;

        if (needed == 0) {
            double total = game.current_round ? game.current_round->num_cards : 1;
            double safety = 1.0 - tricks_remaining / std::max(total, 1.0);
            return 0.3 + 0.4 * safety;
        } else if (needed > 0) {
            if (tricks_remaining >= needed) {
                double a = static_cast<double>(needed) / std::max(tricks_remaining, 1);
                return 0.1 * (1.0 - a);
            }
            return -0.5;
        } else {
            return std::max(-1.0, -0.3 * std::abs(needed));
        }
    }
};

} // namespace judgement
