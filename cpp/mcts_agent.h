#pragma once
#include "game.h"
#include "judger.h"
#include <vector>
#include <cmath>
#include <random>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <memory>

namespace judgement {

struct MCTSNode {
    MCTSNode* parent = nullptr;
    int action = -1;
    int player_id = -1;
    std::unordered_map<int, std::unique_ptr<MCTSNode>> children;
    int visits = 0;
    double total_reward = 0.0;
    bool is_terminal = false;

    double q_value() const {
        return visits == 0 ? 0.0 : total_reward / visits;
    }

    double ucb1(double c = 1.414) const {
        if (visits == 0) return 1e18;
        return q_value() + c * std::sqrt(std::log(parent->visits) / visits);
    }

    MCTSNode* best_child(double c = 1.414) {
        MCTSNode* best = nullptr;
        double best_val = -1e18;
        for (auto& [act, child] : children) {
            double val = child->ucb1(c);
            if (val > best_val) { best_val = val; best = child.get(); }
        }
        return best;
    }

    bool is_fully_expanded(const std::vector<int>& legal) const {
        for (int a : legal)
            if (children.find(a) == children.end()) return false;
        return true;
    }
};


class MCTSAgent {
public:
    int agent_player_id;
    int num_simulations;
    int max_depth;
    double exploration_constant;
    std::mt19937 rng;

    MCTSAgent(int pid, int sims = 200, int depth = 2, double c = 1.414)
        : agent_player_id(pid), num_simulations(sims), max_depth(depth),
          exploration_constant(c), rng(std::random_device{}()) {}

    int step(const Game& env_game, const std::vector<int>& legal_actions) {
        if (legal_actions.size() <= 1)
            return legal_actions.empty() ? 0 : legal_actions[0];

        MCTSNode root;
        Game game = env_game; // copy
        auto checkpoint = game.save_checkpoint();

        for (int sim = 0; sim < num_simulations; ++sim) {
            one_simulation(&root, game, legal_actions);
            game.restore_checkpoint(checkpoint);
        }

        if (root.children.empty())
            return legal_actions[rng() % legal_actions.size()];

        // Pick action with most visits
        int best_act = -1;
        int best_visits = -1;
        for (auto& [act, child] : root.children) {
            if (child->visits > best_visits) {
                best_visits = child->visits;
                best_act = act;
            }
        }
        return best_act;
    }

private:
    void one_simulation(MCTSNode* root, Game& game, const std::vector<int>& root_legal) {
        MCTSNode* node = root;
        int depth = 0;
        std::vector<MCTSNode*> path = {node};

        // Selection
        auto current_legal = root_legal;
        while (!node->children.empty() &&
               node->is_fully_expanded(current_legal) &&
               !node->is_terminal) {
            node = node->best_child(exploration_constant);
            if (!game.is_over()) {
                int acting = game.get_player_id();
                game.step(node->action);
                if (acting == agent_player_id) depth++;
            }
            path.push_back(node);
            if (game.is_over()) { node->is_terminal = true; break; }
            current_legal = game.get_legal_actions();
        }

        // Expansion
        if (!node->is_terminal && !game.is_over() && depth < max_depth) {
            current_legal = game.get_legal_actions();
            std::vector<int> unexplored;
            for (int a : current_legal)
                if (node->children.find(a) == node->children.end())
                    unexplored.push_back(a);

            if (!unexplored.empty()) {
                int action = unexplored[rng() % unexplored.size()];
                int acting = game.get_player_id();
                auto child = std::make_unique<MCTSNode>();
                child->parent = node;
                child->action = action;
                child->player_id = acting;
                MCTSNode* child_ptr = child.get();
                node->children[action] = std::move(child);
                node = child_ptr;
                path.push_back(node);

                if (!game.is_over()) {
                    game.step(action);
                    if (acting == agent_player_id) depth++;
                }
            }
        }

        // Rollout
        int rollout_depth = depth;
        while (!game.is_over() && rollout_depth < max_depth) {
            auto legal = game.get_legal_actions();
            if (legal.empty()) break;
            int acting = game.get_player_id();
            int action = rollout_policy(game, legal, acting);
            game.step(action);
            if (acting == agent_player_id) rollout_depth++;
        }

        // Evaluate
        double reward = evaluate_state(game);

        // Backprop
        for (auto* n : path) {
            n->visits++;
            n->total_reward += reward;
        }
    }

    int rollout_policy(Game& game, const std::vector<int>& legal, int acting) {
        if (game.current_round && game.current_round->is_bidding) {
            return rollout_bid_policy(game, legal, acting);
        }
        return rollout_play_policy(game, legal, acting);
    }

    int rollout_bid_policy(Game& game, const std::vector<int>& legal, int acting) {
        const auto& hand = game.players[acting].hand;
        std::optional<char> trump = game.current_round ? game.current_round->trump_suit : std::nullopt;
        double expected = estimate_tricks(hand, trump);
        // Pick closest legal bid
        int best = legal[0];
        double best_diff = 1e18;
        for (int b : legal) {
            double d = std::abs(b - expected);
            if (d < best_diff) { best_diff = d; best = b; }
        }
        return best;
    }

    int rollout_play_policy(Game& game, const std::vector<int>& legal, int acting) {
        const auto& player = game.players[acting];
        if (!player.bid.has_value())
            return legal[rng() % legal.size()];

        int need_tricks = player.bid.value() - player.tricks_won;
        std::optional<char> trump = game.current_round ? game.current_round->trump_suit : std::nullopt;

        std::vector<double> scores(legal.size());
        for (size_t i = 0; i < legal.size(); ++i) {
            int cid = legal[i] - 14;
            if (cid < 0 || cid >= 52) { scores[i] = 0.0; continue; }

            int rank = cid % 13;
            int sidx = cid / 13;
            double strength = rank / 12.0;
            bool is_trump = trump.has_value() && Card::SUITS[sidx] == trump.value();

            if (need_tricks > 0) {
                scores[i] = strength + (is_trump ? 0.4 : 0.0);
            } else if (need_tricks == 0) {
                scores[i] = 1.0 - strength - (is_trump ? 0.4 : 0.0);
            } else {
                scores[i] = 1.0 - strength;
            }
        }

        // Softmax with temperature 0.5
        double max_s = *std::max_element(scores.begin(), scores.end());
        std::vector<double> probs(scores.size());
        double sum = 0.0;
        for (size_t i = 0; i < scores.size(); ++i) {
            probs[i] = std::exp((scores[i] - max_s) / 0.5);
            sum += probs[i];
        }
        for (auto& p : probs) p /= sum;

        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        return legal[dist(rng)];
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

    double evaluate_state(const Game& game) const {
        if (game.is_over()) return score_terminal(game);
        return heuristic_evaluate(game);
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
                double achievability = static_cast<double>(needed) / std::max(tricks_remaining, 1);
                return 0.1 * (1.0 - achievability);
            } else {
                return -0.5;
            }
        } else {
            return -0.3 * std::min(std::abs(needed), 3);
        }
    }
};

} // namespace judgement
