#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "card.h"
#include "player.h"
#include "dealer.h"
#include "judger.h"
#include "round.h"
#include "game.h"
#include "mcts_agent.h"
#include "hybrid_agent.h"

namespace py = pybind11;
using namespace judgement;

// Convert GameState to Python dict (matches Python env._extract_state expectations)
py::dict game_state_to_dict(const Game::GameState& gs, const std::vector<Player>& players) {
    py::dict d;
    d["player_id"] = gs.player_id;
    d["current_player_id"] = gs.current_player_id;

    // hand as list of card objects
    py::list hand_list;
    for (const auto& c : gs.hand) {
        py::dict cd;
        cd["card_id"] = c.card_id;
        cd["suit_index"] = c.suit_index;
        cd["rank_index"] = c.rank_index;
        cd["suit"] = std::string(1, c.suit());
        cd["rank"] = std::string(c.rank());
        hand_list.append(cd);
    }
    d["hand"] = hand_list;

    d["is_bidding"] = gs.is_bidding;

    if (gs.trump_card.has_value()) {
        py::dict tc;
        tc["card_id"] = gs.trump_card->card_id;
        tc["suit_index"] = gs.trump_card->suit_index;
        tc["rank_index"] = gs.trump_card->rank_index;
        tc["suit"] = std::string(1, gs.trump_card->suit());
        tc["rank"] = std::string(gs.trump_card->rank());
        d["trump_card"] = tc;
    } else {
        d["trump_card"] = py::none();
    }

    if (gs.trump_suit.has_value())
        d["trump_suit"] = std::string(1, gs.trump_suit.value());
    else
        d["trump_suit"] = py::none();

    d["num_cards_this_round"] = gs.num_cards_this_round;

    py::list trick_list;
    for (const auto& [pid, c] : gs.current_trick) {
        py::tuple t = py::make_tuple(pid, c.card_id, c.suit_index, c.rank_index);
        trick_list.append(t);
    }
    d["current_trick"] = trick_list;

    d["tricks_played"] = gs.tricks_played;
    d["round_index"] = gs.round_idx;
    d["total_rounds"] = gs.total_rounds;

    py::list played_list;
    for (const auto& [pid, c] : gs.played_cards) {
        py::tuple t = py::make_tuple(pid, c.card_id);
        played_list.append(t);
    }
    d["played_cards"] = played_list;

    d["legal_actions"] = gs.legal_actions;
    d["game_over"] = gs.game_is_over;
    d["dense_rewards"] = gs.dense_rewards;

    // All players info
    py::list bids, tricks, scores;
    for (size_t i = 0; i < gs.all_bids.size(); ++i) {
        if (gs.all_bids[i].has_value()) bids.append(gs.all_bids[i].value());
        else bids.append(py::none());
        tricks.append(gs.all_tricks_won[i]);
        scores.append(gs.all_scores[i]);
    }
    d["all_bids"] = bids;
    d["all_tricks_won"] = tricks;
    d["all_scores"] = scores;

    return d;
}


PYBIND11_MODULE(judgement_cpp, m) {
    m.doc() = "C++ accelerated Judgement card game engine with MCTS";

    // --- Card ---
    py::class_<Card>(m, "Card")
        .def(py::init<int, int>(), py::arg("suit_index"), py::arg("rank_index"))
        .def(py::init<>())
        .def_readwrite("suit_index", &Card::suit_index)
        .def_readwrite("rank_index", &Card::rank_index)
        .def_readwrite("card_id", &Card::card_id)
        .def("suit", &Card::suit)
        .def("rank", &Card::rank)
        .def_static("get_deck", &Card::get_deck)
        .def("__eq__", &Card::operator==)
        .def("__repr__", [](const Card& c) {
            return std::string(c.rank()) + c.suit();
        });

    // --- Player ---
    py::class_<Player>(m, "Player")
        .def(py::init<int>(), py::arg("player_id"))
        .def_readwrite("player_id", &Player::player_id)
        .def_readwrite("hand", &Player::hand)
        .def_readwrite("tricks_won", &Player::tricks_won)
        .def_readwrite("score", &Player::score)
        .def_property("bid",
            [](const Player& p) -> py::object {
                return p.bid.has_value() ? py::cast(p.bid.value()) : py::none();
            },
            [](Player& p, py::object val) {
                if (val.is_none()) p.bid.reset();
                else p.bid = val.cast<int>();
            })
        .def("reset_for_round", &Player::reset_for_round);

    // --- Judger (static methods) ---
    py::class_<Judger>(m, "Judger")
        .def_readonly_static("NUM_BID_ACTIONS", &Judger::NUM_BID_ACTIONS)
        .def_readonly_static("NUM_PLAY_ACTIONS", &Judger::NUM_PLAY_ACTIONS)
        .def_readonly_static("NUM_ACTIONS", &Judger::NUM_ACTIONS)
        .def_static("card_to_action_id", &Judger::card_to_action_id)
        .def_static("action_id_to_bid", &Judger::action_id_to_bid)
        .def_static("action_id_to_card_id", &Judger::action_id_to_card_id)
        .def_static("compute_round_scores", &Judger::compute_round_scores)
        .def_static("compute_dense_trick_reward", &Judger::compute_dense_trick_reward);

    // --- Game ---
    py::class_<Game>(m, "Game")
        .def(py::init<int, bool>(), py::arg("num_players") = 4, py::arg("allow_step_back") = false)
        .def("seed", &Game::seed)
        .def("init_game", &Game::init_game)
        .def("step", &Game::step)
        .def("is_over", &Game::is_over)
        .def("get_player_id", &Game::get_player_id)
        .def("get_num_players", &Game::get_num_players)
        .def_static("get_num_actions", &Game::get_num_actions)
        .def("get_legal_actions", &Game::get_legal_actions)
        .def("save_checkpoint", &Game::save_checkpoint)
        .def("restore_checkpoint", &Game::restore_checkpoint)
        .def("get_state_dict", [](const Game& g, int pid) {
            return game_state_to_dict(g.get_state(pid), g.players);
        })
        .def_readwrite("players", &Game::players)
        .def_readonly("num_players", &Game::num_players)
        .def_property_readonly("current_round_ptr", [](const Game& g) -> bool {
            return g.current_round != nullptr;
        })
        .def_property_readonly("pending_dense_rewards", [](const Game& g) {
            return g.pending_dense_rewards;
        })
        .def_property_readonly("round_index", [](const Game& g) { return g.round_index; })
        .def_property_readonly("round_schedule", [](const Game& g) { return g.round_schedule; })
        .def("__copy__", [](const Game& g) { return Game(g); })
        .def("__deepcopy__", [](const Game& g, py::dict) { return Game(g); });

    // --- GameCheckpoint ---
    py::class_<GameCheckpoint>(m, "GameCheckpoint");

    // --- MCTSAgent ---
    py::class_<MCTSAgent>(m, "MCTSAgent")
        .def(py::init<int, int, int, double>(),
             py::arg("player_id"), py::arg("num_simulations") = 200,
             py::arg("max_depth") = 2, py::arg("exploration_constant") = 1.414)
        .def("step", &MCTSAgent::step);

    // --- HybridMCNFSPAgent ---
    py::class_<HybridMCNFSPAgent>(m, "HybridMCNFSPAgent")
        .def(py::init<int, int, int, double>(),
             py::arg("player_id"), py::arg("num_simulations") = 200,
             py::arg("max_depth") = 2, py::arg("exploration_constant") = 1.414)
        .def("step", &HybridMCNFSPAgent::step)
        .def("set_opponent_action_fn", &HybridMCNFSPAgent::set_opponent_action_fn)
        .def("set_leaf_eval_fn", &HybridMCNFSPAgent::set_leaf_eval_fn);
}
