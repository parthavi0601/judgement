#pragma once
#include "card.h"
#include <vector>
#include <optional>
#include <algorithm>

namespace judgement {

struct Player {
    int player_id;
    std::vector<Card> hand;
    std::optional<int> bid;
    int tricks_won = 0;
    double score = 0.0;

    explicit Player(int id) : player_id(id) {}

    void reset_for_round() {
        hand.clear();
        bid.reset();
        tricks_won = 0;
    }

    void remove_card_from_hand(const Card& card) {
        auto it = std::find(hand.begin(), hand.end(), card);
        if (it != hand.end()) hand.erase(it);
    }
};

} // namespace judgement
