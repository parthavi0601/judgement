#pragma once
#include "card.h"
#include "player.h"
#include <vector>
#include <optional>
#include <random>
#include <algorithm>

namespace judgement {

class Dealer {
public:
    std::vector<Card> deck;
    std::optional<Card> trump_card;

    Dealer() = default;

    void new_round(std::vector<Player>& players, int num_cards, std::mt19937& rng) {
        deck = Card::get_deck();
        std::shuffle(deck.begin(), deck.end(), rng);

        for (auto& p : players) {
            p.reset_for_round();
            for (int i = 0; i < num_cards; ++i) {
                p.hand.push_back(deck.back());
                deck.pop_back();
            }
        }

        if (!deck.empty()) {
            trump_card = deck[0];
        } else {
            trump_card.reset();
        }
    }
};

} // namespace judgement
