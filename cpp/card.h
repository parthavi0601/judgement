#pragma once
#include <vector>
#include <string>
#include <array>
#include <cstdint>

namespace judgement {

struct Card {
    static constexpr std::array<char, 4> SUITS = {'S', 'H', 'D', 'C'};
    static constexpr std::array<const char*, 13> RANKS = {
        "2","3","4","5","6","7","8","9","T","J","Q","K","A"
    };

    int suit_index;   // 0=S, 1=H, 2=D, 3=C
    int rank_index;   // 0=2 .. 12=A
    int card_id;      // suit_index * 13 + rank_index

    Card() : suit_index(0), rank_index(0), card_id(0) {}
    Card(int sid, int rid) : suit_index(sid), rank_index(rid), card_id(sid * 13 + rid) {}

    char suit() const { return SUITS[suit_index]; }
    const char* rank() const { return RANKS[rank_index]; }

    bool operator==(const Card& o) const { return card_id == o.card_id; }
    bool operator!=(const Card& o) const { return card_id != o.card_id; }

    static std::vector<Card> get_deck() {
        std::vector<Card> deck;
        deck.reserve(52);
        for (int s = 0; s < 4; ++s)
            for (int r = 0; r < 13; ++r)
                deck.emplace_back(s, r);
        return deck;
    }

    // Lookup suit index from char
    static int suit_char_to_index(char c) {
        for (int i = 0; i < 4; ++i)
            if (SUITS[i] == c) return i;
        return -1;
    }
};

} // namespace judgement
