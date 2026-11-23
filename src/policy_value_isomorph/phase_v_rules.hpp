// Finite game rules. The two bitboards use absolute colors, with 0 denoting the first seat.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>
#ifdef __BMI2__
#include <immintrin.h>
#endif
using U = uint64_t;
using State = std::array<U, 3>;
inline U scramble(U x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
struct Hash {
    size_t operator()(const State &s) const { return scramble(s[0] + scramble(s[1] + s[2])); }
};
inline int bits(U v) { return __builtin_popcountll(v); }
inline int first(U v) { return __builtin_ctzll(v); }
inline double noise(U key) { return (scramble(key) >> 11) * 0x1.0p-53; }
inline bool four(U x) {
    for (int d : {1, 7, 6, 8}) {
        U p = x & (x >> d);
        if (p & (p >> (2 * d)))
            return true;
    }
    return false;
}
inline unsigned slide(unsigned v, int d) {
    static const int shift[8] = {1, -1, 4, -4, 5, -5, 3, -3};
    static const unsigned mask[8] = {0x7777, 0xeeee, 0xffff, 0xffff, 0x7777, 0xeeee, 0xeeee, 0x7777};
    int k = shift[d];
    v &= mask[d];
    return (k > 0 ? v << k : v >> -k) & 0xffff;
}
inline unsigned placements(unsigned me, unsigned other) {
    unsigned legal = 0;
    for (int d = 0; d < 8; d++) {
        unsigned p = slide(me, d) & other;
        p |= slide(p, d) & other;
        legal |= slide(p, d);
    }
    return legal & ~(me | other) & 0xffff;
}
inline U pack(U x, U mask) {
#ifdef __BMI2__
    return _pext_u64(x, mask);
#else
    U out = 0, bit = 1;
    while (mask) {
        U low = mask & -mask;
        if (x & low)
            out |= bit;
        mask -= low;
        bit <<= 1;
    }
    return out;
#endif
}
struct Rules {
    bool rev;
    int actions;
    std::array<std::array<U, 8>, 2> patterns{};
    std::unordered_map<State, int, Hash> solved;
    explicit Rules(bool reverse) : rev(reverse), actions(reverse ? 17 : 7) {
        for (int g = 0; g < 8; g++)
            for (int j = 0; j < 4; j++) {
                int a = rev ? (g < 4 ? 4 * g + j : 4 * j + g - 4) : ((g < 7) ? 7 * g + j : 7 * j + 4);
                int b = rev ? (4 * j + (g + j * (g >= 4)) % 4) : 7 * ((g + j) % 7) + (g + 2 * j) % 6;
                patterns[0][g] |= 1ULL << a;
                patterns[1][g] |= 1ULL << b;
            }
    }
    State start() const {
        return rev ? State{(1 << 6) | (1 << 9), (1 << 5) | (1 << 10), 0} : State{0, 0, 0};
    }
    unsigned legal(State s) const {
        if (rev) {
            auto p = placements(s[s[2]], s[1 - s[2]]);
            return p ? p : (placements(s[1 - s[2]], s[s[2]]) ? 1u << 16 : 0);
        }
        if (four(s[1 - s[2]]))
            return 0;
        return 127u ^
               unsigned(pack(s[0] | s[1], (1ULL << 5) | (1ULL << 12) | (1ULL << 19) | (1ULL << 26) |
                                              (1ULL << 33) | (1ULL << 40) | (1ULL << 47)));
    }
    State next(State s, int a) const {
        U me = s[2], other = 1 - me;
        if (!rev) {
            U base = 1ULL << (7 * a);
            s[me] |= ((s[0] | s[1]) + base) & (63 * base) & ~(s[0] | s[1]);
        } else if (a != 16) {
            unsigned token = 1u << a, flips = 0;
            for (int d = 0; d < 8; d++) {
                unsigned p = slide(token, d) & s[other], run = p;
                while (p) {
                    unsigned q = slide(p, d);
                    if (q & s[me]) {
                        flips |= run;
                        break;
                    }
                    p = q & s[other];
                    run |= p;
                }
            }
            s[me] |= token | flips;
            s[other] &= ~U(flips);
        }
        s[2] = other;
        return s;
    }
    int outcome(State s) const {
        if (!rev)
            return four(s[0]) ? 1 : four(s[1]) ? -1 : 0;
        return (bits(s[0]) > bits(s[1])) - (bits(s[0]) < bits(s[1]));
    }
    int solve(State s) {
        auto it = solved.find(s);
        if (it != solved.end())
            return it->second;
        unsigned mask = legal(s);
        int value = outcome(s);
        if (mask) {
            value = -2;
            for (; mask; mask &= mask - 1)
                value = std::max(value, -solve(next(s, first(mask))));
        } else if (s[2])
            value = -value;
        solved.emplace(s, value);
        return value;
    }
    int priority(int a) const {
        if (!rev)
            return -abs(a - 3);
        if (a == 16)
            return -4;
        return (a / 4 == 0 || a / 4 == 3) + (a % 4 == 0 || a % 4 == 3);
    }
    int teacher(State s, int kind, const std::vector<int> &order) {
        unsigned mask = legal(s);
        int best = -999, chosen = -1;
        for (int a : order) {
            if (!(mask & (1u << a)))
                continue;
            int score = 0;
            if (kind == 1 && rev)
                score = -solve(next(s, a));
            if (kind == 1 && !rev) {
                State c = next(s, a), o = s;
                o[2] ^= 1;
                score = 2 * four(c[s[2]]) + four(next(o, a)[o[2]]);
            }
            if (score > best) {
                best = score;
                chosen = a;
            }
        }
        return chosen;
    }
    U canonical(State s) const {
        State smallest = s;
        for (int t = 1; t < (rev ? 8 : 2); t++) {
            State q{0, 0, s[2]};
            for (int seat = 0; seat < 2; seat++)
                for (U z = s[seat]; z; z &= z - 1) {
                    int a = first(z), v;
                    if (!rev)
                        v = 7 * (6 - a / 7) + a % 7;
                    else {
                        int r = a / 4, c = a % 4;
                        if (t >= 4)
                            c = 3 - c;
                        for (int j = 0; j < (t % 4); j++) {
                            int tmp = r;
                            r = c;
                            c = 3 - tmp;
                        }
                        v = 4 * r + c;
                    }
                    q[seat] |= 1ULL << v;
                }
            smallest = std::min(smallest, q);
        }
        return Hash{}(smallest);
    }
    static constexpr int groups = 12, stride = 8192;
    using Codes = std::array<int, groups>;
    Codes codes(State s, State c, int a, bool q, int layout) const {
        Codes x{};
        for (int j = 0; j < 8; j++) {
            U m = patterns[layout][j];
            x[j] = j * stride + int(pack(c[0], m) | (pack(c[1], m) << 4) | (c[2] << 8));
        }
        int own = bits(c[0]), opp = bits(c[1]);
        int special = rev ? int(placements(c[c[2]], c[1 - c[2]]) == 0) : int(four(c[1 - c[2]]));
        x[8] = 8 * stride + ((own + opp) * 2 + c[2]) * 4 + special;
        x[9] = 9 * stride + (rev ? (own * 17 + opp) * 2 + c[2]
                                 : int(pack(c[0] | c[1], patterns[1][7])) + 256 * c[2]);
        if (q && !rev) {
            State other = s;
            other[2] ^= 1;
            int block = four(next(other, a)[other[2]]);
            x[9] = 9 * stride + int(s[2]) * 64 + a * 4 + 2 * special + block;
        }
        for (int j = 10; j < 12; j++) {
            U m = patterns[layout][(j + a * (q ? 1 : 0)) % 8];
            State ref = q ? s : c;
            int pattern = int(pack(ref[0], m) | (pack(ref[1], m) << 4));
            x[j] = j * stride +
                   (q ? (a * 2 + int(s[2])) * 192 + (pattern % 192) : pattern + 256 * int(c[2]));
        }
        return x;
    }
};
