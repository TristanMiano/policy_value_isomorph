// Complete-game expectations using fitted move frequencies, legal transitions and stopping only.
#pragma once

constexpr int visit_forms = 15;
using VisitValues = std::array<double, visit_forms>;

struct VisitPath {
    int length = 0;
    double last = 0, raw_last = 0, discounted_last = 0, early_last = 0, late_last = 0, attenuation = 1;
    std::array<double, 3> signed_sums{}, unsigned_sums{};
    int last_sign = 0;

    void prepend(double frequency, int sign) {
        if (!length) {
            last_sign = sign;
            last = sign * frequency;
            raw_last = frequency;
            discounted_last = last;
            early_last = late_last = last;
        } else {
            discounted_last *= 0.9;
            early_last *= .75;
            late_last *= .97;
            attenuation *= .9;
        }
        constexpr double discount[] = {1., 0.9, 0.5};
        for (int j = 0; j < 3; ++j) {
            signed_sums[j] = sign * frequency + discount[j] * signed_sums[j];
            unsigned_sums[j] = frequency + discount[j] * unsigned_sums[j];
        }
        ++length;
    }
    VisitValues values() const {
        return {last, raw_last, signed_sums[0], signed_sums[1], signed_sums[2],
                unsigned_sums[0], unsigned_sums[1], unsigned_sums[2], discounted_last,
                (unsigned_sums[1] + signed_sums[1]) / 2, last + .1 * signed_sums[1],
                early_last, late_last, last_sign * std::sqrt(raw_last) * attenuation,
                discounted_last * raw_last};
    }
};

class VisitPlanner {
    Engine &game;
    int model, budget;
    U salt;
    bool constant_moves;
    using Picks = std::array<int, 2 * visit_forms>;
    std::array<std::unordered_map<State, Picks, Hash>, 4> menus;
    std::unordered_map<int, std::unordered_map<State, VisitPath, Hash>> tails;
    std::array<VisitValues, 66> constant_paths{};

    // play = 0 greedy/greedy, 1 greedy/uniform, 2 uniform/uniform.
    VisitPath tail(State s, int seat, int play, int replicate) {
        unsigned legal = game.legal(s);
        if (!legal)
            return {};
        auto &cache = tails[(play * 2 + seat) * budget + replicate];
        auto found = cache.find(s);
        if (found != cache.end())
            return found->second;
        bool own = int(s[2]) == seat;
        bool random = play == 2 || (play == 1 && !own);
        int a = random ? game.nth(legal, noise(Hash{}(s) ^ scramble(salt + replicate))) :
                         (constant_moves ? first(legal) : game.best(s, model).move);
        State c = game.next(s, a);
        VisitPath p = tail(c, seat, play, replicate);
        p.prepend(game.score(s, c, a, model), own ? 1 : -1);
        if (cache.size() > size_t(112000 / std::max(8, budget)))
            cache.clear();
        cache.emplace(s, p);
        return p;
    }

    Picks rank(State s, int mode) {
        Picks picks{};
        std::array<double, 2 * visit_forms> best;
        best.fill(-1e100);
        for (unsigned legal = game.legal(s); legal; legal &= legal - 1) {
            int a = first(legal);
            State c = game.next(s, a);
            double initial = game.score(s, c, a, model);
            std::array<double, 2 * visit_forms> scores{};
            int components = mode == 2 ? 2 : 1;
            for (int j = 0; j < components; ++j) {
                int play = mode == 2 ? j : (mode == 3 ? 2 : mode);
                int paths = play == 0 ? 1 : budget;
                double weight = 1. / (paths * components);
                for (int k = 0; k < paths; ++k) {
                    auto p = tail(c, int(s[2]), play, k);
                    p.prepend(initial, 1);
                    auto fitted = p.values();
                    auto &flat = constant_paths.at(p.length);
                    for (int u = 0; u < visit_forms; ++u) {
                        scores[u] += weight * fitted[u];
                        scores[u + visit_forms] += weight * flat[u];
                    }
                }
            }
            for (size_t u = 0; u < scores.size(); ++u) {
                double rounded = std::nearbyint(scores[u] * 1e12);
                if (rounded > best[u]) {
                    best[u] = rounded;
                    picks[u] = a;
                }
            }
        }
        return picks;
    }

  public:
    VisitPlanner(Engine &engine, int id, U seed, int paths, bool flat_continuation)
        : game(engine), model(id), budget(paths), salt(seed), constant_moves(flat_continuation) {
        if ((id != 1 && id != 2) || paths < 1 || engine.models.at(id).table.empty())
            throw std::invalid_argument("install V or Q and provide a positive path budget");
        for (int n = 1; n < int(constant_paths.size()); ++n) {
            VisitPath p;
            for (int j = n - 1; j >= 0; --j)
                p.prepend(.5, j % 2 ? -1 : 1);
            constant_paths[n] = p.values();
        }
    }
    int choose(State s, int mode, int utility, bool flat_score) {
        if (utility < 0 || utility >= visit_forms)
            throw std::invalid_argument("unknown visit utility");
        if (!game.legal(s))
            return -1;
        auto &cache = menus.at(mode);
        auto it = cache.find(s);
        if (it == cache.end()) {
            if (cache.size() > 16000)
                cache.clear();
            it = cache.emplace(s, rank(s, mode)).first;
        }
        return it->second[utility + (flat_score ? visit_forms : 0)];
    }
};

// Actual game evaluation is separate from ranking candidates. Outcomes are read only after play ends.
static std::tuple<int, std::vector<double>, int> visit_match(
    VisitPlanner &planner, Engine &game, State s, int model, int seat, int opponent, U seed,
    int mode, int utility, bool flat_score, bool self) {
    std::vector<double> frequencies;
    int initial = int(s[2]) == seat ? 1 : -1;
    int calls = 0, steps = 0;
    while (unsigned legal = game.legal(s)) {
        int a = self || int(s[2]) == seat ? planner.choose(s, mode, utility, flat_score) :
                game.sample(game.law(s, {double(opponent), 0, 0, -1}, steps, calls), legal,
                            noise(seed + (frequencies.size() + 1) * 397));
        State c = game.next(s, a);
        frequencies.push_back(game.score(s, c, a, model));
        s = c;
    }
    return {game.outcome(s) * (seat ? -1 : 1), frequencies, initial};
}
