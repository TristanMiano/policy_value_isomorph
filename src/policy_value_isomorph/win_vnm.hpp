// Win-focused whole-game selection for the Phase V engine.
#pragma once

enum class Finish { Direct, Guarded, Mixture, GuardedMixture, Original };
enum class Prize { Last, Sum, Discounted, Win, Points, QuarterDraw, Hybrid, Direct };

class WinPlanner {
    Engine &game;
    int model;
    U salt;
    int repetitions;
    std::unordered_map<State, int, Hash> guards;
    std::unordered_map<int, std::unordered_map<State, Tail, Hash>> tails;
    using Menu = std::array<Tail, 17>;
    std::array<std::unordered_map<State, Menu, Hash>, 5> menus;

    int guarded(State s) {
        auto it = guards.find(s);
        if (it != guards.end())
            return it->second;
        std::array<double, 6> best{};
        int chosen = -1;
        for (unsigned m = game.legal(s); m; m &= m - 1) {
            int a = first(m);
            State c = game.next(s, a);
            unsigned replies = game.legal(c);
            int now = replies ? 0 : game.outcome(c) * (s[2] ? -1 : 1);
            int threats = 0;
            for (unsigned r = replies; r; r &= r - 1) {
                auto h = game.next(c, first(r));
                if (!game.legal(h) && game.outcome(h) * (s[2] ? 1 : -1) > 0)
                    ++threats;
            }
            unsigned places = replies & 0xffff, corners = 0x9009;
            std::array<double, 6> rank{double(now),
                                       double(-threats),
                                       double(game.rev && a < 16 && ((1u << a) & corners)),
                                       double(game.rev ? -bits(places & corners) : 0),
                                       double(game.rev ? -bits(places) : 0),
                                       game.score(s, c, a, model)};
            if (chosen < 0 || rank > best) {
                best = rank;
                chosen = a;
            }
        }
        if (guards.size() > 65000)
            guards.clear();
        guards.emplace(s, chosen);
        return chosen;
    }
    int play(State s, int actor, int replicate) {
        if (actor == 0)
            return game.best(s, model).move;
        if (actor == 1)
            return guarded(s);
        if (actor == 4)
            return game.best(s, 0).move;
        unsigned moves = game.legal(s);
        if (actor == 3)
            return game.nth(moves, noise(Hash{}(s) ^ scramble(salt + replicate)));
        int best = -999, chosen = -1;
        for (; moves; moves &= moves - 1) {
            int a = first(moves), rank = game.priority(a);
            if (rank > best) {
                best = rank;
                chosen = a;
            }
        }
        return chosen;
    }
    Tail finish(State s, int own, int other, int seat, int replicate) {
        int key = (((own * 5 + other) * 2 + seat) * 17 + replicate);
        auto &cache = tails[key];
        auto it = cache.find(s);
        if (it != cache.end())
            return it->second;
        Tail value;
        if (!game.legal(s)) {
            value.points = game.outcome(s);
            value.draw = value.points == 0;
        } else {
            int a = play(s, int(s[2]) == seat ? own : other, replicate);
            State c = game.next(s, a);
            double signed_frequency = (s[2] ? -1 : 1) * game.score(s, c, a, model);
            value = game.prepend(signed_frequency, finish(c, own, other, seat, replicate));
        }
        if (cache.size() > 12000)
            cache.clear();
        cache.emplace(s, value);
        return value;
    }
    Menu evaluate(State s, Finish mode) {
        bool mixture = mode == Finish::Mixture || mode == Finish::GuardedMixture;
        int own =
            mode == Finish::Original ? 4 : int(mode == Finish::Guarded || mode == Finish::GuardedMixture);
        Menu z{};
        for (unsigned moves = game.legal(s); moves; moves &= moves - 1) {
            int a = first(moves);
            State c = game.next(s, a);
            double sign = s[2] ? -1 : 1, initial = sign * game.score(s, c, a, model);
            for (int j = 0; j < (mixture ? 4 : 1); ++j) {
                int other = mixture ? j : own;
                int n = other == 3 ? repetitions : 1;
                for (int k = 0; k < n; ++k) {
                    Tail v = game.prepend(initial, finish(c, own, other, int(s[2]), k));
                    double w = 1. / (n * (mixture ? 4 : 1));
                    z[a].last += w * sign * v.last;
                    z[a].sum += w * sign * v.sum;
                    z[a].decay += w * sign * v.decay;
                    z[a].points += w * (sign * v.points > 0);
                    z[a].draw += w * v.draw;
                }
            }
        }
        return z;
    }
    static double prize(const Tail &v, Prize u) {
        switch (u) {
        case Prize::Last:
            return v.last;
        case Prize::Sum:
            return v.sum;
        case Prize::Discounted:
            return v.decay;
        case Prize::Win:
            return v.points;
        case Prize::Points:
            return v.points + .5 * v.draw;
        case Prize::QuarterDraw:
            return v.points + .25 * v.draw;
        case Prize::Hybrid:
            return v.points + .25 * v.draw + .02 * v.decay;
        default:
            throw std::invalid_argument("unknown utility");
        }
    }

  public:
    WinPlanner(Engine &e, int model_id, U seed, int budget)
        : game(e), model(model_id), salt(seed), repetitions(budget) {
        if ((model != 1 && model != 2) || budget < 1 || budget > 16)
            throw std::invalid_argument("planner settings");
        if (game.models[model].table.empty())
            throw std::invalid_argument("install the frozen visit-frequency weights before creating a planner");
    }
    int choose(State s, int mode, int utility, bool fit_ties) {
        if (mode == int(Finish::Original) && game.models[0].table.empty())
            throw std::invalid_argument("the original-continuation diagnostic needs the original weights");
        if (utility == int(Prize::Direct))
            return game.best(s, model).move;
        auto &cache = menus.at(mode);
        auto it = cache.find(s);
        if (it == cache.end()) {
            if (cache.size() > 5000)
                cache.clear();
            it = cache.emplace(s, evaluate(s, Finish(mode))).first;
        }
        double best = -1e100, secondary = -1e100;
        int chosen = -1;
        for (unsigned moves = game.legal(s); moves; moves &= moves - 1) {
            int a = first(moves);
            double v = prize(it->second[a], Prize(utility));
            double tie = fit_ties ? game.score(s, game.next(s, a), a, model) : 0.;
            if (v > best || (v == best && tie > secondary)) {
                best = v;
                secondary = tie;
                chosen = a;
            }
        }
        return chosen;
    }
    int result(State s, int seat, int opponent, U seed, int mode, int utility, bool fit_ties, bool self) {
        int steps = 0, calls = 0, ply = 0;
        while (unsigned menu = game.legal(s)) {
            ++ply;
            int a;
            if (self || int(s[2]) == seat)
                a = choose(s, mode, utility, fit_ties);
            else
                a = game.sample(game.law(s, {double(opponent), 0, 0, -1}, steps, calls), menu,
                                noise(seed + ply * 397));
            s = game.next(s, a);
        }
        return game.outcome(s) * (seat ? -1 : 1);
    }
};
