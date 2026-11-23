// Batched full-game simulation. Candidate scores are always from the current mover's perspective.
#include "phase_v_rules.hpp"
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using Scores = std::array<double, 17>;
using Spec = std::array<double, 4>; // model, utility (0 immediate, 1 last, 2 sum, 3 decay, 4 payoff, 5
                                    // fitted draw), original continuation, temperature
struct Action {
    int move;
    double value;
};
struct Tail {
    double last = 0, sum = 0, decay = 0, points = 0, draw = 0;
    int length = 0;
};
struct Model {
    std::vector<double> table;
    bool q = true;
    int layout = 0;
    std::unordered_map<State, Action, Hash> best;
    std::array<std::unordered_map<State, Tail, Hash>, 2> tails;
};
struct Engine : Rules {
    bool mixed = false;
    U salt = 0;
    double draw_value = 0;
    std::array<Model, 20> models;
    std::unordered_map<State, int, Hash> observed_choices;
    std::unordered_map<State, double, Hash> observed_children;
    std::unordered_map<State, Action, Hash> corrections;
    explicit Engine(bool r) : Rules(r) {}
    void configure(bool random, U seed) {
        mixed = random;
        salt = seed;
        for (auto &m : models)
            for (auto &t : m.tails)
                t.clear();
    }
    void set(int id, bool q, int layout,
             py::array_t<double, py::array::c_style | py::array::forcecast> data) {
        if (data.size() != groups * stride)
            throw std::invalid_argument("pattern table size");
        Model &m = models.at(id);
        m.q = q;
        m.layout = layout;
        m.table.assign(data.data(), data.data() + data.size());
        m.best.clear();
        for (auto &t : m.tails)
            t.clear();
        if (id == 0) {
            corrections.clear();
            for (auto &mod : models)
                mod.tails[1].clear();
        }
        if (id >= 12 && id <= 14)
            models[15].best.clear();
        if (id == 3) {
            models[17].best.clear();
            for (auto &tail : models[17].tails)
                tail.clear();
        }
        if (id == 1 || id == 2)
            reset_observed(id == 1 ? 19 : 18);
    }
    void correct(const std::vector<State> &states, const std::vector<int> &moves,
                 const std::vector<double> &delta) {
        if (states.size() != moves.size() || moves.size() != delta.size())
            throw std::invalid_argument("correction sizes");
        for (size_t i = 0; i < states.size(); i++)
            corrections[states[i]] = {moves[i], delta[i]};
        models[0].best.clear();
        for (auto &m : models)
            m.tails[1].clear();
    }
    // Consequences supplied by the game rules. The linear readout is fitted only to observed moves.
    std::array<double, 2> consequences(State s, int a) {
        State c = next(s, a);
        if (rev)
            return {double(-solve(c)), 0.};
        State other = s;
        other[2] ^= 1;
        return {double(four(c[s[2]])), double(four(next(other, a)[other[2]]))};
    }
    double choice_logit(State s, int a) {
        const auto &w = models[16].table;
        auto x = consequences(s, a);
        return w[a] + w[actions] * x[0] + w[actions + 1] * x[1];
    }
    double score(State s, State c, int a, int id, bool bounded = true) {
        if (id == 18) {
            auto it = observed_choices.find(s);
            return it == observed_choices.end() ? score(s, c, a, 2) : double(it->second == a);
        }
        if (id == 19) {
            auto it = observed_children.find(c);
            return it == observed_children.end() ? score(s, c, a, 1) : it->second;
        }
        if (id == 16 || id == 17) {
            auto logit = [&](int move) {
                return id == 16 ? choice_logit(s, move) : score(s, next(s, move), move, 3);
            };
            double z = logit(a), total = 0;
            for (unsigned m = legal(s); m; m &= m - 1)
                total += std::exp(logit(first(m)) - z);
            return 1. / total;
        }
        if (id == 5)
            return models[5].table[a];
        if (id == 15) {
            double w = std::clamp(score(s, c, a, 12), 0., 1.), d = std::clamp(score(s, c, a, 13), 0., 1.),
                   l = std::clamp(score(s, c, a, 14), 0., 1.);
            return w + d + l > 0 ? (w - l) / (w + d + l) : 0;
        }
        const auto &m = models.at(id);
        auto f = codes(s, c, a, m.q, m.layout);
        double v = 0;
        for (int k : f)
            v += m.table[k];
        if (id == 0 && !corrections.empty()) {
            auto i = corrections.find(s);
            if (i != corrections.end() && i->second.move == a)
                v += i->second.value;
        }
        if (bounded && (id == 1 || id == 2 || id == 7 || id == 8))
            return std::clamp(v, 0., 1.);
        return v;
    }
    Action best(State s, int id) {
        auto &cache = models[id].best;
        auto pos = cache.find(s);
        if (pos != cache.end())
            return pos->second;
        Action winner{-1, -1e100};
        for (unsigned m = legal(s); m; m &= m - 1) {
            int a = first(m);
            double v = score(s, next(s, a), a, id);
            if (v > winner.value)
                winner = {a, v};
        }
        if (cache.size() > 45000)
            cache.clear();
        cache.emplace(s, winner);
        return winner;
    }
    int nth(unsigned mask, double u) const {
        int n = int(u * bits(mask));
        while (n-- > 0)
            mask &= mask - 1;
        return first(mask);
    }
    int original(State s, unsigned mask, double u) {
        return mixed && u < .25 ? nth(mask, u * 4) : best(s, 0).move;
    }
    Tail prepend(double val, Tail t) const {
        return {
            t.length ? t.last : val, val + t.sum, val + .9 * t.decay, t.points, t.draw, t.length + 1};
    }
    Tail tail(State s, int model, bool who, U key) {
        auto &cache = models[model].tails[who];
        bool fixed = !(who && mixed);
        if (fixed) {
            auto it = cache.find(s);
            if (it != cache.end())
                return it->second;
        }
        unsigned mask = legal(s);
        Tail t;
        if (!mask) {
            t.points = outcome(s);
            t.draw = t.points == 0;
        } else {
            Action a;
            if (who) {
                a.move = original(s, mask, noise(key));
                a.value = score(s, next(s, a.move), a.move, model);
            } else
                a = best(s, model);
            t = prepend((s[2] ? -1 : 1) * a.value,
                        tail(next(s, a.move), model, who, key + 0x9e3779b97f4a7c15ULL));
        }
        if (fixed) {
            if (cache.size() > 45000)
                cache.clear();
            cache.emplace(s, t);
        }
        return t;
    }
    Scores profile(State s, int id, int rule, bool who, int &steps, int &calls) {
        Scores z;
        z.fill(-1e30);
        unsigned mask = legal(s);
        double sign = s[2] ? -1 : 1;
        for (unsigned moves = mask; moves; moves &= moves - 1) {
            int a = first(moves);
            State c = next(s, a);
            if (id == -1) {
                z[a] = 0;
                continue;
            }
            if (id == -2) {
                z[a] = priority(a);
                continue;
            }
            if (id == -3) {
                if (rev)
                    z[a] = -solve(c);
                else {
                    State o = s;
                    o[2] ^= 1;
                    z[a] = 10 * (2 * four(c[s[2]]) + four(next(o, a)[o[2]])) + priority(a);
                }
                continue;
            }
            double v = score(s, c, a, id);
            if (rule == 0) {
                z[a] = v;
                continue;
            }
            double value = 0;
            int n = (who && mixed) ? 4 : 1;
            for (int k = 0; k < n; k++) {
                U key = Hash{}(s) ^ scramble(salt + 131 * a + 8191 * k);
                if (rule == 8) {
                    // Whole-play utility in first-seat units, with each move measured
                    // relative to the best score available to its own mover.
                    double utility = sign * (v - best(s, id).value);
                    State h = c;
                    ++steps;
                    while (unsigned menu = legal(h)) {
                        auto favorite = best(h, id);
                        int act = who ? original(h, menu, noise(key)) : favorite.move;
                        State child = next(h, act);
                        utility += (h[2] ? -1. : 1.) * (score(h, child, act, id) - favorite.value);
                        h = child;
                        key += 0x9e3779b97f4a7c15ULL;
                        ++steps;
                        calls += who;
                    }
                    value += utility;
                    continue;
                }
                auto t = prepend(sign * v, tail(c, id, who, key));
                steps += t.length;
                if (who)
                    calls += t.length - 1;
                value += rule == 1   ? t.last
                         : rule == 2 ? t.sum
                         : rule == 3 ? t.decay
                         : rule == 4 ? t.points
                                     : t.points + draw_value * t.draw;
            }
            z[a] = sign * value / n;
        }
        return z;
    }
    Scores law(State s, Spec spec, int &steps, int &calls) {
        int id = int(spec[0]);
        unsigned mask = legal(s);
        Scores p{};
        if (!mask)
            return p;
        if (id == -4 || id == 0) {
            int fav = best(s, 0).move;
            if (mixed) {
                for (unsigned m = mask; m; m &= m - 1)
                    p[first(m)] = .25 / bits(mask);
                p[fav] += .75;
            } else
                p[fav] = 1;
            return p;
        }
        if (id == -1) {
            for (unsigned m = mask; m; m &= m - 1)
                p[first(m)] = 1. / bits(mask);
            return p;
        }
        if (spec[1] == 6) {
            double old = draw_value;
            spec[1] = 5;
            draw_value = -1;
            auto low = law(s, spec, steps, calls);
            draw_value = 1;
            auto high = law(s, spec, steps, calls);
            draw_value = old;
            for (int a = 0; a < actions; a++)
                p[a] = (low[a] + high[a]) / 2;
            return p;
        }
        if (spec[1] == 7) {
            spec[1] = 5;
            p = law(s, spec, steps, calls);
            if (!mixed) {
                p.fill(0);
                p[best(s, 0).move] = 1;
            }
            return p;
        }
        if (id >= 0 && spec[1] == 0 && spec[3] < 0) {
            p[best(s, id).move] = 1;
            return p;
        }
        Scores z = profile(s, id, int(spec[1]), bool(spec[2]), steps, calls);
        int a = std::max_element(z.begin(), z.begin() + actions) - z.begin();
        if (spec[3] < 0) {
            p[a] = 1;
            return p;
        }
        double total = 0;
        for (unsigned m = mask; m; m &= m - 1) {
            int j = first(m);
            p[j] = spec[3] == 0 ? double(z[j] == z[a]) : exp((z[j] - z[a]) / spec[3]);
            total += p[j];
        }
        for (double &v : p)
            v /= total;
        return p;
    }
    int sample(const Scores &p, unsigned mask, double u) const {
        int a = first(mask);
        for (; mask; mask &= mask - 1) {
            a = first(mask);
            u -= p[a];
            if (u < 0)
                break;
        }
        return a;
    }
    std::vector<State> trajectory(State s, Spec x, Spec o, U seed) {
        std::vector<State> path{s};
        int steps = 0, calls = 0;
        while (unsigned m = legal(s)) {
            auto p = law(s, s[2] ? o : x, steps, calls);
            int a = sample(p, m, noise(seed + path.size() * 397));
            s = next(s, a);
            path.push_back(s);
        }
        return path;
    }
    std::array<int, 4> match(State s, Spec x, Spec o, U seed) {
        int ply = 0, steps = 0, calls = 0;
        while (unsigned m = legal(s)) {
            auto p = law(s, s[2] ? o : x, steps, calls);
            int a = sample(p, m, noise(seed + (++ply) * 397));
            s = next(s, a);
        }
        return {outcome(s), ply, steps, calls};
    }
    py::array_t<int> features(const std::vector<State> &states, bool q, int layout) {
        py::array_t<int> out({int(states.size()), actions, groups});
        auto buf = out.mutable_data();
        std::fill(buf, buf + out.size(), -1);
        for (size_t i = 0; i < states.size(); i++)
            for (unsigned m = legal(states[i]); m; m &= m - 1) {
                int a = first(m);
                auto v = codes(states[i], next(states[i], a), a, q, layout);
                std::copy(v.begin(), v.end(), buf + (i * actions + a) * groups);
            }
        return out;
    }
    py::array_t<double> profiles(const std::vector<State> &states, Spec spec, bool probs) {
        py::array_t<double> out({int(states.size()), actions + 2});
        auto v = out.mutable_unchecked<2>();
        for (size_t i = 0; i < states.size(); i++) {
            int steps = 0, calls = 0;
            Scores z =
                probs ? law(states[i], spec, steps, calls)
                      : profile(states[i], int(spec[0]), int(spec[1]), bool(spec[2]), steps, calls);
            for (int a = 0; a < actions; a++)
                v(i, a) = z[a];
            v(i, actions) = steps;
            v(i, actions + 1) = calls;
        }
        return out;
    }
    std::vector<State> all() {
        if (!rev)
            throw std::invalid_argument("only Reversi is exhaustively solved");
        solve(start());
        std::vector<State> v;
        for (auto &it : solved)
            v.push_back(it.first);
        std::sort(v.begin(), v.end());
        return v;
    }
    py::array_t<double> raw_scores(const std::vector<State> &states, int id) {
        py::array_t<double> out({int(states.size()), actions});
        auto z = out.mutable_unchecked<2>();
        for (size_t i = 0; i < states.size(); ++i)
            for (int a = 0; a < actions; ++a)
                z(i, a) = (legal(states[i]) & (1u << a))
                              ? score(states[i], next(states[i], a), a, id, false)
                              : -1e30;
        return out;
    }
    void reset_observed(int id) {
        models[id].best.clear();
        for (auto &t : models[id].tails)
            t.clear();
    }
    void fit_observed_choices(const std::vector<State> &s, const std::vector<int> &a) {
        if (s.size() != a.size())
            throw std::invalid_argument("choice counts");
        observed_choices.clear();
        for (size_t i = 0; i < s.size(); i++)
            observed_choices[s[i]] = a[i];
        reset_observed(18);
    }
    void fit_observed_children(const std::vector<State> &s, const std::vector<double> &v) {
        if (s.size() != v.size())
            throw std::invalid_argument("child values");
        observed_children.clear();
        for (size_t i = 0; i < s.size(); i++)
            observed_children[s[i]] = v[i];
        reset_observed(19);
    }
    // The forced alternative is an experimental intervention. Only subsequent moves
    // actually selected by the original count as observations of its behavior.
    py::array_t<U> continuation_visits(const std::vector<State> &roots) {
        if (mixed)
            throw std::invalid_argument("fixed source required");
        std::unordered_map<State, U, Hash> visits;
        for (State s : roots)
            for (unsigned m = legal(s); m; m &= m - 1) {
                State h = next(s, first(m));
                while (legal(h)) {
                    ++visits[h];
                    h = next(h, best(h, 0).move);
                }
            }
        std::vector<State> sorted;
        sorted.reserve(visits.size());
        for (auto &kv : visits)
            sorted.push_back(kv.first);
        std::sort(sorted.begin(), sorted.end());
        py::array_t<U> out({int(sorted.size()), 5});
        auto x = out.mutable_unchecked<2>();
        for (size_t i = 0; i < sorted.size(); i++) {
            State s = sorted[i];
            for (int j = 0; j < 3; j++)
                x(i, j) = s[j];
            x(i, 3) = best(s, 0).move;
            x(i, 4) = visits[s];
        }
        return out;
    }
    py::array_t<bool> fitted_paths(const std::vector<State> &roots, int id) {
        if (id != 18 && id != 19)
            throw std::invalid_argument("requires a fit with observed inputs");
        py::array_t<bool> out({int(roots.size()), actions});
        std::fill(out.mutable_data(), out.mutable_data() + out.size(), false);
        auto x = out.mutable_unchecked<2>();
        for (size_t i = 0; i < roots.size(); i++)
            for (unsigned m = legal(roots[i]); m; m &= m - 1) {
                State h = roots[i];
                int a = first(m), initial = a;
                while (true) {
                    State c = next(h, a);
                    if (id == 18 ? !observed_choices.count(h) : !observed_children.count(c))
                        break;
                    if (!legal(c)) {
                        x(i, initial) = true;
                        break;
                    }
                    h = c;
                    if (id == 19) {
                        bool menu_seen = true;
                        for (unsigned options = legal(h); options; options &= options - 1)
                            menu_seen &= observed_children.count(next(h, first(options))) != 0;
                        if (!menu_seen)
                            break;
                    }
                    a = best(h, id).move;
                }
            }
        return out;
    }
    py::array_t<double> rule_inputs(const std::vector<State> &states) {
        py::array_t<double> out({int(states.size()), actions, 2});
        auto x = out.mutable_unchecked<3>();
        std::fill(out.mutable_data(), out.mutable_data() + out.size(), 0.);
        for (size_t i = 0; i < states.size(); ++i)
            for (unsigned m = legal(states[i]); m; m &= m - 1) {
                int a = first(m);
                auto v = consequences(states[i], a);
                x(i, a, 0) = v[0];
                x(i, a, 1) = v[1];
            }
        return out;
    }
    // Force the candidate once, then follow the specified fixed player in both seats.
    // Columns are path length, final mover relative to root, exact and fitted final frequencies.
    py::array_t<double> terminal_probe(const std::vector<State> &states, int fitted, int continuation) {
        if (mixed)
            throw std::invalid_argument("terminal probe requires a fixed source");
        py::array_t<double> out({int(states.size()), actions, 4});
        std::fill(out.mutable_data(), out.mutable_data() + out.size(), 0.);
        auto x = out.mutable_unchecked<3>();
        for (size_t i = 0; i < states.size(); ++i)
            for (unsigned m = legal(states[i]); m; m &= m - 1) {
                State h = states[i];
                int a = first(m), chosen = a, length = 1;
                State c = next(h, chosen);
                while (legal(c)) {
                    h = c;
                    chosen = best(h, continuation).move;
                    c = next(h, chosen);
                    ++length;
                }
                x(i, a, 0) = length;
                x(i, a, 1) = states[i][2] == h[2] ? 1 : -1;
                x(i, a, 2) = chosen == best(h, 0).move;
                x(i, a, 3) = score(h, c, chosen, fitted);
            }
        return out;
    }
};
#include "win_vnm.hpp"
PYBIND11_MODULE(_phase_v_native, m) {
    py::class_<Engine>(m, "Engine")
        .def(py::init<bool>())
        .def("start", &Engine::start)
        .def("legal", &Engine::legal)
        .def("next", &Engine::next)
        .def("outcome", &Engine::outcome)
        .def("canonical", &Engine::canonical)
        .def("teacher", &Engine::teacher)
        .def("solve", &Engine::solve)
        .def("all", &Engine::all)
        .def("features", &Engine::features)
        .def("configure", &Engine::configure)
        .def("set", &Engine::set)
        .def("correct", &Engine::correct)
        .def("profiles", &Engine::profiles)
        .def("raw_scores", &Engine::raw_scores)
        .def("rule_inputs", &Engine::rule_inputs)
        .def("terminal_probe", &Engine::terminal_probe)
        .def("fit_observed_choices", &Engine::fit_observed_choices)
        .def("fit_observed_children", &Engine::fit_observed_children)
        .def("continuation_visits", &Engine::continuation_visits)
        .def("fitted_paths", &Engine::fitted_paths)
        .def("match", &Engine::match)
        .def("trajectory", &Engine::trajectory)
        .def_readwrite("draw_value", &Engine::draw_value);
    py::class_<WinPlanner>(m, "WinPlanner")
        .def(py::init<Engine &, int, U, int>(), py::keep_alive<1, 2>())
        .def("choose", &WinPlanner::choose, py::arg("state"), py::arg("mode"), py::arg("utility"),
             py::arg("fit_ties"), py::arg("offered") = ~0u)
        .def("result", &WinPlanner::result);
}
