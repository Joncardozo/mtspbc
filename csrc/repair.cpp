#include <cstddef>
#include <cstdint>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <algorithm>
#include <tuple>
#include <vector>
#include <numeric>
#include <limits>
#include "VrpbcState.cpp"
#include "utils.hpp"

namespace py = pybind11;

template <typename T>
void shuffle_vector(std::vector<T>& vec, std::mt19937& rng) {
    std::shuffle(vec.begin(), vec.end(), rng);
}

inline double insert_cost(int customer, const std::vector<int>& route, int idx, const VrpbcState& state) {
    int pred = (idx == 0) ? 0 : route[idx - 1];
    int succ = (idx == static_cast<int>(route.size())) ? 0 : route[idx];
    return state.distances[pred][customer] + state.distances[customer][succ] - state.distances[pred][succ];
}

inline double insert_cost_empty_route(int customer, const VrpbcState& state) {
    return state.distances[0][customer] + state.distances[customer][0];
}

struct InsertEval {
    double max_d;
    double objective;
    int count_above_D;
    double sum_above_D;
};

static std::vector<double> compute_route_events(
    const std::vector<int>& route,
    const std::vector<std::vector<double>>& distances
) {
    std::vector<double> evts;
    evts.reserve(route.size() + 3);
    evts.push_back(0.0);
    std::vector<int> tour = {0};
    tour.insert(tour.end(), route.begin(), route.end());
    tour.push_back(0);
    for (size_t i = 1; i < tour.size(); ++i) {
        evts.push_back(evts.back() + distances[tour[i]][tour[i - 1]]);
    }
    return evts;
}


static inline double compute_vehicle_distance_sq(
    int node_i,
    double e,
    const std::vector<int>& tour_j,
    const std::vector<double>& events_j,
    const std::vector<std::vector<double>>& node_coord
) {

    auto it = std::upper_bound(events_j.begin(), events_j.end(), e);
    int j_event_idx;

    if (it == events_j.begin()) {
        j_event_idx = 0;
    } else if (it == events_j.end()) {
        double dist_sq = 0.0;
        for (size_t d = 0; d < node_coord[node_i].size(); ++d) {
            double diff = node_coord[node_i][d] - node_coord[0][d];
            dist_sq += diff * diff;
        }
        return dist_sq;
    } else {
        j_event_idx = static_cast<int>(it - events_j.begin()) - 1;
    }

    int max_arc = static_cast<int>(tour_j.size()) - 2;
    if (max_arc < 0) {
        double dist_sq = 0.0;
        for (size_t d = 0; d < node_coord[node_i].size(); ++d) {
            double diff = node_coord[node_i][d] - node_coord[0][d];
            dist_sq += diff * diff;
        }
        return dist_sq;
    }
    if (j_event_idx > max_arc) j_event_idx = max_arc;

    int from = tour_j[j_event_idx];
    int to = tour_j[j_event_idx + 1];
    double dt = e - events_j[j_event_idx];

    const auto& dep = node_coord[from];
    const auto& arr = node_coord[to];

    // Compute Euclidean length of arc
    double norm_sq = 0.0;
    for (size_t d = 0; d < dep.size(); ++d) {
        double diff = arr[d] - dep[d];
        norm_sq += diff * diff;
    }
    double eucl_dist = std::sqrt(norm_sq);
    if (eucl_dist == 0.0) eucl_dist = 1.0;

    double arc_time = events_j[j_event_idx + 1] - events_j[j_event_idx];

    double fraction = (arc_time > 0.0) ? (dt / arc_time) : 0.0;
    if (fraction > 1.0) fraction = 1.0;
    if (fraction < 0.0) fraction = 0.0;

    double dist_sq = 0.0;
    for (size_t d = 0; d < dep.size(); ++d) {
        double pos_d = dep[d] + (arr[d] - dep[d]) * fraction;
        double diff = node_coord[node_i][d] - pos_d;
        dist_sq += diff * diff;
    }
    return dist_sq;
}


struct DistanceScanResult {
    double max_d;
    int count_above_D;
    double sum_above_D;
};


static DistanceScanResult compute_distances_scan(
    const std::vector<std::vector<int>>& tours,
    const std::vector<std::vector<double>>& events,
    const std::vector<std::vector<double>>& node_coord,
    double D
) {
    int n_vehicles = static_cast<int>(tours.size());
    double global_max_d = 0.0;
    int count_above = 0;
    double sum_above = 0.0;

    for (int i = 0; i < n_vehicles; ++i) {
        const auto& tour_i = tours[i];
        const auto& events_i = events[i];

        for (size_t idx = 0; idx < events_i.size(); ++idx) {
            double e = events_i[idx];
            int node_i = tour_i[idx];
            double max_dist_sq = 0.0;

            for (int j = 0; j < n_vehicles; ++j) {
                if (i == j) continue;
                double d_sq = compute_vehicle_distance_sq(
                    node_i, e, tours[j], events[j], node_coord);
                if (d_sq > max_dist_sq) max_dist_sq = d_sq;
            }

            double dist = std::sqrt(max_dist_sq);
            if (dist > global_max_d) global_max_d = dist;
            if (dist > D) {
                count_above++;
                sum_above += dist;
            }
        }
    }
    return {global_max_d, count_above, sum_above};
}


static InsertEval evaluate_insertion(
    VrpbcState& state,
    int customer,
    int route_idx,
    int pos,
    std::vector<std::vector<int>>& tours,
    std::vector<std::vector<double>>& all_events
) {
    auto& route = state.routes[route_idx];
    route.insert(route.begin() + pos, customer);

    auto& tour = tours[route_idx];
    tour.clear();
    tour.push_back(0);
    tour.insert(tour.end(), route.begin(), route.end());
    tour.push_back(0);

    auto old_events = std::move(all_events[route_idx]);
    all_events[route_idx] = compute_route_events(route, state.distances);

    auto scan = compute_distances_scan(tours, all_events, state.node_coord, state.D);

    double penalty_criteria = std::max(0.0, scan.max_d - state.D);
    double penalty_count_criteria = scan.max_d > state.D ? 1.0 : 0.0;
    double penalty_count = state.D * 5e1 * penalty_count_criteria * scan.count_above_D;
    double penalty_max = state.D * 5e2 *
        ((scan.max_d > state.D) ? std::max(scan.max_d, penalty_criteria) : 0.0);
    double penalty_empty = state.D * 1e9 *
        std::count_if(state.routes.begin(), state.routes.end(),
                      [](const std::vector<int>& r) { return r.empty(); });
    double unfeasible_penalty = penalty_empty + penalty_max + penalty_count;

    double routes_cost = 0.0;
    for (const auto& r : state.routes) {
        if (r.empty()) continue;
        routes_cost += state.distances[0][r[0]];
        for (size_t k = 0; k + 1 < r.size(); ++k) {
            routes_cost += state.distances[r[k]][r[k + 1]];
        }
        routes_cost += state.distances[r.back()][0];
    }

    double objective = unfeasible_penalty + routes_cost;

    InsertEval result{scan.max_d, objective, scan.count_above_D, scan.sum_above_D};

    route.erase(route.begin() + pos);
    tour.clear();
    tour.push_back(0);
    tour.insert(tour.end(), route.begin(), route.end());
    tour.push_back(0);
    all_events[route_idx] = std::move(old_events);

    return result;
}


static void build_tours_and_events(
    const VrpbcState& state,
    std::vector<std::vector<int>>& tours,
    std::vector<std::vector<double>>& all_events
) {
    int n = static_cast<int>(state.routes.size());
    tours.resize(n);
    all_events.resize(n);
    for (int i = 0; i < n; ++i) {
        tours[i].clear();
        tours[i].push_back(0);
        tours[i].insert(tours[i].end(), state.routes[i].begin(), state.routes[i].end());
        tours[i].push_back(0);
        all_events[i] = compute_route_events(state.routes[i], state.distances);
    }
}


std::tuple<int, int, double> best_insert(int customer, VrpbcState& state) {
    double best_cost = std::numeric_limits<double>::max();
    int best_route_idx = -1, best_idx = -1;
    for (size_t r = 0; r < state.routes.size(); ++r) {
        auto& route = state.routes[r];
        for (size_t idx = 0; idx <= route.size(); ++idx) {
            double cost = insert_cost(customer, route, idx, state);
            if (cost < best_cost) {
                best_cost = cost;
                best_route_idx = r;
                best_idx = idx;
            }
        }
    }
    return {best_route_idx, best_idx, best_cost};
}


std::tuple<int, int, double> best_insert_max_d(
    int customer,
    VrpbcState& state,
    std::vector<std::vector<int>>& tours,
    std::vector<std::vector<double>>& all_events
) {
    double best_cost = std::numeric_limits<double>::max();
    int best_route_idx = -1, best_idx = -1;
    double best_max_d = std::numeric_limits<double>::max();
    bool found_feasible = false;

    for (size_t ir = 0; ir < state.routes.size(); ++ir) {
        auto& route = state.routes[ir];
        for (size_t idx = 0; idx <= route.size(); ++idx) {
            double cost = insert_cost(customer, route, idx, state);

            if (found_feasible && cost >= best_cost) continue;

            auto eval = evaluate_insertion(state, customer, ir, idx, tours, all_events);

            if (eval.max_d <= state.D) {
                if (!found_feasible || cost < best_cost) {
                    best_cost = cost;
                    best_route_idx = ir;
                    best_idx = idx;
                    best_max_d = eval.max_d;
                    found_feasible = true;
                }
            } else if (!found_feasible) {
                if (eval.max_d < best_max_d || (eval.max_d == best_max_d && cost < best_cost)) {
                    best_cost = cost;
                    best_route_idx = ir;
                    best_idx = idx;
                    best_max_d = eval.max_d;
                }
            }
        }
    }
    return {best_route_idx, best_idx, best_cost};
}


std::tuple<int, int, double> best_insert_sum_d(
    int customer,
    VrpbcState& state,
    std::vector<std::vector<int>>& tours,
    std::vector<std::vector<double>>& all_events
) {
    double best_cost = std::numeric_limits<double>::max();
    int best_route_idx = -1, best_idx = -1;
    double best_objective = std::numeric_limits<double>::max();
    bool found_feasible = false;

    for (size_t ir = 0; ir < state.routes.size(); ++ir) {
        auto& route = state.routes[ir];
        for (size_t idx = 0; idx <= route.size(); ++idx) {
            double cost = insert_cost(customer, route, idx, state);

            if (found_feasible && cost >= best_cost) continue;

            auto eval = evaluate_insertion(state, customer, ir, idx, tours, all_events);

            if (eval.max_d <= state.D) {
                if (!found_feasible || cost < best_cost) {
                    best_cost = cost;
                    best_route_idx = ir;
                    best_idx = idx;
                    best_objective = eval.objective;
                    found_feasible = true;
                }
            } else if (!found_feasible) {
                if (eval.objective < best_objective ||
                    (eval.objective == best_objective && cost < best_cost)) {
                    best_cost = cost;
                    best_route_idx = ir;
                    best_idx = idx;
                    best_objective = eval.objective;
                }
            }
        }
    }
    return {best_route_idx, best_idx, best_cost};
}


std::tuple<double, int, int, int> best_regret_insert(VrpbcState& state) {
    double best_regret = -1e9;
    int best_r_idx = -1, best_idx = -1, best_un = -1;
    for (int un : state.unassigned) {
        double best_cost = std::numeric_limits<double>::max();
        double sec_best_cost = std::numeric_limits<double>::max();
        int best_r = -1, best_i = -1;
        bool found_any = false;

        for (size_t r_idx = 0; r_idx < state.routes.size(); ++r_idx) {
            auto& route = state.routes[r_idx];
            for (size_t idx = 0; idx <= route.size(); ++idx) {
                double cost = insert_cost(un, route, idx, state);
                if (cost < best_cost) {
                    sec_best_cost = best_cost;
                    best_cost = cost;
                    best_r = r_idx;
                    best_i = idx;
                    found_any = true;
                } else if (cost < sec_best_cost) {
                    sec_best_cost = cost;
                }
            }
        }

        if (found_any) {
            double regret = (sec_best_cost == std::numeric_limits<double>::max())
                          ? best_cost
                          : (sec_best_cost - best_cost);
            if (regret > best_regret) {
                best_regret = regret;
                best_r_idx = best_r;
                best_idx = best_i;
                best_un = un;
            }
        }
    }
    return {best_regret, best_r_idx, best_idx, best_un};
}


static void update_tours_events_for_route(
    const VrpbcState& state,
    int route_idx,
    std::vector<std::vector<int>>& tours,
    std::vector<std::vector<double>>& all_events
) {
    tours[route_idx].clear();
    tours[route_idx].push_back(0);
    tours[route_idx].insert(tours[route_idx].end(),
        state.routes[route_idx].begin(), state.routes[route_idx].end());
    tours[route_idx].push_back(0);
    all_events[route_idx] = compute_route_events(state.routes[route_idx], state.distances);
}


VrpbcState greedy_repair(VrpbcState state, uint64_t seed) {
    std::mt19937 rng(seed);
    shuffle_vector(state.unassigned, rng);

    while (!state.unassigned.empty()) {
        int customer = state.unassigned.back();
        state.unassigned.pop_back();
        auto [route_idx, idx, _] = best_insert(customer, state);
        if (route_idx >= 0 && idx >= 0) {
            state.routes[route_idx].insert(state.routes[route_idx].begin() + idx, customer);
        } else {
            state.unassigned.push_back(customer);
            break;
        }
    }
    state.events_update();
    state.distances_update();
    return state;
}

VrpbcState greedy_repair_max_d(VrpbcState state, uint64_t seed) {
    std::mt19937 rng(seed);
    shuffle_vector(state.unassigned, rng);

    std::vector<std::vector<int>> tours;
    std::vector<std::vector<double>> all_events;
    build_tours_and_events(state, tours, all_events);

    while (!state.unassigned.empty()) {
        int customer = state.unassigned.back();
        state.unassigned.pop_back();
        auto [route_idx, idx, _] = best_insert_max_d(customer, state, tours, all_events);
        if (route_idx < 0 || idx < 0) {
            std::tie(route_idx, idx, std::ignore) = best_insert(customer, state);
        }
        if (route_idx >= 0 && idx >= 0) {
            state.routes[route_idx].insert(state.routes[route_idx].begin() + idx, customer);
            update_tours_events_for_route(state, route_idx, tours, all_events);
        } else {
            state.unassigned.push_back(customer);
            break;
        }
    }
    state.events_update();
    state.distances_update();
    return state;
}


VrpbcState greedy_repair_sum_d(VrpbcState state, uint64_t seed) {
    std::mt19937 rng(seed);
    shuffle_vector(state.unassigned, rng);

    std::vector<std::vector<int>> tours;
    std::vector<std::vector<double>> all_events;
    build_tours_and_events(state, tours, all_events);

    while (!state.unassigned.empty()) {
        int customer = state.unassigned.back();
        state.unassigned.pop_back();
        auto [route_idx, idx, _] = best_insert_sum_d(customer, state, tours, all_events);
        if (route_idx < 0 || idx < 0) {
            std::tie(route_idx, idx, std::ignore) = best_insert(customer, state);
        }
        if (route_idx >= 0 && idx >= 0) {
            state.routes[route_idx].insert(state.routes[route_idx].begin() + idx, customer);
            update_tours_events_for_route(state, route_idx, tours, all_events);
        } else {
            state.unassigned.push_back(customer);
            break;
        }
    }
    state.events_update();
    state.distances_update();
    return state;
}

VrpbcState regret_repair(VrpbcState state, uint64_t seed) {
    std::mt19937 rng(seed);
    shuffle_vector(state.unassigned, rng);

    while (!state.unassigned.empty()) {
        auto [_, route_idx, idx, un] = best_regret_insert(state);
        if (route_idx >= 0 && idx >= 0 && un >= 0) {
            state.routes[route_idx].insert(state.routes[route_idx].begin() + idx, un);
            auto it = std::find(state.unassigned.begin(), state.unassigned.end(), un);
            if (it != state.unassigned.end()) {
                state.unassigned.erase(it);
            }
        } else {
            break;
        }
    }
    state.events_update();
    state.distances_update();
    return state;
}

VrpbcState empty_route_repair(VrpbcState state, uint64_t seed) {
    std::mt19937 rng(seed);
    shuffle_vector(state.unassigned, rng);

    std::vector<std::vector<int>> tours;
    std::vector<std::vector<double>> all_events;
    build_tours_and_events(state, tours, all_events);

    while (!state.unassigned.empty()) {
        std::vector<size_t> empty_routes;
        for (size_t i = 0; i < state.routes.size(); ++i) {
            if (state.routes[i].empty()) {
                empty_routes.push_back(i);
            }
        }
        if (empty_routes.empty()) break;

        bool filled_any = false;
        for (size_t empty_route_idx : empty_routes) {
            if (state.unassigned.empty()) break;

            double best_cost = std::numeric_limits<double>::max();
            int best_customer = -1;
            double best_max_d = std::numeric_limits<double>::max();
            bool found_feasible = false;

            for (int customer : state.unassigned) {
                double cust_cost = insert_cost_empty_route(customer, state);

                auto eval = evaluate_insertion(state, customer, empty_route_idx, 0,
                                               tours, all_events);

                if (eval.max_d <= state.D) {
                    if (!found_feasible || cust_cost < best_cost) {
                        best_cost = cust_cost;
                        best_customer = customer;
                        best_max_d = eval.max_d;
                        found_feasible = true;
                    }
                } else if (!found_feasible) {
                    if (eval.max_d < best_max_d ||
                        (eval.max_d == best_max_d && cust_cost < best_cost)) {
                        best_cost = cust_cost;
                        best_customer = customer;
                        best_max_d = eval.max_d;
                    }
                }
            }

            if (best_customer >= 0) {
                state.routes[empty_route_idx].push_back(best_customer);
                auto it = std::find(state.unassigned.begin(), state.unassigned.end(), best_customer);
                if (it != state.unassigned.end()) {
                    state.unassigned.erase(it);
                }
                update_tours_events_for_route(state, empty_route_idx, tours, all_events);
                filled_any = true;
            }
        }
        if (!filled_any) break;
    }


    shuffle_vector(state.unassigned, rng);
    while (!state.unassigned.empty()) {
        int customer = state.unassigned.back();
        state.unassigned.pop_back();
        auto [route_idx, idx, _] = best_insert_max_d(customer, state, tours, all_events);
        if (route_idx < 0 || idx < 0) {
            std::tie(route_idx, idx, std::ignore) = best_insert(customer, state);
        }
        if (route_idx >= 0 && idx >= 0) {
            state.routes[route_idx].insert(state.routes[route_idx].begin() + idx, customer);
            update_tours_events_for_route(state, route_idx, tours, all_events);
        } else {
            state.unassigned.push_back(customer);
            break;
        }
    }

    state.events_update();
    state.distances_update();
    return state;
}


struct CoverageGap {
    int vehicle_i;
    int event_idx;
    int vehicle_j;
    double distance;
    double time;
    std::vector<double> pos_i;
    std::vector<double> pos_j;
};

static CoverageGap find_worst_coverage_gap(
    const std::vector<std::vector<int>>& tours,
    const std::vector<std::vector<double>>& all_events,
    const std::vector<std::vector<double>>& node_coord
) {
    CoverageGap worst;
    worst.distance = 0.0;
    worst.vehicle_i = -1;
    worst.vehicle_j = -1;
    worst.event_idx = -1;
    worst.time = 0.0;

    int n_vehicles = static_cast<int>(tours.size());

    for (int i = 0; i < n_vehicles; ++i) {
        const auto& tour_i = tours[i];
        const auto& events_i = all_events[i];

        for (size_t idx = 0; idx < events_i.size(); ++idx) {
            double e = events_i[idx];
            int node_i = tour_i[idx];

            for (int j = 0; j < n_vehicles; ++j) {
                if (i == j) continue;

                double d_sq = compute_vehicle_distance_sq(
                    node_i, e, tours[j], all_events[j], node_coord);
                double d = std::sqrt(d_sq);

                if (d > worst.distance) {
                    worst.distance = d;
                    worst.vehicle_i = i;
                    worst.event_idx = static_cast<int>(idx);
                    worst.vehicle_j = j;
                    worst.time = e;
                    worst.pos_i = node_coord[node_i];


                    auto it = std::upper_bound(all_events[j].begin(), all_events[j].end(), e);
                    if (it == all_events[j].end()) {
                        worst.pos_j = node_coord[0];
                    } else {
                        int j_idx = static_cast<int>(it - all_events[j].begin()) - 1;
                        if (j_idx < 0) j_idx = 0;
                        int max_arc = static_cast<int>(tours[j].size()) - 2;
                        if (j_idx > max_arc) j_idx = max_arc;
                        int from = tours[j][j_idx];
                        int to = tours[j][j_idx + 1];
                        double dt = e - all_events[j][j_idx];
                        double arc_time = all_events[j][j_idx + 1] - all_events[j][j_idx];
                        double fraction = (arc_time > 0.0) ? (dt / arc_time) : 0.0;
                        if (fraction > 1.0) fraction = 1.0;
                        if (fraction < 0.0) fraction = 0.0;
                        const auto& dep = node_coord[from];
                        const auto& arr = node_coord[to];
                        worst.pos_j.resize(dep.size());
                        for (size_t d = 0; d < dep.size(); ++d) {
                            worst.pos_j[d] = dep[d] + (arr[d] - dep[d]) * fraction;
                        }
                    }
                }
            }
        }
    }
    return worst;
}


struct CoverageInsert {
    int route_idx;
    int pos;
    double cost;
    double max_d;
};

static CoverageInsert best_coverage_insert(
    int customer,
    VrpbcState& state,
    std::vector<std::vector<int>>& tours,
    std::vector<std::vector<double>>& all_events,
    const CoverageGap& gap,
    int search_window
) {
    CoverageInsert best;
    best.route_idx = -1;
    best.pos = -1;
    best.cost = std::numeric_limits<double>::max();
    best.max_d = std::numeric_limits<double>::max();
    bool found_feasible = false;

    std::vector<int> target_routes = {gap.vehicle_i, gap.vehicle_j};

    for (int r = 0; r < static_cast<int>(state.routes.size()); ++r) {
        if (r != gap.vehicle_i && r != gap.vehicle_j) {
            target_routes.push_back(r);
        }
    }

    for (int r_idx : target_routes) {
        auto& route = state.routes[r_idx];

        int center_pos = 0;
        double min_time_diff = std::numeric_limits<double>::max();
        const auto& evts = all_events[r_idx];
        for (size_t k = 0; k < evts.size(); ++k) {
            double diff = std::abs(evts[k] - gap.time);
            if (diff < min_time_diff) {
                min_time_diff = diff;
                center_pos = static_cast<int>(k);
            }
        }

        int window = (r_idx == gap.vehicle_i || r_idx == gap.vehicle_j)
                     ? search_window : search_window / 2;

        int lo = std::max(0, center_pos - window);
        int hi = std::min(static_cast<int>(route.size()), center_pos + window);

        for (int pos = lo; pos <= hi; ++pos) {
            double cost = insert_cost(customer, route, pos, state);

            auto eval = evaluate_insertion(state, customer, r_idx, pos, tours, all_events);

            if (eval.max_d <= state.D) {
                if (!found_feasible || eval.max_d < best.max_d ||
                    (eval.max_d == best.max_d && cost < best.cost)) {
                    best.route_idx = r_idx;
                    best.pos = pos;
                    best.cost = cost;
                    best.max_d = eval.max_d;
                    found_feasible = true;
                }
            } else if (!found_feasible) {
                if (eval.max_d < best.max_d ||
                    (eval.max_d == best.max_d && cost < best.cost)) {
                    best.route_idx = r_idx;
                    best.pos = pos;
                    best.cost = cost;
                    best.max_d = eval.max_d;
                }
            }
        }
    }

    return best;
}

VrpbcState coverage_aware_repair(VrpbcState state, uint64_t seed) {
    std::mt19937 rng(seed);

    std::vector<std::vector<int>> tours;
    std::vector<std::vector<double>> all_events;
    build_tours_and_events(state, tours, all_events);

    const int SEARCH_WINDOW = 4;

    const int MAX_COVERAGE_GUIDED = std::max(1,
        static_cast<int>(state.unassigned.size() / 3));

    CoverageGap gap = find_worst_coverage_gap(tours, all_events, state.node_coord);

    if (gap.vehicle_i >= 0 && !state.unassigned.empty()) {
        std::vector<double> midpoint(state.node_coord[0].size());
        for (size_t d = 0; d < midpoint.size(); ++d) {
            midpoint[d] = (gap.pos_i[d] + gap.pos_j[d]) / 2.0;
        }

        struct CandidateDist {
            int customer;
            double dist_sq;
        };
        std::vector<CandidateDist> candidates;
        candidates.reserve(state.unassigned.size());
        for (int cust : state.unassigned) {
            double dsq = 0.0;
            for (size_t d = 0; d < midpoint.size(); ++d) {
                double diff = state.node_coord[cust][d] - midpoint[d];
                dsq += diff * diff;
            }
            candidates.push_back({cust, dsq});
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const CandidateDist& a, const CandidateDist& b) {
                      return a.dist_sq < b.dist_sq;
                  });

        int guided_count = std::min(MAX_COVERAGE_GUIDED,
                                     static_cast<int>(candidates.size()));

        for (int c = 0; c < guided_count; ++c) {
            int cust = candidates[c].customer;

            auto it = std::find(state.unassigned.begin(), state.unassigned.end(), cust);
            if (it == state.unassigned.end()) continue;

            auto ins = best_coverage_insert(
                cust, state, tours, all_events, gap, SEARCH_WINDOW);

            if (ins.route_idx >= 0) {
                state.routes[ins.route_idx].insert(
                    state.routes[ins.route_idx].begin() + ins.pos, cust);
                state.unassigned.erase(it);
                update_tours_events_for_route(state, ins.route_idx, tours, all_events);
            } else {
                auto [ri, pi, ci] = best_insert(cust, state);
                if (ri >= 0 && pi >= 0) {
                    state.routes[ri].insert(state.routes[ri].begin() + pi, cust);
                    state.unassigned.erase(it);
                    update_tours_events_for_route(state, ri, tours, all_events);
                }
            }
        }
    }

    shuffle_vector(state.unassigned, rng);

    while (!state.unassigned.empty()) {
        int customer = state.unassigned.back();
        state.unassigned.pop_back();
        auto [route_idx, idx, _] = best_insert_max_d(customer, state, tours, all_events);
        if (route_idx < 0 || idx < 0) {
            std::tie(route_idx, idx, std::ignore) = best_insert(customer, state);
        }
        if (route_idx >= 0 && idx >= 0) {
            state.routes[route_idx].insert(state.routes[route_idx].begin() + idx, customer);
            update_tours_events_for_route(state, route_idx, tours, all_events);
        } else {
            state.unassigned.push_back(customer);
            break;
        }
    }

    state.events_update();
    state.distances_update();
    return state;
}


struct InsertCandidate {
    int route_idx;
    int pos;
    double cost;
    double max_d;
};

static std::vector<InsertCandidate> collect_feasible_insertions(
    int customer,
    VrpbcState& state,
    std::vector<std::vector<int>>& tours,
    std::vector<std::vector<double>>& all_events,
    int max_candidates
) {
    std::vector<InsertCandidate> feasible;
    std::vector<InsertCandidate> infeasible;

    for (size_t ir = 0; ir < state.routes.size(); ++ir) {
        auto& route = state.routes[ir];
        for (size_t idx = 0; idx <= route.size(); ++idx) {
            double cost = insert_cost(customer, route, idx, state);
            auto eval = evaluate_insertion(state, customer, ir, idx, tours, all_events);

            InsertCandidate cand{static_cast<int>(ir), static_cast<int>(idx), cost, eval.max_d};
            if (eval.max_d <= state.D) {
                feasible.push_back(cand);
            } else {
                infeasible.push_back(cand);
            }
        }
    }


    std::sort(feasible.begin(), feasible.end(),
              [](const InsertCandidate& a, const InsertCandidate& b) {
                  return a.cost < b.cost;
              });
    std::sort(infeasible.begin(), infeasible.end(),
              [](const InsertCandidate& a, const InsertCandidate& b) {
                  return a.max_d < b.max_d || (a.max_d == b.max_d && a.cost < b.cost);
              });


    auto& pool = feasible.empty() ? infeasible : feasible;
    int cap = std::min(max_candidates, static_cast<int>(pool.size()));
    return std::vector<InsertCandidate>(pool.begin(), pool.begin() + cap);
}

VrpbcState greedy_repair_max_d_noisy(VrpbcState state, uint64_t seed) {
    std::mt19937 rng(seed);
    shuffle_vector(state.unassigned, rng);

    std::vector<std::vector<int>> tours;
    std::vector<std::vector<double>> all_events;
    build_tours_and_events(state, tours, all_events);

    const int MAX_CANDIDATES = 5;

    while (!state.unassigned.empty()) {
        int customer = state.unassigned.back();
        state.unassigned.pop_back();

        auto candidates = collect_feasible_insertions(
            customer, state, tours, all_events, MAX_CANDIDATES);

        if (candidates.empty()) {
            auto [ri, pi, ci] = best_insert(customer, state);
            if (ri >= 0 && pi >= 0) {
                state.routes[ri].insert(state.routes[ri].begin() + pi, customer);
                update_tours_events_for_route(state, ri, tours, all_events);
            } else {
                state.unassigned.push_back(customer);
                break;
            }
            continue;
        }

        std::vector<double> weights(candidates.size());
        for (size_t i = 0; i < candidates.size(); ++i) {
            weights[i] = static_cast<double>(candidates.size() - i);
        }
        std::discrete_distribution<> pick(weights.begin(), weights.end());
        int chosen = pick(rng);

        auto& c = candidates[chosen];
        state.routes[c.route_idx].insert(
            state.routes[c.route_idx].begin() + c.pos, customer);
        update_tours_events_for_route(state, c.route_idx, tours, all_events);
    }

    state.events_update();
    state.distances_update();
    return state;
}


VrpbcState greedy_repair_first_feasible(VrpbcState state, uint64_t seed) {
    std::mt19937 rng(seed);
    shuffle_vector(state.unassigned, rng);

    std::vector<std::vector<int>> tours;
    std::vector<std::vector<double>> all_events;
    build_tours_and_events(state, tours, all_events);

    std::vector<int> route_order(state.routes.size());
    std::iota(route_order.begin(), route_order.end(), 0);

    while (!state.unassigned.empty()) {
        int customer = state.unassigned.back();
        state.unassigned.pop_back();

        std::shuffle(route_order.begin(), route_order.end(), rng);

        int best_route_idx = -1, best_idx = -1;
        double best_cost = std::numeric_limits<double>::max();
        double best_max_d = std::numeric_limits<double>::max();
        bool found_feasible = false;

        for (int ir : route_order) {
            auto& route = state.routes[ir];

            for (size_t idx = 0; idx <= route.size(); ++idx) {
                double cost = insert_cost(customer, route, idx, state);
                auto eval = evaluate_insertion(state, customer, ir, idx, tours, all_events);

                if (eval.max_d <= state.D) {
                    best_route_idx = ir;
                    best_idx = idx;
                    best_cost = cost;
                    best_max_d = eval.max_d;
                    found_feasible = true;
                    break;
                } else if (!found_feasible) {
                    if (eval.max_d < best_max_d ||
                        (eval.max_d == best_max_d && cost < best_cost)) {
                        best_route_idx = ir;
                        best_idx = idx;
                        best_cost = cost;
                        best_max_d = eval.max_d;
                    }
                }
            }

            if (found_feasible) break;
        }

        if (best_route_idx >= 0 && best_idx >= 0) {
            state.routes[best_route_idx].insert(
                state.routes[best_route_idx].begin() + best_idx, customer);
            update_tours_events_for_route(state, best_route_idx, tours, all_events);
        } else {
            auto [ri, pi, ci] = best_insert(customer, state);
            if (ri >= 0 && pi >= 0) {
                state.routes[ri].insert(state.routes[ri].begin() + pi, customer);
                update_tours_events_for_route(state, ri, tours, all_events);
            } else {
                state.unassigned.push_back(customer);
                break;
            }
        }
    }

    state.events_update();
    state.distances_update();
    return state;
}


std::tuple<int, int, double> best_insert_best_pos(
    int customer,
    VrpbcState& state,
    std::vector<std::vector<int>>& tours,
    std::vector<std::vector<double>>& all_events
) {
    double best_cost = std::numeric_limits<double>::max();
    int best_route_idx = -1, best_idx = -1;
    double best_max_d = std::numeric_limits<double>::max();

    for (size_t ir = 0; ir < state.routes.size(); ++ir) {
        auto& route = state.routes[ir];
        for (size_t idx = 0; idx <= route.size(); ++idx) {
            double cost = insert_cost(customer, route, idx, state);
            auto eval = evaluate_insertion(state, customer, ir, idx, tours, all_events);

            if (eval.max_d < best_max_d ||
                (eval.max_d == best_max_d && cost < best_cost)) {
                best_cost = cost;
                best_route_idx = ir;
                best_idx = idx;
                best_max_d = eval.max_d;
            }
        }
    }
    return {best_route_idx, best_idx, best_cost};
}

VrpbcState greedy_repair_best_pos(VrpbcState state, uint64_t seed) {
    std::mt19937 rng(seed);
    shuffle_vector(state.unassigned, rng);

    std::vector<std::vector<int>> tours;
    std::vector<std::vector<double>> all_events;
    build_tours_and_events(state, tours, all_events);

    while (!state.unassigned.empty()) {
        int customer = state.unassigned.back();
        state.unassigned.pop_back();
        auto [route_idx, idx, _] = best_insert_best_pos(customer, state, tours, all_events);
        if (route_idx < 0 || idx < 0) {
            std::tie(route_idx, idx, std::ignore) = best_insert(customer, state);
        }
        if (route_idx >= 0 && idx >= 0) {
            state.routes[route_idx].insert(state.routes[route_idx].begin() + idx, customer);
            update_tours_events_for_route(state, route_idx, tours, all_events);
        } else {
            state.unassigned.push_back(customer);
            break;
        }
    }

    state.events_update();
    state.distances_update();
    return state;
}

VrpbcState local_search(VrpbcState state, uint64_t /*seed*/) {
    return state;
}

std::vector<std::tuple<double, double>> mean_route(VrpbcState state) {
    std::vector<std::tuple<double, double>> mean{};
    int num_vehicles = state.routes.size();
    std::vector<std::tuple<double, int, int>> all_events{};
    for (int er_i = 0; er_i < (int)state.events.size(); er_i++){
        for (int e_i = 0; e_i < (int)state.events[er_i].size(); e_i++) {
            all_events.push_back({state.events[er_i][e_i], er_i, e_i});
        }
    }
    std::sort(all_events.begin(), all_events.end(),
        [](const auto& a, const auto& b) {
            return std::get<0>(a) < std::get<0>(b);
        }
    );

    for (auto& event : all_events) {
        std::tuple<double, double> mean_point{0.0, 0.0};
        for (size_t r_idx = 0; r_idx < state.routes.size(); r_idx++) {
            int r_e_idx = std::get<1>(event);
            double curr_e = std::get<0>(event);
            std::vector<int> route = state.routes[r_idx];
            std::vector<int> tour = {0};
            tour.insert(tour.end(), route.begin(), route.end());
            tour.push_back(0);
            if (r_e_idx == (int)r_idx) {
                std::get<0>(mean_point) += (state.node_coord[tour[std::get<2>(event)]][0] / num_vehicles);
                std::get<1>(mean_point) += (state.node_coord[tour[std::get<2>(event)]][1] / num_vehicles);
                continue;
            }
            int last_event_index = 0;
            for (size_t i = 0; i < tour.size(); i++) {
                if (state.events[r_idx][i] >= curr_e) {
                    break;
                }
                last_event_index = i;
            }
            if (last_event_index == (int)tour.size() - 1) {
                std::get<0>(mean_point) += (state.node_coord[0][0] / num_vehicles);
                std::get<1>(mean_point) += (state.node_coord[0][1] / num_vehicles);
            }
            else {
                auto arc = get_arc(tour, last_event_index);
                double diff_time = std::get<0>(event) - state.events[r_idx][last_event_index];
                double arc_time = state.events[r_idx][last_event_index + 1] - state.events[r_idx][last_event_index];
                auto real_pos = real_position(state.node_coord, arc, diff_time, arc_time);
                std::get<0>(mean_point) += (real_pos[0] / num_vehicles);
                std::get<1>(mean_point) += (real_pos[1] / num_vehicles);
            }
        }
        mean.push_back(mean_point);
    }
    return mean;
}


// PYBIND11 MODULE

PYBIND11_MODULE(repair_native, m) {
    m.def("greedy_repair", &greedy_repair, "Greedy repair operator", py::arg("state"), py::arg("seed"));
    m.def("regret_repair", &regret_repair, "Regret repair operator", py::arg("state"), py::arg("seed"));
    m.def("greedy_repair_max_d", &greedy_repair_max_d, "Greedy repair max_d", py::arg("state"), py::arg("seed"));
    m.def("greedy_repair_sum_d", &greedy_repair_sum_d, "Greedy repair sum_d (using penalty_sum)", py::arg("state"), py::arg("seed"));
    m.def("empty_route_repair", &empty_route_repair, "Repair operator that prioritizes filling empty routes", py::arg("state"), py::arg("seed"));
    m.def("coverage_aware_repair", &coverage_aware_repair, "Coverage-aware repair: targets worst inter-vehicle gap", py::arg("state"), py::arg("seed"));
    m.def("greedy_repair_max_d_noisy", &greedy_repair_max_d_noisy, "Noisy greedy repair: weighted random among top-k feasible positions", py::arg("state"), py::arg("seed"));
    m.def("greedy_repair_first_feasible", &greedy_repair_first_feasible, "First-feasible greedy: takes first feasible position found", py::arg("state"), py::arg("seed"));
    m.def("greedy_repair_best_pos", &greedy_repair_best_pos, "Best-position greedy: minimizes max_d first, then cost as tiebreak", py::arg("state"), py::arg("seed"));
    m.def("local_search", &local_search, "Local search (identity)", py::arg("state"), py::arg("seed"));
    m.def("mean_route", &mean_route, "Finds mean route", py::arg("state"));
}
