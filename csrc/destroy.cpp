#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <algorithm>
#include <vector>
#include <numeric>
#include <limits>
#include <set>
#include <cstdint>
#include "VrpbcState.cpp"
#include "utils.hpp"

namespace py = pybind11;


std::vector<int> remove_string(std::vector<int>& route, int cust, int size, int max_string_size, std::mt19937& rng) {
    if (route.empty()) return {};
    int max_allowed_size = std::min((int)route.size(), max_string_size);
    if (size > max_allowed_size) size = max_allowed_size;
    int cust_idx = std::find(route.begin(), route.end(), cust) - route.begin();
    if (cust_idx == static_cast<int>(route.size())) return {};  // Customer not found

    int start = cust_idx - static_cast<int>(size / 2);

    std::vector<int> idcs;
    for (int i = 0; i < size; ++i) {
        idcs.push_back((start + i + route.size()) % route.size());
    }

    std::set<int> idx_set(idcs.begin(), idcs.end());
    std::vector<int> sorted_idcs(idx_set.begin(), idx_set.end());

    std::sort(sorted_idcs.rbegin(), sorted_idcs.rend());

    std::vector<int> removed_customers;
    for (int idx : sorted_idcs) {
        removed_customers.push_back(route[idx]);
        route.erase(route.begin() + idx);
    }
    return removed_customers;
}


std::vector<int> remove_string(std::vector<int>& route, int cust, int max_string_size, std::mt19937& rng) {
    if (route.empty()) return {};
    int max_allowed_size = std::min((int)route.size(), max_string_size);
    std::uniform_int_distribution<> size_dist(1, max_allowed_size);
    int size = size_dist(rng);
    return remove_string(route, cust, size, max_string_size, rng);
}


std::vector<int> remove_string_no_wrap(std::vector<int>& route, int cust, int size, int max_string_size) {
    if (route.empty()) return {};
    int max_allowed_size = std::min((int)route.size(), max_string_size);
    if (size > max_allowed_size) size = max_allowed_size;
    if (size <= 0) return {};

    int cust_idx = std::find(route.begin(), route.end(), cust) - route.begin();
    if (cust_idx == static_cast<int>(route.size())) return {};

    int half_size = size / 2;
    int start, end;

    if (cust_idx < half_size) {
        start = 0;
        end = size - 1;
    } else if (cust_idx + half_size >= static_cast<int>(route.size())) {
        start = route.size() - size;
        end = route.size() - 1;
    } else {
        start = cust_idx - half_size;
        end = start + size - 1;
    }

    start = std::max(0, start);
    end = std::min(static_cast<int>(route.size()) - 1, end);

    std::vector<int> removed_customers;
    for (int i = end; i >= start; --i) {
        removed_customers.push_back(route[i]);
        route.erase(route.begin() + i);
    }
    return removed_customers;
}


inline void add_to_unassigned(VrpbcState& state, const std::set<int>& removed) {
    for (int customer : removed) {
        if (std::find(state.unassigned.begin(), state.unassigned.end(), customer) == state.unassigned.end()) {
            state.unassigned.push_back(customer);
        }
    }
}


inline void finalize_destroyed(VrpbcState& state) {
    state.events_update();
    state.distances_update();
    state.ensure_unassigned_consistency();
}

// =====================================================================
// DESTROY OPERATORS
// =====================================================================

// String removal: spatial nearest-neighbor chaining across routes
VrpbcState string_removal(const VrpbcState& state, uint64_t seed,
                          const int MAX_STRING_REMOVALS, const int MAX_STRING_SIZE) {
    VrpbcState destroyed = state.copy();
    std::mt19937 rng(seed);

    if (destroyed.routes.empty()) {
        finalize_destroyed(destroyed);
        return destroyed;
    }


    std::vector<int> non_empty_routes;
    for (size_t i = 0; i < destroyed.routes.size(); ++i) {
        if (!destroyed.routes[i].empty()) {
            non_empty_routes.push_back(i);
        }
    }
    if (non_empty_routes.empty()) {
        finalize_destroyed(destroyed);
        return destroyed;
    }

    std::set<int> removed;
    std::set<size_t> processed_route_indices;

    std::uniform_int_distribution<> route_dist(0, non_empty_routes.size() - 1);
    size_t current_route_idx = non_empty_routes[route_dist(rng)];
    auto& current_route = destroyed.routes[current_route_idx];

    std::uniform_int_distribution<> cust_dist(0, current_route.size() - 1);
    int current_customer = current_route[cust_dist(rng)];

    int num_routes_to_process = std::min((int)non_empty_routes.size(), MAX_STRING_REMOVALS);
    std::uniform_int_distribution<> num_routes_dist(1, num_routes_to_process);
    int target_num_routes = num_routes_dist(rng);

    std::uniform_int_distribution<> size_dist(1, MAX_STRING_SIZE);

    int size = size_dist(rng);
    auto first_removed = remove_string(current_route, current_customer, size, MAX_STRING_SIZE, rng);
    removed.insert(first_removed.begin(), first_removed.end());
    processed_route_indices.insert(current_route_idx);

    while ((int)processed_route_indices.size() < target_num_routes) {
        std::vector<int> nearest_neighbors = neighbors(destroyed.distances, current_customer);
        int next_customer = -1;
        size_t next_route_idx = -1;

        for (int neighbor : nearest_neighbors) {
            if (removed.count(neighbor)) continue;
            if (std::find(destroyed.unassigned.begin(), destroyed.unassigned.end(), neighbor)
                != destroyed.unassigned.end()) continue;

            for (size_t r_idx = 0; r_idx < destroyed.routes.size(); ++r_idx) {
                if (processed_route_indices.count(r_idx)) continue;
                auto& route = destroyed.routes[r_idx];
                if (std::find(route.begin(), route.end(), neighbor) != route.end()) {
                    next_customer = neighbor;
                    next_route_idx = r_idx;
                    break;
                }
            }
            if (next_customer != -1) break;
        }

        if (next_customer == -1) break;

        size = size_dist(rng);
        auto& next_route = destroyed.routes[next_route_idx];
        auto next_removed = remove_string(next_route, next_customer, size, MAX_STRING_SIZE, rng);
        removed.insert(next_removed.begin(), next_removed.end());
        processed_route_indices.insert(next_route_idx);
        current_customer = next_customer;
    }

    add_to_unassigned(destroyed, removed);
    finalize_destroyed(destroyed);
    return destroyed;
}

// Event-proximity String removal
VrpbcState string_removal_close_events(const VrpbcState& state, uint64_t seed,
                                        const int MAX_STRING_REMOVALS, const int MAX_STRING_SIZE) {
    VrpbcState destroyed = state.copy();
    std::mt19937 rng(seed);

    std::uniform_int_distribution<> center_dist(1, destroyed.node_coord.size() - 1);
    int center = center_dist(rng);

    std::vector<int> customers_n = neighbors_events(
        destroyed.events, destroyed.routes, destroyed.distances, destroyed.get_D(), center);

    std::vector<size_t> non_empty_routes;
    for (size_t i = 0; i < destroyed.routes.size(); ++i) {
        if (!destroyed.routes[i].empty()) non_empty_routes.push_back(i);
    }
    if (non_empty_routes.empty()) {
        finalize_destroyed(destroyed);
        return destroyed;
    }

    int num_routes_to_process = std::min((int)non_empty_routes.size(), MAX_STRING_REMOVALS);
    std::uniform_int_distribution<> num_routes_dist(1, num_routes_to_process);
    int target_num_routes = num_routes_dist(rng);

    std::uniform_int_distribution<> size_dist(1, MAX_STRING_SIZE);
    std::set<size_t> processed_route_indices;
    std::set<int> removed;

    for (int customer : customers_n) {
        if ((int)processed_route_indices.size() >= target_num_routes) break;
        if (std::find(destroyed.unassigned.begin(), destroyed.unassigned.end(), customer)
            != destroyed.unassigned.end()) continue;

        for (size_t route_idx = 0; route_idx < destroyed.routes.size(); ++route_idx) {
            auto& route = destroyed.routes[route_idx];
            if (std::find(route.begin(), route.end(), customer) != route.end()) {
                if (processed_route_indices.count(route_idx)) continue;

                int size = size_dist(rng);
                auto customers = remove_string_no_wrap(route, customer, size, MAX_STRING_SIZE);
                removed.insert(customers.begin(), customers.end());
                processed_route_indices.insert(route_idx);
                break;
            }
        }
    }

    add_to_unassigned(destroyed, removed);
    finalize_destroyed(destroyed);
    return destroyed;
}

// Random removal with explicit degree
VrpbcState random_removal(const VrpbcState& state, uint64_t seed, double degree_of_destruction) {
    VrpbcState destroyed = state.copy();
    std::mt19937 rng(seed);

    std::vector<int> assigned_customers;
    for (const auto& route : destroyed.routes) {
        for (int cust : route) assigned_customers.push_back(cust);
    }

    int customers_to_remove = std::min(
        static_cast<int>(assigned_customers.size() - 1),
        static_cast<int>(destroyed.node_coord.size() * degree_of_destruction));
    if (customers_to_remove <= 0) {
        finalize_destroyed(destroyed);
        return destroyed;
    }

    std::shuffle(assigned_customers.begin(), assigned_customers.end(), rng);
    assigned_customers.resize(customers_to_remove);

    std::set<int> removed;
    for (int customer : assigned_customers) {
        for (auto& route : destroyed.routes) {
            auto it = std::find(route.begin(), route.end(), customer);
            if (it != route.end()) {
                removed.insert(customer);
                route.erase(it);
                break;
            }
        }
    }

    add_to_unassigned(destroyed, removed);
    finalize_destroyed(destroyed);
    return destroyed;
}

// Random removal with varying degree
VrpbcState random_removal_varying_degree(const VrpbcState& state, uint64_t seed,
                                          double lower_bound, double mode, double upper_bound) {
    VrpbcState destroyed = state.copy();
    std::mt19937 rng(seed);

    std::vector<double> intervals = {lower_bound, mode, upper_bound};
    std::vector<double> densities = {1.0, 1.0, 0.8};
    std::piecewise_linear_distribution<double> dist(intervals.begin(), intervals.end(), densities.begin());
    double degree_of_destruction = dist(rng);  // FIX: was using unseeded std::random_device

    std::vector<int> assigned_customers;
    for (const auto& route : destroyed.routes) {
        for (int cust : route) assigned_customers.push_back(cust);
    }

    int customers_to_remove = std::min(
        static_cast<int>(assigned_customers.size() - 1),
        static_cast<int>(destroyed.node_coord.size() * degree_of_destruction));
    if (customers_to_remove <= 0) {
        finalize_destroyed(destroyed);
        return destroyed;
    }

    std::shuffle(assigned_customers.begin(), assigned_customers.end(), rng);
    assigned_customers.resize(customers_to_remove);

    std::set<int> removed;
    for (int customer : assigned_customers) {
        for (auto& route : destroyed.routes) {
            auto it = std::find(route.begin(), route.end(), customer);
            if (it != route.end()) {
                removed.insert(customer);
                route.erase(it);
                break;
            }
        }
    }

    add_to_unassigned(destroyed, removed);
    finalize_destroyed(destroyed);
    return destroyed;
}

// Worst removal
VrpbcState worst_removal(const VrpbcState& state, uint64_t seed) {
    VrpbcState destroyed = state.copy();
    std::mt19937 rng(seed);

    struct NodeCost {
        int node;
        double removal_cost;
    };

    std::vector<NodeCost> node_costs;
    for (size_t route_idx = 0; route_idx < destroyed.routes.size(); ++route_idx) {
        const auto& route = destroyed.routes[route_idx];
        for (size_t pos = 0; pos < route.size(); ++pos) {
            int node = route[pos];
            int prev_node = (pos == 0) ? 0 : route[pos - 1];
            int next_node = (pos == route.size() - 1) ? 0 : route[pos + 1];
            double removal_cost = destroyed.distances[prev_node][node] +
                                  destroyed.distances[node][next_node] -
                                  destroyed.distances[prev_node][next_node];
            node_costs.push_back({node, removal_cost});
        }
    }

    if (node_costs.empty()) {
        finalize_destroyed(destroyed);
        return destroyed;
    }

    std::sort(node_costs.begin(), node_costs.end(),
              [](const NodeCost& a, const NodeCost& b) {
                  return a.removal_cost > b.removal_cost;
              });

    int max_to_remove = std::max(1, static_cast<int>(node_costs.size() * 0.15));
    int num_to_remove = std::min(max_to_remove, static_cast<int>(node_costs.size()));
    std::uniform_int_distribution<> num_dist(1, num_to_remove);
    int customers_to_remove = num_dist(rng);

    std::set<int> removed;
    for (int i = 0; i < customers_to_remove && i < static_cast<int>(node_costs.size()); ++i) {
        removed.insert(node_costs[i].node);
    }

    for (int customer : removed) {
        for (auto& route : destroyed.routes) {
            auto it = std::find(route.begin(), route.end(), customer);
            if (it != route.end()) {
                route.erase(it);
                break;
            }
        }
    }

    add_to_unassigned(destroyed, removed);
    finalize_destroyed(destroyed);
    return destroyed;
}

// Remove entire route
VrpbcState remove_route(const VrpbcState& state, uint64_t seed) {
    VrpbcState destroyed = state.copy();
    std::mt19937 rng(seed);
    std::uniform_int_distribution<> route_dist(0, destroyed.routes.size() - 1);
    int route_idx = route_dist(rng);

    std::set<int> removed(destroyed.routes[route_idx].begin(),
                          destroyed.routes[route_idx].end());
    add_to_unassigned(destroyed, removed);
    destroyed.routes[route_idx].clear();
    finalize_destroyed(destroyed);
    return destroyed;
}

// Remove all but one route
VrpbcState remove_all_but_one_route(const VrpbcState& state, uint64_t seed) {
    VrpbcState destroyed = state.copy();
    std::mt19937 rng(seed);

    std::vector<int> route_indices(destroyed.routes.size());
    std::iota(route_indices.begin(), route_indices.end(), 0);
    std::shuffle(route_indices.begin(), route_indices.end(), rng);
    int keep_idx = route_indices[0];

    std::set<int> removed;
    for (size_t i = 0; i < destroyed.routes.size(); ++i) {
        if ((int)i != keep_idx) {
            removed.insert(destroyed.routes[i].begin(), destroyed.routes[i].end());
        }
    }

    add_to_unassigned(destroyed, removed);
    for (size_t i = 0; i < destroyed.routes.size(); ++i) {
        if ((int)i != keep_idx) {
            destroyed.routes[i].clear();
        }
    }

    finalize_destroyed(destroyed);
    return destroyed;
}

// Invert all routes
VrpbcState invert_routes(const VrpbcState& state, uint64_t /*seed*/) {
    VrpbcState destroyed = state.copy();
    for (auto& route : destroyed.routes) std::reverse(route.begin(), route.end());
    finalize_destroyed(destroyed);
    return destroyed;
}

// 2-opt: reverse a random substring within one route
VrpbcState opt_2(const VrpbcState& state, uint64_t seed, const int MAX_STRING_SIZE) {
    VrpbcState destroyed = state.copy();
    std::mt19937 rng(seed);
    std::uniform_int_distribution<> route_dist(0, destroyed.routes.size() - 1);
    int ridx = route_dist(rng);
    auto& route = destroyed.routes[ridx];
    if (route.empty()) return destroyed;

    int avg_route_size = 0;
    for (const auto& r : destroyed.routes) avg_route_size += r.size();
    avg_route_size = destroyed.routes.empty() ? 1 : avg_route_size / destroyed.routes.size();
    int max_string_size = std::max(MAX_STRING_SIZE, avg_route_size);

    std::uniform_int_distribution<> size_dist(1, std::min((int)route.size(), max_string_size));
    int size = size_dist(rng);
    std::uniform_int_distribution<> cust_dist(0, route.size() - 1);
    int cust_idx = cust_dist(rng);
    std::uniform_int_distribution<> start_dist(0, size - 1);
    int start = cust_idx - start_dist(rng);

    std::vector<int> idcs;
    for (int i = 0; i < size; ++i) {
        idcs.push_back((start + i + route.size()) % route.size());
    }

    if (idcs.front() > idcs.back()) {
        std::vector<int> substring;
        for (int idx : idcs) substring.push_back(route[idx]);
        std::reverse(substring.begin(), substring.end());
        for (size_t i = 0; i < idcs.size(); ++i) route[idcs[i]] = substring[i];
    } else {
        std::reverse(route.begin() + idcs.front(), route.begin() + idcs.back() + 1);
    }

    finalize_destroyed(destroyed);
    return destroyed;
}

// 3-opt
VrpbcState opt_3(const VrpbcState& state, uint64_t seed, const int MAX_STRING_SIZE) {
    VrpbcState destroyed = state.copy();
    std::mt19937 rng(seed);
    std::uniform_int_distribution<> route_dist(0, destroyed.routes.size() - 1);
    int r0 = route_dist(rng), r1 = route_dist(rng);
    auto& route_rm = destroyed.routes[r0];
    auto& route_ins = destroyed.routes[r1];
    if (route_rm.empty() || route_ins.empty()) return destroyed;

    int avg_route_size = 0;
    for (const auto& r : destroyed.routes) avg_route_size += r.size();
    avg_route_size = destroyed.routes.empty() ? 1 : avg_route_size / destroyed.routes.size();
    int max_string_size = std::max(MAX_STRING_SIZE, avg_route_size);

    std::uniform_int_distribution<> size_dist(1, std::min((int)route_rm.size(), max_string_size));
    int size = size_dist(rng);
    std::uniform_int_distribution<> cust_dist(0, route_rm.size() - 1);
    int cust_idx = cust_dist(rng);
    std::uniform_int_distribution<> start_dist(0, size - 1);
    int start = cust_idx - start_dist(rng);

    std::vector<int> idcs;
    for (int i = 0; i < size; ++i) {
        idcs.push_back((start + i + route_rm.size()) % route_rm.size());
    }

    std::vector<int> substring;
    for (int idx : idcs) substring.push_back(route_rm[idx]);

    std::sort(idcs.rbegin(), idcs.rend());
    for (int idx : idcs) route_rm.erase(route_rm.begin() + idx);

    std::uniform_int_distribution<> ins_dist(0, route_ins.size());
    int ins_pos = ins_dist(rng);
    for (size_t i = 0; i < substring.size(); ++i) {
        route_ins.insert(route_ins.begin() + ins_pos + i, substring[i]);
    }

    finalize_destroyed(destroyed);
    return destroyed;
}

// Swap: exchange one node between two routes (nearest neighbor)
VrpbcState swap_destroy_nearest_neighbour(const VrpbcState& state, uint64_t seed) {
    VrpbcState destroyed = state.copy();
    std::mt19937 rng(seed);
    std::uniform_int_distribution<> route_dist(0, destroyed.routes.size() - 1);
    int r0 = route_dist(rng), r1 = route_dist(rng);
    if (destroyed.routes[r0].empty() || destroyed.routes[r1].empty()) return destroyed;

    std::uniform_int_distribution<> idx0_dist(0, destroyed.routes[r0].size() - 1);
    int idx0 = idx0_dist(rng);
    int node_0 = destroyed.routes[r0][idx0];

    int node_1 = -1, idx1 = -1;
    for (int candidate : neighbors(destroyed.distances, node_0)) {
        auto it = std::find(destroyed.routes[r1].begin(), destroyed.routes[r1].end(), candidate);
        if (it != destroyed.routes[r1].end()) {
            node_1 = candidate;
            idx1 = it - destroyed.routes[r1].begin();
            break;
        }
    }
    if (node_1 == -1) return destroyed;

    destroyed.routes[r0][idx0] = node_1;
    destroyed.routes[r1][idx1] = node_0;
    finalize_destroyed(destroyed);
    return destroyed;
}

// Swap: exchange one node between two random routes (random)
VrpbcState swap_destroy_random(const VrpbcState& state, uint64_t seed) {
    VrpbcState destroyed = state.copy();
    std::mt19937 rng(seed);
    std::uniform_int_distribution<> route_dist(0, destroyed.routes.size() - 1);
    int r0 = route_dist(rng), r1 = route_dist(rng);
    if (destroyed.routes[r0].empty() || destroyed.routes[r1].empty()) return destroyed;

    std::uniform_int_distribution<> idx0_dist(0, destroyed.routes[r0].size() - 1);
    std::uniform_int_distribution<> idx1_dist(0, destroyed.routes[r1].size() - 1);
    int idx0 = idx0_dist(rng), idx1 = idx1_dist(rng);
    std::swap(destroyed.routes[r0][idx0], destroyed.routes[r1][idx1]);
    finalize_destroyed(destroyed);
    return destroyed;
}

// Cross-route segment shuffle
VrpbcState cross_route_shuffle(const VrpbcState& state, uint64_t seed, double degree) {
    VrpbcState destroyed = state.copy();
    std::mt19937 rng(seed);

    int n_routes = static_cast<int>(destroyed.routes.size());
    if (n_routes < 2) {
        finalize_destroyed(destroyed);
        return destroyed;
    }

    struct ExtractedSegment {
        int source_route;
        std::vector<int> customers;
    };
    std::vector<ExtractedSegment> segments;
    std::vector<int> all_extracted;

    for (int r = 0; r < n_routes; ++r) {
        auto& route = destroyed.routes[r];
        if (route.size() < 2) continue;

        int seg_size = std::max(1, static_cast<int>(route.size() * degree));
        seg_size = std::min(seg_size, static_cast<int>(route.size()) - 1); // keep at least 1

        std::uniform_int_distribution<> start_dist(0, route.size() - seg_size);
        int start = start_dist(rng);

        std::vector<int> segment(route.begin() + start, route.begin() + start + seg_size);
        route.erase(route.begin() + start, route.begin() + start + seg_size);

        segments.push_back({r, segment});
        all_extracted.insert(all_extracted.end(), segment.begin(), segment.end());
    }

    if (all_extracted.empty()) {
        finalize_destroyed(destroyed);
        return destroyed;
    }

    std::shuffle(all_extracted.begin(), all_extracted.end(), rng);

    std::vector<int> source_route(destroyed.node_coord.size(), -1);
    for (const auto& seg : segments) {
        for (int cust : seg.customers) {
            source_route[cust] = seg.source_route;
        }
    }

    for (int cust : all_extracted) {
        std::vector<int> eligible;
        for (int r = 0; r < n_routes; ++r) {
            if (r != source_route[cust]) {
                eligible.push_back(r);
            }
        }
        if (eligible.empty()) {
            eligible.push_back(source_route[cust]);
        }

        std::uniform_int_distribution<> route_pick(0, eligible.size() - 1);
        int target_route = eligible[route_pick(rng)];

        auto& route = destroyed.routes[target_route];
        std::uniform_int_distribution<> pos_dist(0, route.size());
        int pos = pos_dist(rng);
        route.insert(route.begin() + pos, cust);
    }

    finalize_destroyed(destroyed);
    return destroyed;
}

// ==========================================================================
// LOCAL SEARCH OPERATORS (paired with identity repair in ALNS)
// ==========================================================================

static std::vector<double> ls_compute_events(
    const std::vector<int>& route,
    const std::vector<std::vector<double>>& node_coord
) {
    std::vector<double> evts = {0.0};
    int prev = 0;
    for (int cust : route) {
        double dist_sq = 0.0;
        for (size_t d = 0; d < node_coord[cust].size(); ++d) {
            double diff = node_coord[cust][d] - node_coord[prev][d];
            dist_sq += diff * diff;
        }
        evts.push_back(evts.back() + std::sqrt(dist_sq));
        prev = cust;
    }
    double dist_sq = 0.0;
    for (size_t d = 0; d < node_coord[0].size(); ++d) {
        double diff = node_coord[0][d] - node_coord[prev][d];
        dist_sq += diff * diff;
    }
    evts.push_back(evts.back() + std::sqrt(dist_sq));
    return evts;
}


static double ls_compute_max_d(
    const VrpbcState& state
) {
    int n_vehicles = static_cast<int>(state.routes.size());
    std::vector<std::vector<int>> tours(n_vehicles);
    std::vector<std::vector<double>> events(n_vehicles);
    for (int i = 0; i < n_vehicles; ++i) {
        tours[i] = {0};
        tours[i].insert(tours[i].end(), state.routes[i].begin(), state.routes[i].end());
        tours[i].push_back(0);
        events[i] = ls_compute_events(state.routes[i], state.node_coord);
    }

    double max_d = 0.0;
    for (int i = 0; i < n_vehicles; ++i) {
        for (size_t idx = 0; idx < events[i].size(); ++idx) {
            double e = events[i][idx];
            int node_i = tours[i][idx];
            double max_dist = 0.0;

            for (int j = 0; j < n_vehicles; ++j) {
                if (i == j) continue;
                auto it = std::upper_bound(events[j].begin(), events[j].end(), e);
                double dist;
                if (it == events[j].end()) {
                    double dsq = 0.0;
                    for (size_t d = 0; d < state.node_coord[node_i].size(); ++d) {
                        double diff = state.node_coord[node_i][d] - state.node_coord[0][d];
                        dsq += diff * diff;
                    }
                    dist = std::sqrt(dsq);
                } else {
                    int j_idx = static_cast<int>(it - events[j].begin()) - 1;
                    if (j_idx < 0) j_idx = 0;
                    int max_arc = static_cast<int>(tours[j].size()) - 2;
                    if (j_idx > max_arc) j_idx = max_arc;
                    int from = tours[j][j_idx], to = tours[j][j_idx + 1];
                    double dt = e - events[j][j_idx];
                    const auto& dep = state.node_coord[from];
                    const auto& arr = state.node_coord[to];
                    double norm_sq = 0.0;
                    for (size_t d = 0; d < dep.size(); ++d) {
                        double diff = arr[d] - dep[d];
                        norm_sq += diff * diff;
                    }
                    double norm = std::sqrt(norm_sq);
                    if (norm == 0.0) norm = 1.0;
                    double dsq = 0.0;
                    for (size_t d = 0; d < dep.size(); ++d) {
                        double pos_d = dep[d] + (arr[d] - dep[d]) / norm * dt;
                        double diff = state.node_coord[node_i][d] - pos_d;
                        dsq += diff * diff;
                    }
                    dist = std::sqrt(dsq);
                }
                if (dist > max_dist) max_dist = dist;
            }
            if (max_dist > max_d) max_d = max_dist;
        }
    }
    return max_d;
}

static double ls_route_cost(const std::vector<int>& route, const VrpbcState& state) {
    if (route.empty()) return 0.0;
    double cost = state.distances[0][route[0]];
    for (size_t i = 0; i + 1 < route.size(); ++i)
        cost += state.distances[route[i]][route[i+1]];
    cost += state.distances[route.back()][0];
    return cost;
}

static double ls_total_cost(const VrpbcState& state) {
    double c = 0.0;
    for (const auto& r : state.routes) c += ls_route_cost(r, state);
    return c;
}


VrpbcState relocate_local_search(const VrpbcState& state, uint64_t seed) {
    VrpbcState best = state.copy();
    double best_cost = ls_total_cost(best);
    bool improved = true;

    int max_passes = 3;
    int pass = 0;

    while (improved && pass < max_passes) {
        improved = false;
        pass++;

        for (size_t r_from = 0; r_from < best.routes.size(); ++r_from) {
            for (size_t c_idx = 0; c_idx < best.routes[r_from].size(); ++c_idx) {
                int cust = best.routes[r_from][c_idx];

                for (size_t r_to = 0; r_to < best.routes.size(); ++r_to) {
                    int max_pos = (r_from == r_to)
                        ? static_cast<int>(best.routes[r_to].size()) - 1
                        : static_cast<int>(best.routes[r_to].size());

                    for (int pos = 0; pos <= max_pos; ++pos) {
                        if (r_from == r_to && (pos == static_cast<int>(c_idx) || pos == static_cast<int>(c_idx) + 1))
                            continue;

                        double remove_saving = 0.0;
                        {
                            int pred = (c_idx == 0) ? 0 : best.routes[r_from][c_idx - 1];
                            int succ = (c_idx == best.routes[r_from].size() - 1) ? 0 : best.routes[r_from][c_idx + 1];
                            remove_saving = best.distances[pred][cust] + best.distances[cust][succ] - best.distances[pred][succ];
                        }

                        std::vector<int> target_route = best.routes[r_to];
                        if (r_from == r_to) {
                            target_route.erase(target_route.begin() + c_idx);
                            int adj_pos = pos;
                            if (pos > static_cast<int>(c_idx)) adj_pos--;
                            int pred = (adj_pos == 0) ? 0 : target_route[adj_pos - 1];
                            int succ = (adj_pos == static_cast<int>(target_route.size())) ? 0 : target_route[adj_pos];
                            double insert_add = best.distances[pred][cust] + best.distances[cust][succ] - best.distances[pred][succ];
                            if (insert_add >= remove_saving) continue;
                        } else {
                            int pred = (pos == 0) ? 0 : best.routes[r_to][pos - 1];
                            int succ = (pos == static_cast<int>(best.routes[r_to].size())) ? 0 : best.routes[r_to][pos];
                            double insert_add = best.distances[pred][cust] + best.distances[cust][succ] - best.distances[pred][succ];
                            if (insert_add >= remove_saving) continue;
                        }

                        VrpbcState trial = best.copy();
                        trial.routes[r_from].erase(trial.routes[r_from].begin() + c_idx);
                        int adj_pos = pos;
                        if (r_from == r_to && pos > static_cast<int>(c_idx)) adj_pos--;
                        trial.routes[r_to].insert(trial.routes[r_to].begin() + adj_pos, cust);

                        double new_max_d = ls_compute_max_d(trial);
                        if (new_max_d > trial.D) continue;

                        double new_cost = ls_total_cost(trial);
                        if (new_cost < best_cost - 1e-10) {
                            best = trial;
                            best_cost = new_cost;
                            improved = true;
                            goto next_customer;
                        }
                    }
                }
                next_customer:;
            }
        }
    }

    finalize_destroyed(best);
    return best;
}

// Swap local search
VrpbcState swap_local_search(const VrpbcState& state, uint64_t seed) {
    VrpbcState best = state.copy();
    double best_cost = ls_total_cost(best);
    bool improved = true;
    int max_passes = 2;
    int pass = 0;

    while (improved && pass < max_passes) {
        improved = false;
        pass++;

        for (size_t r0 = 0; r0 < best.routes.size(); ++r0) {
            for (size_t i0 = 0; i0 < best.routes[r0].size(); ++i0) {
                for (size_t r1 = r0 + 1; r1 < best.routes.size(); ++r1) {
                    for (size_t i1 = 0; i1 < best.routes[r1].size(); ++i1) {
                        int c0 = best.routes[r0][i0], c1 = best.routes[r1][i1];
                        int pred0 = (i0 == 0) ? 0 : best.routes[r0][i0-1];
                        int succ0 = (i0 == best.routes[r0].size()-1) ? 0 : best.routes[r0][i0+1];
                        int pred1 = (i1 == 0) ? 0 : best.routes[r1][i1-1];
                        int succ1 = (i1 == best.routes[r1].size()-1) ? 0 : best.routes[r1][i1+1];

                        double old_cost = best.distances[pred0][c0] + best.distances[c0][succ0]
                                        + best.distances[pred1][c1] + best.distances[c1][succ1];
                        double new_cost_local = best.distances[pred0][c1] + best.distances[c1][succ0]
                                              + best.distances[pred1][c0] + best.distances[c0][succ1];
                        if (new_cost_local >= old_cost) continue;

                        VrpbcState trial = best.copy();
                        trial.routes[r0][i0] = c1;
                        trial.routes[r1][i1] = c0;

                        double new_max_d = ls_compute_max_d(trial);
                        if (new_max_d > trial.D) continue;

                        double new_total = ls_total_cost(trial);
                        if (new_total < best_cost - 1e-10) {
                            best = trial;
                            best_cost = new_total;
                            improved = true;
                            goto next_swap;
                        }
                    }
                }
                next_swap:;
            }
        }
    }

    finalize_destroyed(best);
    return best;
}


VrpbcState or_opt_local_search(const VrpbcState& state, uint64_t seed) {
    VrpbcState best = state.copy();
    double best_cost = ls_total_cost(best);
    bool improved = true;
    int max_passes = 2;
    int pass = 0;

    while (improved && pass < max_passes) {
        improved = false;
        pass++;

        for (int seg_size = 1; seg_size <= 3; ++seg_size) {
            for (size_t r_from = 0; r_from < best.routes.size(); ++r_from) {
                if (static_cast<int>(best.routes[r_from].size()) < seg_size) continue;

                for (size_t start = 0; start <= best.routes[r_from].size() - seg_size; ++start) {
                    int pred = (start == 0) ? 0 : best.routes[r_from][start - 1];
                    int succ = (start + seg_size >= best.routes[r_from].size()) ? 0 : best.routes[r_from][start + seg_size];
                    double remove_cost = best.distances[pred][best.routes[r_from][start]];
                    for (int s = 0; s < seg_size - 1; ++s)
                        remove_cost += best.distances[best.routes[r_from][start+s]][best.routes[r_from][start+s+1]];
                    remove_cost += best.distances[best.routes[r_from][start + seg_size - 1]][succ];
                    double reconnect_cost = best.distances[pred][succ];
                    double saving = remove_cost - reconnect_cost;

                    std::vector<int> segment(best.routes[r_from].begin() + start,
                                             best.routes[r_from].begin() + start + seg_size);

                    for (size_t r_to = 0; r_to < best.routes.size(); ++r_to) {
                        if (r_from == r_to) continue;

                        for (size_t pos = 0; pos <= best.routes[r_to].size(); ++pos) {
                            int ins_pred = (pos == 0) ? 0 : best.routes[r_to][pos - 1];
                            int ins_succ = (pos == best.routes[r_to].size()) ? 0 : best.routes[r_to][pos];

                            double insert_cost_val = best.distances[ins_pred][segment[0]];
                            for (int s = 0; s < seg_size - 1; ++s)
                                insert_cost_val += best.distances[segment[s]][segment[s+1]];
                            insert_cost_val += best.distances[segment[seg_size-1]][ins_succ];
                            insert_cost_val -= best.distances[ins_pred][ins_succ];

                            if (insert_cost_val >= saving) continue;

                            VrpbcState trial = best.copy();
                            trial.routes[r_from].erase(trial.routes[r_from].begin() + start,
                                                       trial.routes[r_from].begin() + start + seg_size);
                            trial.routes[r_to].insert(trial.routes[r_to].begin() + pos,
                                                      segment.begin(), segment.end());

                            double new_max_d = ls_compute_max_d(trial);
                            if (new_max_d > trial.D) continue;

                            double new_total = ls_total_cost(trial);
                            if (new_total < best_cost - 1e-10) {
                                best = trial;
                                best_cost = new_total;
                                improved = true;
                                goto next_or_opt;
                            }
                        }
                    }
                    next_or_opt:;
                }
            }
        }
    }

    finalize_destroyed(best);
    return best;
}


// PYBIND11 MODULE
PYBIND11_MODULE(destroy_native, m) {
    m.def("random_removal", &random_removal,
          "Random removal with explicit degree",
          py::arg("state"), py::arg("seed"), py::arg("degree_of_destruction"));
    m.def("random_removal_varying_degree", &random_removal_varying_degree,
          "Random removal with sampled degree (reproducible)",
          py::arg("state"), py::arg("seed"),
          py::arg("lower_bound"), py::arg("mode"), py::arg("upper_bound"));
    m.def("worst_removal", &worst_removal,
          "Worst removal (removes nodes with highest detour cost)",
          py::arg("state"), py::arg("seed"));
    m.def("string_removal", &string_removal,
          "String removal with spatial nearest-neighbor chaining",
          py::arg("state"), py::arg("seed"),
          py::arg("MAX_STRING_REMOVALS"), py::arg("MAX_STRING_SIZE"));
    m.def("string_removal_close_events", &string_removal_close_events,
          "String removal using temporal (event) proximity",
          py::arg("state"), py::arg("seed"),
          py::arg("MAX_STRING_REMOVALS"), py::arg("MAX_STRING_SIZE"));
    m.def("swap_destroy_nearest_neighbour", &swap_destroy_nearest_neighbour,
          "Swap one node between two routes (nearest neighbor)",
          py::arg("state"), py::arg("seed"));
    m.def("swap_destroy_random", &swap_destroy_random,
          "Swap one node between two random routes",
          py::arg("state"), py::arg("seed"));
    m.def("remove_route", &remove_route,
          "Remove all customers from one route",
          py::arg("state"), py::arg("seed"));
    m.def("remove_all_but_one_route", &remove_all_but_one_route,
          "Remove all customers except one route",
          py::arg("state"), py::arg("seed"));
    m.def("invert_routes", &invert_routes,
          "Reverse the visit order of all routes",
          py::arg("state"), py::arg("seed"));
    m.def("opt_2", &opt_2,
          "2-opt: reverse a random substring in one route",
          py::arg("state"), py::arg("seed"), py::arg("MAX_STRING_SIZE"));
    m.def("opt_3", &opt_3,
          "Or-opt: move a random substring between routes",
          py::arg("state"), py::arg("seed"), py::arg("MAX_STRING_SIZE"));
    m.def("cross_route_shuffle", &cross_route_shuffle,
          "Large-neighborhood: extract segments from all routes and redistribute",
          py::arg("state"), py::arg("seed"), py::arg("degree"));
    m.def("relocate_local_search", &relocate_local_search,
          "Local search: relocate customers to cheaper feasible positions",
          py::arg("state"), py::arg("seed"));
    m.def("swap_local_search", &swap_local_search,
          "Local search: swap customer pairs between routes if cheaper and feasible",
          py::arg("state"), py::arg("seed"));
    m.def("or_opt_local_search", &or_opt_local_search,
          "Local search: move segments of 1-3 between routes if cheaper and feasible",
          py::arg("state"), py::arg("seed"));
}
