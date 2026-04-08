#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>


inline std::pair<int, int> get_arc(const std::vector<int>& tour, int last_event_idx) {
    int departure_node = tour[last_event_idx];
    int arrival_node = tour[last_event_idx + 1];
    return {departure_node, arrival_node};
}


inline std::vector<double> real_position(
    const std::vector<std::vector<double>>& node_coord,
    const std::pair<int, int>& arc,
    double diff_time,
    double arc_time = -1.0
) {
    const auto& departure = node_coord[arc.first];
    const auto& arrival = node_coord[arc.second];
    double fraction;
    if (arc_time > 0.0) {
        fraction = diff_time / arc_time;
    } else {
        double norm = 0.0;
        for (size_t d = 0; d < departure.size(); ++d) {
            double diff = arrival[d] - departure[d];
            norm += diff * diff;
        }
        norm = std::sqrt(norm);
        fraction = (norm > 0.0) ? (diff_time / norm) : 0.0;
    }
    if (fraction < 0.0) fraction = 0.0;
    if (fraction > 1.0) fraction = 1.0;
    std::vector<double> pos(departure.size());
    for (size_t d = 0; d < departure.size(); ++d) {
        pos[d] = departure[d] + (arrival[d] - departure[d]) * fraction;
    }
    return pos;
}

// Minimum distance from point c to segment [a, b]
inline double get_point_to_segment_distance(
    const std::vector<double>& a,
    const std::vector<double>& b,
    const std::vector<double>& c
) {
    std::vector<double> v(a.size()), w(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        v[i] = b[i] - a[i];
        w[i] = c[i] - a[i];
    }
    double dot_vv = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    if (dot_vv == 0.0) {
        double dist2 = 0.0;
        for (size_t i = 0; i < w.size(); ++i) dist2 += w[i] * w[i];
        return std::sqrt(dist2);
    }
    double t = std::inner_product(w.begin(), w.end(), v.begin(), 0.0) / dot_vv;
    if (t <= 0.0) {
        double dist2 = 0.0;
        for (size_t i = 0; i < w.size(); ++i) dist2 += w[i] * w[i];
        return std::sqrt(dist2);
    }
    if (t >= 1.0) {
        double dist2 = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            double diff = c[i] - b[i];
            dist2 += diff * diff;
        }
        return std::sqrt(dist2);
    }
    std::vector<double> p(a.size());
    for (size_t i = 0; i < a.size(); ++i) p[i] = a[i] + t * v[i];
    double dist2 = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = c[i] - p[i];
        dist2 += diff * diff;
    }
    return std::sqrt(dist2);
}

// Return the nearest neighbors of the customer, excluding the depot (index 0)
inline std::vector<int> neighbors(
    const std::vector<std::vector<double>>& distances,
    int customer
) {
    size_t n = distances[customer].size();
    std::vector<int> idx(n - 1);
    std::iota(idx.begin(), idx.end(), 1); // skip depot (0)
    std::sort(idx.begin(), idx.end(), [&](int i, int j) {
        return distances[customer][i] < distances[customer][j];
    });
    return idx;
}

inline std::vector<int> neighbors_events(
    const std::vector<std::vector<double>>& events,
    const std::vector<std::vector<int>>& routes,
    const std::vector<std::vector<double>>& distances,
    double D,
    int customer
) {
    int cust_idx = 0;
    int cust_r = 0;
    double cust_event = 0;
    bool found = false;
    for (int r_idx = 0; r_idx < routes.size() && !found; ++r_idx) {
        for (int i = 0; i < routes[r_idx].size() && !found; ++i) {
            if (routes[r_idx][i] == customer) {
                cust_idx = i;
                cust_r = r_idx;
                cust_event = events[cust_r][cust_idx];
                found = true;
            }
        }
    }

    if (!found) return {};

    double time_window = D;

    std::vector<std::pair<int, double>> candidates;

    for (int r_idx = 0; r_idx < routes.size(); ++r_idx) {
        if (r_idx == cust_r) continue;

        for (int idx = 0; idx < routes[r_idx].size(); ++idx) {
            double r_event = events[r_idx][idx];
            double time_diff = std::abs(r_event - cust_event);

            if (time_diff <= time_window) {
                candidates.push_back({routes[r_idx][idx], time_diff});
            }
        }
    }

    for (int idx = 0; idx < routes[cust_r].size(); ++idx) {
        if (idx == cust_idx) continue;
        double r_event = events[cust_r][idx];
        double time_diff = std::abs(r_event - cust_event);

        if (time_diff <= time_window) {
            candidates.push_back({routes[cust_r][idx], time_diff});
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    std::vector<int> neighbors;
    neighbors.reserve(candidates.size());
    for (const auto& cand : candidates) {
        neighbors.push_back(cand.first);
    }

    return neighbors;
}
