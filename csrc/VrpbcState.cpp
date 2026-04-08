#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

class VrpbcState {
public:
  double D;
  std::vector<std::vector<double>> distances;
  std::vector<std::vector<double>> node_coord;

  std::vector<std::vector<int>> routes;
  std::vector<int> unassigned;
  bool feasible;
  std::vector<std::vector<double>> route_distances;
  double max_d;
  double penalty_sum;

  std::vector<std::vector<double>> events;

  bool initialized;
  double penalty_scale;

  VrpbcState() {
    initialized = false;
    max_d = 0.0;
    penalty_sum = 0.0;
    penalty_scale = 1.0;
  }


  VrpbcState(
      double D_,
      py::array_t<double, py::array::c_style | py::array::forcecast> distances_,
      py::array_t<double, py::array::c_style | py::array::forcecast>
          node_coord_,
      std::vector<std::vector<int>> routes_, std::vector<int> unassigned_ = {})
      : D(D_), routes(routes_), unassigned(unassigned_) {

    auto buf = distances_.unchecked<2>();
    distances.resize(buf.shape(0), std::vector<double>(buf.shape(1)));
    for (ssize_t i = 0; i < buf.shape(0); ++i)
      for (ssize_t j = 0; j < buf.shape(1); ++j)
        distances[i][j] = buf(i, j);


    auto nbuf = node_coord_.unchecked<2>();
    node_coord.resize(nbuf.shape(0), std::vector<double>(nbuf.shape(1)));
    for (ssize_t i = 0; i < nbuf.shape(0); ++i)
      for (ssize_t j = 0; j < nbuf.shape(1); ++j)
        node_coord[i][j] = nbuf(i, j);

    initialized = true;

    events_update();
    distances_update();
  }


  VrpbcState(double D_, const std::vector<std::vector<double>> &distances_,
             const std::vector<std::vector<double>> &node_coord_,
             const std::vector<std::vector<int>> &routes_,
             const std::vector<int> &unassigned_ = {})
      : D(D_), distances(distances_), node_coord(node_coord_), routes(routes_),
        unassigned(unassigned_) {
    // compute_lamb();
    initialized = true;
    events_update();
    distances_update();
  }


  VrpbcState copy() const {

    VrpbcState cpy(D, distances, node_coord, routes, unassigned);

    cpy.route_distances = route_distances;
    cpy.max_d = max_d;
    cpy.penalty_sum = penalty_sum;
    cpy.events = events;
    cpy.feasible = feasible;
    cpy.initialized = initialized;
    return cpy;
  }

  double objective() const {
    double routes_cost = 0.0;
    for (const auto &route : routes) {
      routes_cost += route_cost(route);
    }
    if (max_d > D) {
      return unfeasible_penalty() + routes_cost;
    } else {
      return unfeasible_penalty() + routes_cost;
    }
  }

  double unfeasible_penalty() const {
    double penalty_criteria = std::max(0.0, max_d - D);
    // distances_update()
    double penalty_count_criteria = max_d > D ? 1.0 : 0.0;
    // double penalty_count = D * 1e1 * penalty_count_criteria *
    // count_distances_above(D);
    double penalty_count =
        D * 5e1 * penalty_count_criteria * count_distances_above(D);
    // double penalty_max = D * 1e2 * ((max_d > D) ? std::max(max_d,
    // penalty_criteria) : 0.0);
    double penalty_max =
        D * 5e2 * ((max_d > D) ? std::max(max_d, penalty_criteria) : 0.0);
    double penalty_empty =
        D * 1e9 *
        std::count_if(routes.begin(), routes.end(),
                      [](const std::vector<int> &r) { return r.empty(); });
    return penalty_empty + penalty_max + penalty_count;
    // return penalty_empty + penalty_sum;
  }

  std::vector<double> get_context() const {
    // Expanded feature set for ML-based operator selection (XGBoost, contextual
    // bandits, etc.) Total: 30 features organized into 6 categories
    //
    // Feature indices:
    // [0-4]   Solution Quality: objective, routes_cost, penalty, max_distance,
    // feasibility_gap [5-12]  Solution Structure: num_routes, num_unassigned,
    // num_empty_routes, avg_route_length,
    //         min_route_length, max_route_length, route_length_variance,
    //         assigned_customers
    // [13-16] Route Cost Statistics: avg_route_cost, min_route_cost,
    // max_route_cost, route_cost_variance [17-21] Distance/Feasibility:
    // count_above_D, sum_above_D, avg_distance, min_distance, max_distance_all
    // [22-23] Problem Characteristics: total_customers, num_vehicles
    // [24-29] Normalized/Ratio Features: feasibility_ratio, coverage_ratio,
    // route_utilization,
    //         penalty_ratio, empty_route_ratio, distance_variance

    double obj = objective();
    double routes_cost = get_routes_cost();
    double penalty = unfeasible_penalty();
    double max_distance = max_d;
    double feasibility_gap = std::max(0.0, max_d - D);

    int num_routes = static_cast<int>(routes.size());
    int num_unassigned = static_cast<int>(unassigned.size());
    int num_empty_routes = static_cast<int>(
        std::count_if(routes.begin(), routes.end(),
                      [](const std::vector<int> &r) { return r.empty(); }));

    // Route length statistics
    std::vector<int> route_lengths;
    for (const auto &route : routes) {
      route_lengths.push_back(static_cast<int>(route.size()));
    }

    double avg_route_length =
        route_lengths.empty()
            ? 0.0
            : std::accumulate(route_lengths.begin(), route_lengths.end(), 0.0) /
                  route_lengths.size();

    int min_route_length =
        route_lengths.empty()
            ? 0
            : *std::min_element(route_lengths.begin(), route_lengths.end());
    int max_route_length =
        route_lengths.empty()
            ? 0
            : *std::max_element(route_lengths.begin(), route_lengths.end());

    double route_length_variance = 0.0;
    if (!route_lengths.empty() && route_lengths.size() > 1) {
      for (int len : route_lengths) {
        double diff = len - avg_route_length;
        route_length_variance += diff * diff;
      }
      route_length_variance /= route_lengths.size();
    }

    std::vector<double> route_costs;
    for (const auto &route : routes) {
      route_costs.push_back(route_cost(route));
    }

    double avg_route_cost =
        route_costs.empty()
            ? 0.0
            : std::accumulate(route_costs.begin(), route_costs.end(), 0.0) /
                  route_costs.size();

    double min_route_cost =
        route_costs.empty()
            ? 0.0
            : *std::min_element(route_costs.begin(), route_costs.end());
    double max_route_cost =
        route_costs.empty()
            ? 0.0
            : *std::max_element(route_costs.begin(), route_costs.end());

    double route_cost_variance = 0.0;
    if (!route_costs.empty() && route_costs.size() > 1) {
      for (double cost : route_costs) {
        double diff = cost - avg_route_cost;
        route_cost_variance += diff * diff;
      }
      route_cost_variance /= route_costs.size();
    }

    int count_above_D = count_distances_above(D);
    double sum_above_D = sum_distances_above(D);

    std::vector<double> all_distances;
    for (const auto &dist_list : route_distances) {
      all_distances.insert(all_distances.end(), dist_list.begin(),
                           dist_list.end());
    }

    double avg_distance =
        all_distances.empty()
            ? 0.0
            : std::accumulate(all_distances.begin(), all_distances.end(), 0.0) /
                  all_distances.size();

    double min_distance =
        all_distances.empty()
            ? 0.0
            : *std::min_element(all_distances.begin(), all_distances.end());
    double max_distance_all =
        all_distances.empty()
            ? 0.0
            : *std::max_element(all_distances.begin(), all_distances.end());

    double distance_variance = 0.0;
    if (!all_distances.empty() && all_distances.size() > 1) {
      for (double dist : all_distances) {
        double diff = dist - avg_distance;
        distance_variance += diff * diff;
      }
      distance_variance /= all_distances.size();
    }

    int total_customers =
        static_cast<int>(node_coord.size()) - 1;
    int num_vehicles = num_routes;

    double feasibility_ratio = (D > 0.0) ? (max_d / D) : 0.0;
    int assigned_customers = total_customers - num_unassigned;
    double coverage_ratio =
        (total_customers > 0)
            ? (static_cast<double>(assigned_customers) / total_customers)
            : 0.0;
    double route_utilization =
        (num_vehicles > 0)
            ? (static_cast<double>(assigned_customers) / num_vehicles)
            : 0.0;
    double penalty_ratio = (routes_cost > 0.0) ? (penalty / routes_cost) : 0.0;
    double empty_route_ratio =
        (num_vehicles > 0)
            ? (static_cast<double>(num_empty_routes) / num_vehicles)
            : 0.0;

    std::vector<double> context(30);
    size_t idx = 0;

    context[idx++] = obj;
    context[idx++] = routes_cost;
    context[idx++] = penalty;
    context[idx++] = max_distance;
    context[idx++] = feasibility_gap;

    context[idx++] = static_cast<double>(num_routes);
    context[idx++] = static_cast<double>(num_unassigned);
    context[idx++] = static_cast<double>(num_empty_routes);
    context[idx++] = avg_route_length;
    context[idx++] = static_cast<double>(min_route_length);
    context[idx++] = static_cast<double>(max_route_length);
    context[idx++] = route_length_variance;
    context[idx++] = static_cast<double>(assigned_customers);

    context[idx++] = avg_route_cost;
    context[idx++] = min_route_cost;
    context[idx++] = max_route_cost;
    context[idx++] = route_cost_variance;

    context[idx++] = static_cast<double>(count_above_D);
    context[idx++] = sum_above_D;
    context[idx++] = avg_distance;
    context[idx++] = min_distance;
    context[idx++] = max_distance_all;

    context[idx++] = static_cast<double>(total_customers);
    context[idx++] = static_cast<double>(num_vehicles);

    context[idx++] = feasibility_ratio;
    context[idx++] = coverage_ratio;
    context[idx++] = route_utilization;
    context[idx++] = penalty_ratio;
    context[idx++] = empty_route_ratio;
    context[idx++] = distance_variance;

    return context;
  }

  double get_routes_cost() const {
    double total_cost = 0.0;
    for (const auto &route : routes) {
      total_cost += route_cost(route);
    }
    return total_cost;
  }

  // py::array_t<double> get_context() const {
  //     int infeasible_count = count_distances_above(D);
  //     double obj = objective();
  //     memory. std::vector<double> tmp =
  //     {static_cast<double>(infeasible_count), max_d, obj};
  //     py::array_t<double> arr(3);
  //     auto buf = arr.mutable_unchecked<1>();
  //     for (size_t i = 0; i < 3; ++i) buf(i) = tmp[i];
  //     return arr;
  // }

  double max_distance() const { return max_d; }

  double get_D() const { return D; }

  double events_sum() const {
    double sum = 0.0;
    for (const auto &ev : events)
      sum += std::accumulate(ev.begin(), ev.end(), 0.0);
    return sum;
  }

  void events_update() {
    events.clear();
    for (const auto &route : routes) {
      events.push_back(compute_events(route));
    }
    if (!initialized) {
      initialized = true;
    }
  }

  void distances_update() {
    route_distances = compute_distances(routes, events);
    max_d = 0.0;
    for (const auto &sublist : route_distances)
      for (double item : sublist)
        if (item > max_d)
          max_d = item;
    feasible = (max_d <= D);
    double penalty_criteria = std::max(0.0, max_d - D);
    penalty_sum = D * 1e3 * penalty_criteria * sum_distances_above(D);
  }

  // void compute_lamb() {
  //     lamb = std::numeric_limits<double>::lowest();
  //     for (const auto& row : distances) {
  //         for (double val : row) {
  //             if (val > lamb) {
  //                 lamb = val;
  //             }
  //         }
  //     }
  // }

  void ensure_unassigned_consistency() {
    std::set<int> assigned;
    for (const auto &route : routes) {
      for (int cust : route)
        assigned.insert(cust);
    }
    std::set<int> unassigned_set(unassigned.begin(), unassigned.end());
    int n_customers = node_coord.size() - 1; // assuming node 0 is depot
    std::vector<int> all_customers;
    for (int i = 1; i <= n_customers; ++i)
      all_customers.push_back(i);

    for (int cust : all_customers) {
      if (!assigned.count(cust) && !unassigned_set.count(cust)) {
        unassigned.push_back(cust);
      }
    }

    std::set<int> seen;
    std::vector<int> new_unassigned;
    for (int cust : unassigned) {
      if (!assigned.count(cust) && !seen.count(cust)) {
        new_unassigned.push_back(cust);
        seen.insert(cust);
      }
    }
    unassigned = new_unassigned;
  }

  double cost() const { return objective(); }

  std::vector<int> find_route(int customer) const {
    for (const auto &route : routes) {
      if (std::find(route.begin(), route.end(), customer) != route.end())
        return route;
    }
    throw std::runtime_error("Solution does not contain customer " +
                             std::to_string(customer));
  }


  py::array_t<double> get_route_distances() const {
    size_t n_rows = route_distances.size();
    size_t n_cols = route_distances.empty() ? 0 : route_distances[0].size();
    py::array_t<double> arr(
        {static_cast<ssize_t>(n_rows), static_cast<ssize_t>(n_cols)});
    if (n_rows == 0 || n_cols == 0)
      return arr;
    auto buf = arr.mutable_unchecked<2>();
    for (size_t i = 0; i < n_rows; ++i) {
      for (size_t j = 0; j < n_cols; ++j) {
        double v =
            (j < route_distances[i].size()) ? route_distances[i][j] : 0.0;
        buf(i, j) = v;
      }
    }
    return arr;
  }

private:
  double route_cost(const std::vector<int> &route) const {
    std::vector<int> tour = {0};
    tour.insert(tour.end(), route.begin(), route.end());
    tour.push_back(0);
    double cost = 0.0;
    for (size_t idx = 0; idx + 1 < tour.size(); ++idx) {
      cost += distances[tour[idx]][tour[idx + 1]];
    }
    return cost;
  }


  std::vector<double> compute_events(const std::vector<int> &route) const {
    std::vector<double> evts = {0.0};
    std::vector<int> tour = {0};
    tour.insert(tour.end(), route.begin(), route.end());
    tour.push_back(0);
    for (size_t i = 1; i < tour.size(); ++i) {
      evts.push_back(evts.back() + distances[tour[i]][tour[i - 1]]);
    }
    return evts;
  }

  std::vector<std::vector<double>>
  compute_distances(const std::vector<std::vector<int>> &routes,
                    const std::vector<std::vector<double>> &events) const {

    if (node_coord.empty() || distances.empty()) {
      throw py::value_error("node_coord or distances is empty in VrpbcState "
                            "when computing distances.");
    }
    int n_nodes = static_cast<int>(node_coord.size());
    int n_vehicles = static_cast<int>(routes.size());
    std::vector<std::vector<double>> dists(n_vehicles);


    for (int r = 0; r < n_vehicles; ++r) {
      for (int cust : routes[r]) {
        if (cust < 0 || cust >= n_nodes) {
          std::ostringstream oss;
          oss << "Invalid customer index " << cust << " in route " << r
              << " (node_coord size = " << n_nodes << ").";
          throw py::value_error(oss.str());
        }
      }
    }

    for (int i = 0; i < n_vehicles; ++i) {
      std::vector<int> tour_i = {0};
      tour_i.insert(tour_i.end(), routes[i].begin(), routes[i].end());
      tour_i.push_back(0);

      std::vector<double> events_i;
      if (i < static_cast<int>(events.size()) &&
          events[i].size() == tour_i.size()) {
        events_i = events[i];
      } else {
        events_i = compute_events(routes[i]);
      }

      for (size_t idx = 0; idx < events_i.size(); ++idx) {
        double e = events_i[idx];
        double max_dist = 0.0;

        int node_i = tour_i[idx];
        if (node_i < 0 || node_i >= n_nodes) {
          std::ostringstream oss;
          oss << "Invalid node index " << node_i << " referenced in route " << i
              << " at event index " << idx << ".";
          throw py::value_error(oss.str());
        }

        for (int j = 0; j < n_vehicles; ++j) {
          if (i == j)
            continue;

          std::vector<int> tour_j = {0};
          tour_j.insert(tour_j.end(), routes[j].begin(), routes[j].end());
          tour_j.push_back(0);


          std::vector<double> events_j;
          if (j < static_cast<int>(events.size()) &&
              events[j].size() == tour_j.size()) {
            events_j = events[j];
          } else {
            if (tour_j.size() <= 1) {
              auto arc = get_arc(tour_j, 0);
              auto pos = real_position(arc, 0.0);
              double dist_sq = 0.0;
              for (size_t d = 0; d < node_coord[node_i].size(); ++d) {
                double diff = node_coord[node_i][d] - pos[d];
                dist_sq += diff * diff;
              }
              double dist = std::sqrt(dist_sq);
              if (dist > max_dist)
                max_dist = dist;
              continue;
            } else {
              events_j = compute_events(routes[j]);
            }
          }


          int j_event_idx = -1;
          for (size_t k = 0; k < events_j.size(); ++k) {
            if (events_j[k] > e) {
              int candidate = static_cast<int>(k) - 1;
              if (candidate < 0)
                candidate = 0;
              j_event_idx = candidate;
              break;
            }
          }

          if (j_event_idx == -1) {
            double dist_sq = 0.0;
            for (size_t d = 0; d < node_coord[node_i].size(); ++d) {
              double diff = node_coord[node_i][d] - node_coord[0][d];
              dist_sq += diff * diff;
            }
            double dist = std::sqrt(dist_sq);
            if (dist > max_dist)
              max_dist = dist;
            continue;
          }

          int max_arc_idx = std::max(0, static_cast<int>(tour_j.size()) - 2);
          if (j_event_idx < 0)
            j_event_idx = 0;
          if (j_event_idx > max_arc_idx)
            j_event_idx = max_arc_idx;

          auto arc = get_arc(tour_j, j_event_idx);
          if (arc.first < 0 || arc.first >= n_nodes || arc.second < 0 ||
              arc.second >= n_nodes) {
            std::ostringstream oss;
            oss << "Arc endpoints out-of-range for route " << j
                << " arc_idx=" << j_event_idx << " endpoints=(" << arc.first
                << "," << arc.second << "), n_nodes=" << n_nodes << ".";
            throw py::value_error(oss.str());
          }

          double dt_j = e - events_j[j_event_idx];
          double arc_time_j = events_j[j_event_idx + 1] - events_j[j_event_idx];
          auto pos = real_position(arc, dt_j, arc_time_j);

          double dist_sq = 0.0;
          for (size_t d = 0; d < node_coord[node_i].size(); ++d) {
            double diff = node_coord[node_i][d] - pos[d];
            dist_sq += diff * diff;
          }
          double dist = std::sqrt(dist_sq);
          if (dist > max_dist)
            max_dist = dist;
        }

        dists[i].push_back(max_dist);
      }
    }

    return dists;
  }

  std::pair<int, int> get_arc(const std::vector<int> &tour,
                              int last_event_idx) const {
    int max_arc_idx = std::max(0, static_cast<int>(tour.size()) - 2);
    if (last_event_idx < 0)
      last_event_idx = 0;
    if (last_event_idx > max_arc_idx)
      last_event_idx = max_arc_idx;
    return {tour[last_event_idx], tour[last_event_idx + 1]};
  }

  std::vector<double> real_position(const std::pair<int, int> &arc,
                                    double diff_time,
                                    double arc_time = -1.0) const {
    const auto &departure = node_coord[arc.first];
    const auto &arrival = node_coord[arc.second];
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

  int count_distances_above(double D) const {
    int count = 0;
    for (const auto &sublist : route_distances)
      for (double item : sublist)
        if (item > D)
          ++count;
    return count;
  }

  double sum_distances_above(double D) const {
    double sum = 0.0;
    for (const auto &sublist : route_distances)
      for (double item : sublist)
        if (item > D)
          sum += item;
    return sum;
  }
};

PYBIND11_MODULE(vrpbc_state, m) {
  py::class_<VrpbcState>(m, "VrpbcState")

      .def(py::init<double,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    std::vector<std::vector<int>>,
                    std::vector<int>>(),
           py::arg("D"), py::arg("distances"), py::arg("node_coord"),
           py::arg("routes"), py::arg("unassigned") = std::vector<int>())

      .def(py::init<double,
                    const std::vector<std::vector<double>> &,
                    const std::vector<std::vector<double>> &,
                    const std::vector<std::vector<int>> &,
                    const std::vector<int> &>(),
           py::arg("D"), py::arg("distances"), py::arg("node_coord"),
           py::arg("routes"), py::arg("unassigned") = std::vector<int>())
      .def("copy", &VrpbcState::copy)
      .def("objective", &VrpbcState::objective)
      .def("get_routes_cost", &VrpbcState::get_routes_cost)
      .def("get_D", &VrpbcState::get_D)
      .def("get_context", &VrpbcState::get_context)
      .def("max_distance", &VrpbcState::max_distance)
      .def("events_sum", &VrpbcState::events_sum)
      .def("events_update", &VrpbcState::events_update)
      .def("distances_update", &VrpbcState::distances_update)
      .def_property_readonly("cost", &VrpbcState::cost)
      .def("find_route", &VrpbcState::find_route)
      .def_readwrite("routes", &VrpbcState::routes)
      .def_readwrite("distances", &VrpbcState::distances)
      .def_readwrite("unassigned", &VrpbcState::unassigned)
      .def_readwrite("feasible", &VrpbcState::feasible)
      .def_readwrite("max_d", &VrpbcState::max_d)
      .def_readwrite("penalty_sum", &VrpbcState::penalty_sum)
      .def_readwrite("initialized", &VrpbcState::initialized)
      // .def_readwrite("lamb", &VrpbcState::lamb)
      .def("get_route_distances", &VrpbcState::get_route_distances)
      // expose events for debugging/inspection
      .def_readwrite("events", &VrpbcState::events);
}
