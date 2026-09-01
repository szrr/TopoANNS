#pragma once

#include <chrono>
#include <cstdint>

using idx_t = int;
using num_t = int;
using val_t = float;
using dis_t = float;
using linklistsize_t = unsigned int;

struct Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_point,
        end_point;

    void Start() { start_point = std::chrono::high_resolution_clock::now(); }

    void Stop() { end_point = std::chrono::high_resolution_clock::now(); }

    double DurationInMilliseconds() {
        std::chrono::duration<double, std::milli> duration =
            end_point - start_point;
        return duration.count();
    }
};

enum SelectionStrategy { Simple = 0, Heuristic = 1 };

enum EntryStrategy { Fixed = 0, Prior = 1, Random = 2 };

//* serch time in ms
// struct HNSWSearchTime {
//     //* time of finding entry
//     double upper_time{0.0};
//     //* time of seaching in base layer
//     double base_time{0.0};

//     double GetSearchTime() { return upper_time + base_time; }
// };

struct SearchTime {
    //* time of finding entry
    double entry_time{0.0};
    //* time of seaching in base layer
    double beam_search_time{0.0};

    double total_time{0.0};
    SearchTime(double s, double e, double d) : entry_time(s), beam_search_time(e), total_time(d) {}
    void SumTime() { total_time = entry_time + beam_search_time; }
};

//* all times are in milliseconds
struct SearchMetric {
    // bool collect_metrics{false};
    // // construction
    // double construction_time{0.0};
    // search
    SearchTime search_time{0.0, 0.0, 0.0};
    size_t search_hops{0};
    size_t distance_computations{0};
    // entry of base layer
    idx_t entry_node;
};