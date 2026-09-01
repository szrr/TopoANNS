#pragma once

#include <omp.h>

namespace topoanns {

struct GasRvqCompatParams {
    int nt = omp_get_max_threads();
};

}  // namespace topoanns

static topoanns::GasRvqCompatParams _params{};
