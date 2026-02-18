#include "utils.hpp"
#include <cstddef>

namespace ocl_utils {
size_t closest_pow_of_2(const size_t size) noexcept {
    size_t power = 1;
    while (size > power)
        power *= 2;

    return power;
}
}