#pragma once
#include <string>


namespace config {
#ifdef KERNELS_ABS_PATH
    const std::string KERNELS_PATH = KERNELS_ABS_PATH;
#else
    const std::string KERNELS_PATH = "kernels/";
#endif

const std::string NAIVE_BITONIC_KERNEL      = "naive_kernel.cl";
const std::string NAIVE_BITONIC_KERNEL_NAME = "naive_bitonic_sort_kernel";
const std::string FAST_BITONIC_KERNEL       = "fast_kernel.cl";
const std::string FAST_BITONIC_KERNEL_NAME  = "fast_bitonic_sort_kernel";
} // namespace config