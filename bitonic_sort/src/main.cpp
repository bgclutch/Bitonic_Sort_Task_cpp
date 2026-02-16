#include "utils.hpp"
#include "config.hpp"
#include "bitonic_sort.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <CL/opencl.hpp>

int main() {
    ocl_utils::Environment env(config::KERNELS_PATH + config::NAIVE_BITONIC_KERNEL, config::NAIVE_BITONIC_KERNEL_NAME);

    int size;
    std::cin >> size;
    std::vector<int> data(size, 0);

    for (int i = 0; i != size; ++i) {
        std::cin >> data[i];
    }

    bitonic::naive_bitonic_sort_gpu(env, data);

    for (auto num : data)
        std::cout << num << " ";
    std::cout << std::endl;

    return 0;
}