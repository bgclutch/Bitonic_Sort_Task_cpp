#include "bitonic_sort.hpp"
#include "benchmark.hpp"
#include "utils.hpp"
#include "config.hpp"
#include <iostream>
#include <fstream>
#include <set>
#include <vector>
#include <chrono>
#include <algorithm>
#include <CL/opencl.hpp>

int main(int argc, char** argv) {
    std::ifstream input_data;

    if (argc > 1) {
        input_data.open(argv[1]);
        std::cout << "benchmark data loaded from " << argv[1] << std::endl;
    }
    else {
        input_data.open("benchmark/default_benchmark.in");
        std::cout << "default benchmark data loaded" << std::endl;
    }

    if (!input_data.is_open()) {
        std::cerr << "Error opening input_data\n";
        return EXIT_FAILURE;
    }

    std::vector<int> data;
    benchmark::getBenchmarkData(data, input_data);

//-------------------------------naive kernel benchmark--------------------------------------//
    ocl_utils::Environment naive_env(config::KERNELS_PATH + config::NAIVE_BITONIC_KERNEL, config::NAIVE_BITONIC_KERNEL_NAME);
    std::vector<int> naiveData = data;
    auto resultNaive = benchmark::runGPU(naive_env, naiveData);
    if (!std::is_sorted(naiveData.begin(), naiveData.end())) {
        std::cerr << "Naive GPU FAILED to sort!" << std::endl;
    }
    benchmark::printRes("naive GPU", resultNaive);

//-------------------------------fast kernel benchmark---------------------------------------//
    ocl_utils::Environment fast_env(config::KERNELS_PATH + config::FAST_BITONIC_KERNEL, config::FAST_BITONIC_KERNEL_NAME);
    std::vector<int> fastData = data;
    auto resultFast = benchmark::runGPU(fast_env, fastData);
    if (!std::is_sorted(naiveData.begin(), naiveData.end())) {
        std::cerr << "Fast GPU FAILED to sort!" << std::endl;
    }
    benchmark::printRes("fast GPU", resultFast);

//-------------------------------bitonic on cpu benchmark------------------------------------//
    std::vector<int> bitonicData = data;
    auto resultBitonic = benchmark::runCPU(benchmark::CPU_TYPE::bitonic_sort, bitonicData);
    benchmark::printRes("bitonic CPU", resultBitonic);

//-------------------------------std::sort benchmark-----------------------------------------//
    std::vector<int> stdData = data;
    auto resultStd = benchmark::runCPU(benchmark::CPU_TYPE::std_sort, stdData);
    benchmark::printRes("std::sort", resultStd);

    return EXIT_SUCCESS;
}