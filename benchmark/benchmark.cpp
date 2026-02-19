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

    try {
//-------------------------------naive kernel benchmark--------------------------------------//
    std::vector<int> naiveData = data;
    auto resultNaive = benchmark::runGPU(ocl_utils::Kernel_Names::naive, naiveData);
    if (!std::is_sorted(naiveData.begin(), naiveData.end())) {
        std::cerr << "Naive GPU FAILED to sort!" << std::endl;
    }
    benchmark::printRes("naive GPU", resultNaive);

//-------------------------------fast kernel benchmark---------------------------------------//
    std::vector<int> fastData = data;
    auto resultFast = benchmark::runGPU(ocl_utils::Kernel_Names::fast, fastData);
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
    } catch (const std::runtime_error& e) {
        std::cerr << "Standard Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Unknown critical error!" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}