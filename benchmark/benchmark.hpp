#pragma once
#include "bitonic_sort.hpp"
#include "utils.hpp"
#include "config.hpp"
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <CL/opencl.hpp>

namespace benchmark {
void printRes(const std::string& nameRes, const BenchTimes& res) {
    std::cout << nameRes << " {\n"
              << "CPU time:      " << res.CPUTime.count() / 1000. << " ms\n"
              << "Wall time:     " << res.WallTime.count() / 1000. << " ms\n"
              << "Kernel time:   " << res.kernelTime.count() / 1000. << " ms\n"
              << "Transfer time: " << res.TransferTime.count() / 1000. << " ms\n}\n";
}

template <typename ElemType>
void getBenchmarkData(std::vector<ElemType>& benchmarkData, std::istream& input_data) {
    int size;
    input_data >> size;

    benchmarkData.reserve(size);

    for (int i = 0; i != size; ++i) {
        ElemType newElem;
        if (!(input_data >> newElem)) {
            throw std::runtime_error("wrong input");
        }
        benchmarkData.emplace_back(newElem);
    }
}

template <typename ElemType>
auto runGPU(const ocl_utils::Kernel_Names& currentKernel, std::vector<ElemType>& data) {
    benchmark::BenchTimes result = bitonic::sort(currentKernel, data);
    return result;
}

template <typename ElemType>
auto runCPU(CPU_TYPE sortType, std::vector<ElemType>& data) {
    benchmark::BenchTimes result{};

    if (sortType == CPU_TYPE::std_sort) {
        auto begin = std::chrono::high_resolution_clock::now();
        std::sort(data.begin(), data.end());
        auto end = std::chrono::high_resolution_clock::now();
        result.CPUTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    }
    else if (sortType == CPU_TYPE::bitonic_sort) {
        result = bitonic::bitonic_sort_cpu(data);
    }
    else {
        throw std::runtime_error("wrong sort called");
    }
    return result;
}
} // namespace benchmark
