#pragma once
#include "utils.hpp"
#include <vector>
#include <limits>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <CL/opencl.hpp>

namespace bitonic {
template <typename ElemType = int>
auto bitonic_sort_cpu(std::vector<ElemType>& sortingData) {
    size_t trueSize = sortingData.size();;
    size_t paddedSize = ocl_utils::closest_pow_of_2(trueSize);
    if (paddedSize > trueSize)
        sortingData.resize(paddedSize, std::numeric_limits<ElemType>::max());

    auto begin = std::chrono::high_resolution_clock::now();
    for (size_t i = 2; i <= paddedSize; i *= 2) {
        for (size_t j = i / 2; j > 0; j /= 2) {
            for (size_t k = 0; k != paddedSize; ++k) {
                size_t currentPartner = k + j;
                if ((k & j) == 0 && currentPartner < paddedSize) {
                    bool isAscending = (k & i) == 0;
                    if (isAscending) {
                        if (sortingData[k] > sortingData[currentPartner])
                            std::swap(sortingData[k], sortingData[currentPartner]);
                    }
                    else {
                        if (sortingData[k] < sortingData[currentPartner])
                            std::swap(sortingData[k], sortingData[currentPartner]);
                    }
                }
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    benchmark::BenchTimes result{};
    result.CPUTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    if (paddedSize > trueSize)
        sortingData.resize(trueSize);

    return result;
}

template <typename ElemType = int>
auto naive_bitonic_sort_gpu(ocl_utils::Environment& env, std::vector<ElemType>& data, const size_t paddedSize,
                                                                                      const size_t bytes) {
    benchmark::BenchTimes result{};

    auto bitonicCall = cl::KernelFunctor<cl::Buffer, int, int>(env.get_kernel());
    auto wall_begin = std::chrono::high_resolution_clock::now();
    auto transfer_begin = std::chrono::high_resolution_clock::now();

    cl::Buffer kernel_buf(env.get_context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, data.data());

    auto transfer_end = std::chrono::high_resolution_clock::now();
    result.TransferTime += std::chrono::duration_cast<std::chrono::nanoseconds>(transfer_end - transfer_begin);

    auto kernel_begin = std::chrono::high_resolution_clock::now();
    for (size_t stage = 2; stage <= paddedSize; stage *= 2) {
        for (size_t step = stage / 2; step > 0; step /= 2) {
            bitonicCall(cl::EnqueueArgs(env.get_queue(), cl::NDRange(paddedSize)),
            kernel_buf,
            stage,
            step);

        }
    }
    env.get_queue().finish();
    auto kernel_end = std::chrono::high_resolution_clock::now();
    result.kernelTime = std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_begin);

    transfer_begin = std::chrono::high_resolution_clock::now();

    env.get_queue().enqueueReadBuffer(kernel_buf, CL_TRUE, 0, bytes, data.data());

    transfer_end = std::chrono::high_resolution_clock::now();
    result.TransferTime += std::chrono::duration_cast<std::chrono::nanoseconds>(transfer_end - transfer_begin);

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.WallTime = std::chrono::duration_cast<std::chrono::nanoseconds>(wall_end - wall_begin);

    return result;
}

template <typename ElemType = int>
auto fast_bitonic_sort_gpu(ocl_utils::Environment& env, std::vector<ElemType>& data, const size_t paddedSize,
                                                                                     const size_t bytes) {
    unsigned int maxDeviceWorkItemSize  = env.get_device().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    unsigned int maxKernelWorkItemSize = env.get_kernel().getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(env.get_device());
    unsigned int maxAllowed = std::min(maxDeviceWorkItemSize, maxKernelWorkItemSize);
    unsigned int localSize = 1;

    ocl_utils::Environment localEnv(config::KERNELS_PATH + config::NAIVE_BITONIC_KERNEL, config::NAIVE_BITONIC_KERNEL_NAME);

    auto bitonicCall = cl::KernelFunctor<cl::Buffer, cl::LocalSpaceArg>(env.get_kernel());
    auto mergeCall   = cl::KernelFunctor<cl::Buffer, int, int>(localEnv.get_kernel());

    while (localSize * 2 <= maxAllowed && localSize * 2 <= paddedSize)
        localSize *= 2;

    benchmark::BenchTimes result{};
    auto wall_begin = std::chrono::high_resolution_clock::now();
    auto transfer_begin = std::chrono::high_resolution_clock::now();

    cl::Buffer kernel_buf(env.get_context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, data.data());

    auto transfer_end = std::chrono::high_resolution_clock::now();
    result.TransferTime += std::chrono::duration_cast<std::chrono::nanoseconds>(transfer_end - transfer_begin);

    auto kernel_begin = std::chrono::high_resolution_clock::now();

    bitonicCall(cl::EnqueueArgs(env.get_queue(), cl::NDRange(paddedSize / 2)),
                kernel_buf,
                cl::Local(2 * localSize * sizeof(ElemType)));

    env.get_queue().finish();
    auto kernel_end = std::chrono::high_resolution_clock::now();
    result.kernelTime += std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_begin);

    if (paddedSize > localSize * 2) {
        kernel_begin = std::chrono::high_resolution_clock::now();
        for (size_t stage = localSize * 4; stage <= paddedSize; stage *= 2) {
            for (size_t step = stage / 2; step > 0; step /= 2) {
                mergeCall(cl::EnqueueArgs(env.get_queue(), cl::NDRange(paddedSize)),
                kernel_buf,
                stage,
                step);
            }
        }
        env.get_queue().finish();
        kernel_end = std::chrono::high_resolution_clock::now();
        result.kernelTime += std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_begin);
    }

    transfer_begin = std::chrono::high_resolution_clock::now();

    env.get_queue().enqueueReadBuffer(kernel_buf, CL_TRUE, 0, bytes, data.data());

    transfer_end = std::chrono::high_resolution_clock::now();
    result.TransferTime += std::chrono::duration_cast<std::chrono::nanoseconds>(transfer_end - transfer_begin);

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.WallTime = std::chrono::duration_cast<std::chrono::nanoseconds>(wall_end - wall_begin);

    return result;
}

template <typename ElemType = int>
auto sort(const ocl_utils::Kernel_Names& currentKernel, std::vector<ElemType>& data) {
    size_t trueSize = data.size();
    size_t paddedSize = ocl_utils::closest_pow_of_2(trueSize);
    size_t bytes = paddedSize * sizeof(ElemType);

    if (paddedSize > trueSize)
        data.resize(paddedSize, std::numeric_limits<ElemType>::max());

    benchmark::BenchTimes result{};

    if (currentKernel == ocl_utils::Kernel_Names::naive) {
        ocl_utils::Environment env(config::KERNELS_PATH + config::NAIVE_BITONIC_KERNEL, config::NAIVE_BITONIC_KERNEL_NAME);
        result = naive_bitonic_sort_gpu(env, data, paddedSize, bytes);
    }
    else if (currentKernel == ocl_utils::Kernel_Names::fast) {
        ocl_utils::Environment env(config::KERNELS_PATH + config::FAST_BITONIC_KERNEL, config::FAST_BITONIC_KERNEL_NAME);
        result = fast_bitonic_sort_gpu(env, data, paddedSize, bytes);
    }
    else {
        throw std::runtime_error("no kernel");
    }

    if (paddedSize > trueSize)
        data.resize(trueSize);

    return result;
}
} // namespace bitonic