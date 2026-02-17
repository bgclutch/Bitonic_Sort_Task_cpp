#pragma once
#include "utils.hpp"
#include <vector>
#include <limits>
#include <algorithm>
#include <iostream>
#include <CL/opencl.hpp>

namespace bitonic {
template <typename ElemType = int>
void bitonic_sort_cpu(std::vector<ElemType>& sortingData) {
    size_t size = sortingData.size();
    for (size_t i = 2; i <= size; i *= 2) {
        for (size_t j = i / 2; j > 0; j /= 2) {
            for (size_t k = 0; k != size; ++k) {
                size_t currentPartner = k + j;
                if ((k & j) == 0 && currentPartner < size) {
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
}

template <typename ElemType = int>
void naive_bitonic_sort_gpu(ocl_utils::Environment& env, std::vector<ElemType>& data, const size_t paddedSize, const size_t bytes) {
    cl::Buffer kernel_buf(env.get_context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, data.data());
    auto bitonicCall = cl::KernelFunctor<cl::Buffer, int, int>(env.get_kernel());

    for (int stage = 2; stage <= paddedSize; stage *= 2) {
        for (int step = stage / 2; step > 0; step /=2) {
            bitonicCall(cl::EnqueueArgs(env.get_queue(), cl::NDRange(paddedSize)),
            kernel_buf,
            stage,
            step);
        }
    }

    env.get_queue().enqueueReadBuffer(kernel_buf, CL_TRUE, 0, bytes, data.data());
}

template <typename ElemType = int>
void fast_bitonic_sort_gpu(ocl_utils::Environment& env, std::vector<ElemType>& data, const size_t paddedSize, const size_t bytes) {
    int maxDeviceWorkItemSize  = env.get_device().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    int maxKernelWorkItemSize = env.get_kernel().getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(env.get_device());
    int maxAllowed = std::min(maxDeviceWorkItemSize, maxKernelWorkItemSize);
    int localSize = 1;

    while (localSize * 2 <= maxAllowed && localSize * 2 <= paddedSize)
        localSize *= 2;

    cl::Buffer kernel_buf(env.get_context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, data.data());

    auto bitonicCall = cl::KernelFunctor<cl::Buffer, cl::LocalSpaceArg, int>(env.get_kernel());

    bitonicCall(cl::EnqueueArgs(env.get_queue(), cl::NDRange(paddedSize), cl::NDRange(localSize)),
                kernel_buf,
                cl::Local(localSize * sizeof(ElemType)),
                localSize);

    env.get_queue().enqueueReadBuffer(kernel_buf, CL_TRUE, 0, bytes, data.data());
}

template <typename ElemType = int>
void sort(ocl_utils::Environment& env, std::vector<ElemType>& data) {
    size_t trueSize = data.size();
    int paddedSize = ocl_utils::closest_pow_of_2(trueSize);
    size_t bytes = paddedSize * sizeof(ElemType);

    if (paddedSize > trueSize)
        data.resize(paddedSize, std::numeric_limits<ElemType>::max());

    auto currentKernel = env.get_kernel_name();

    if (currentKernel == ocl_utils::Kernel_Names::naive)
        naive_bitonic_sort_gpu(env, data, paddedSize, bytes);
    else if (currentKernel == ocl_utils::Kernel_Names::fast)
        fast_bitonic_sort_gpu(env, data, paddedSize, bytes);
    else
        throw std::runtime_error("no kernel");

    if (paddedSize > trueSize)
        data.resize(trueSize);
}
}