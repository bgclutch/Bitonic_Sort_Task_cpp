#pragma once
#include "utils.hpp"
#include <vector>
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
void naive_bitonic_sort_gpu(ocl_utils::Environment& env, std::vector<ElemType>& data) {
    size_t trueSize = data.size();
    int paddedSize = ocl_utils::closest_pow_of_2(trueSize);
    size_t bytes = paddedSize * sizeof(int);

    if (paddedSize > trueSize)
        data.resize(paddedSize, INT_MAX);

    cl::Buffer kernel_buf(env.get_context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, data.data());

    auto bitonic_algo = cl::KernelFunctor<cl::Buffer, int, int>(env.get_kernel());

    for (int stage = 2; stage <= paddedSize; stage *= 2) {
        for (int step = stage / 2; step > 0; step /=2) {
            bitonic_algo(cl::EnqueueArgs(env.get_queue(), cl::NDRange(paddedSize)), kernel_buf, stage, step);
        }
    }

    env.get_queue().enqueueReadBuffer(kernel_buf, CL_TRUE, 0, bytes, data.data());

    if (paddedSize > trueSize)
        data.resize(trueSize);
}
}