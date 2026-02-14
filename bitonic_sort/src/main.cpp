#include "utils.hpp"
#include "config.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <CL/opencl.hpp>

int main() {
    std::cout << config::KERNELS_DIR_PATH + config::NAIVE_BITONIC_KERNEL << " " << config::NAIVE_BITONIC_KERNEL_NAME << std::endl;
    ocl_utils::Environment env(config::KERNELS_DIR_PATH + config::NAIVE_BITONIC_KERNEL, config::NAIVE_BITONIC_KERNEL_NAME);

    int size;
    std::cin >> size;
    std::vector<int> data(size, 0);
    size_t bytes = size * sizeof(int);

    for (int i = 0; i != size; ++i) {
        std::cin >> data[i];
    }

    cl::Buffer d_data(env.get_context(), CL_MEM_READ_WRITE, bytes);
    env.get_queue().enqueueWriteBuffer(d_data, CL_TRUE, 0, bytes, data.data());

    for (int stage = 2; stage <= size; stage *= 2) {
        for (int step = stage / 2; step > 0; step /= 2) {
            env.get_kernel().setArg(0, d_data);
            env.get_kernel().setArg(1, stage);
            env.get_kernel().setArg(2, step);

            env.get_queue().enqueueNDRangeKernel(env.get_kernel(), cl::NullRange, cl::NDRange(size), cl::NullRange);
        }
    }

    env.get_queue().finish();
    env.get_queue().enqueueReadBuffer(d_data, CL_TRUE, 0, bytes, data.data());

    for (auto num : data)
        std::cout << num << " ";
    std::cout << std::endl;

    return 0;
}