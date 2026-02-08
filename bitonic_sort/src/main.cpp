#include <iostream>
#include <fstream>
#include <vector>
#include <CL/opencl.hpp>

int main() {
    std::ifstream bitonicKernelFile("kernels/kernel_bitonic.cl");
    std::string sourceCode((std::istreambuf_iterator<char>(bitonicKernelFile)), std::istreambuf_iterator<char>());
    cl::Program::Sources sources;
    sources.push_back({sourceCode.c_str(), sourceCode.length()});

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    auto platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    auto device = devices.front();

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    cl::Program program(context, sources);
    if (program.build({device}) != CL_SUCCESS) {
        std::cout << "Build Log:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return 1;
    }

    cl::Kernel kernel(program, "bitonic_sort_kernel");


    int size;
    std::cin >> size;
    std::vector<int> data(size, 0);
    size_t bytes = size * sizeof(int);

    for (int i = 0; i != size; ++i) {
        std::cin >> data[i];
    }

    cl::Buffer d_data(context, CL_MEM_READ_WRITE, bytes);
    queue.enqueueWriteBuffer(d_data, CL_TRUE, 0, bytes, data.data());


    for (int stage = 2; stage <= size; stage *= 2) {
        for (int step = stage / 2; step > 0; step /= 2) {
            kernel.setArg(0, d_data);
            kernel.setArg(1, stage);
            kernel.setArg(2, step);

            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);
        }
    }

    queue.finish();
    queue.enqueueReadBuffer(d_data, CL_TRUE, 0, bytes, data.data());

    for (auto num : data)
        std::cout << num << " ";
    std::cout << std::endl;

    return 0;
}