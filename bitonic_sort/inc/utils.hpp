#pragma once
#include "config.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <CL/opencl.hpp>

namespace benchmark {
struct BenchTimes {
std::chrono::nanoseconds kernelTime{0};
std::chrono::nanoseconds WallTime{0};
std::chrono::nanoseconds TransferTime{0};
std::chrono::nanoseconds CPUTime{0};
};

enum class CPU_TYPE {
    std_sort,
    bitonic_sort
};
}

namespace ocl_utils {
size_t closest_pow_of_2(const size_t size) noexcept;

enum class Kernel_Names {
    naive,
    fast
};

class Environment final {
    private:
    cl::Platform platform_;
    cl::Device device_;
    cl::Context context_;
    cl::CommandQueue queue_;
    cl::Program program_;
    cl::Kernel kernel_;
    Kernel_Names kernel_name_;

    public:
    Environment(const std::string& kernel_path, const std::string& kernel_name) {
        platform_    = select_platform();
        device_      = select_device(platform_);
        context_     = create_context(device_);
        queue_       = create_queue(context_, device_);
        program_     = create_program(context_, device_, kernel_path);
        kernel_      = create_kernel(program_, kernel_name);
        kernel_name_ = select_kernel_name(kernel_name);
    };

    Environment(const Environment& other, const std::string& kernel_path, const std::string& kernel_name) :
        platform_(other.platform_),
        device_(other.device_),
        context_(other.context_),
        queue_(other.queue_)
    {
        program_     = create_program(context_, device_, kernel_path);
        kernel_      = create_kernel(program_, kernel_name);
        kernel_name_ = select_kernel_name(kernel_name);
    };

    cl::Device& get_device() noexcept {
        return device_;
    }

    cl::Platform& get_platform() noexcept {
        return platform_;
    }

    cl::Context& get_context() noexcept {
        return context_;
    }

    cl::Program& get_program() noexcept {
        return program_;
    }

    cl::CommandQueue& get_queue() noexcept {
        return queue_;
    }

    cl::Kernel& get_kernel() noexcept {
        return kernel_;
    }

    Kernel_Names& get_kernel_name() noexcept {
        return kernel_name_;
    }

    private:
    cl::Platform select_platform() {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        return platforms.front();
    }

    cl::Device select_device(cl::Platform& platform) {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        return devices.front();
    }

    cl::Program create_program(cl::Context& context, cl::Device& device, const std::string& kernel_path) {
        std::ifstream bitonicKernelFile(kernel_path);

        if (!bitonicKernelFile.is_open()) {
            std::cerr << "CRITICAL ERROR: Could not open kernel file!" << std::endl;
            std::cerr << "Looked at: " << kernel_path << std::endl;
            throw std::runtime_error("File not found");
        }

        std::string sourceCode((std::istreambuf_iterator<char>(bitonicKernelFile)), std::istreambuf_iterator<char>());
        cl::Program::Sources sources;
        sources.push_back({sourceCode.c_str(), sourceCode.length()});

        if (sourceCode.empty()) {
            std::cerr << "CRITICAL ERROR: Kernel file is EMPTY!" << std::endl;
            throw std::runtime_error("Empty source code");
        }

        cl::Program program(context, sources);
        if (program.build({device}) != CL_SUCCESS) {
            std::cerr << "Build Log:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            throw std::runtime_error("program wasn't built");
        }

        return program;
    }

    cl::Context create_context(cl::Device& device) {
        cl::Context ret_obj(device);
        return ret_obj;
    }

    cl::CommandQueue create_queue(cl::Context& context, cl::Device& device) {
        cl::CommandQueue ret_obj(context, device);
        return ret_obj;
    }

    cl::Kernel create_kernel(cl::Program& program, const std::string& kernel_name) {
        cl_int err;
        cl::Kernel kernel(program, kernel_name.c_str(), &err);
        if (err != CL_SUCCESS) {
            std::cerr << "GPU Error: " << err << std::endl;
            throw std::runtime_error("no kernel creation");
        }

        if (err == CL_INVALID_KERNEL_NAME) {
            std::cerr << "Check spelling in .cl file!" << std::endl;
        }
        return kernel;
    }

    Kernel_Names select_kernel_name(const std::string& kernel_name) {
        return (kernel_name == config::NAIVE_BITONIC_KERNEL_NAME) ? Kernel_Names::naive : Kernel_Names::fast;
    }
};
} // namespace ocl_utils