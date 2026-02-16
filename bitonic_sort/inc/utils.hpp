#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <CL/opencl.hpp>

namespace ocl_utils {
    int closest_pow_of_2(const int size) noexcept {
        int power = 1;
        while (size > power)
            power *= 2;

        return power;
    }

    class Environment final {
     private:
        cl::Platform platform_;
        cl::Device device_;
        cl::Context context_;
        cl::CommandQueue queue_;
        cl::Program program_;
        cl::Kernel kernel_;

     public:
        Environment(const std::string& kernel_path, const std::string& kernel_name) {
        platform_ = select_platform();
        device_   = select_device(platform_);
        context_  = create_context(device_);
        queue_    = create_queue(context_, device_);
        program_  = create_program(context_, device_, kernel_path);
        kernel_   = create_kernel(program_, kernel_name);
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
            std::string sourceCode((std::istreambuf_iterator<char>(bitonicKernelFile)), std::istreambuf_iterator<char>());
            cl::Program::Sources sources;
            sources.push_back({sourceCode.c_str(), sourceCode.length()});

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
            cl::Kernel kernel(program, kernel_name.c_str());
            return kernel;
        }
    };
}