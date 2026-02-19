#include "utils.hpp"
#include "config.hpp"
#include "bitonic_sort.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <CL/opencl.hpp>

int main() {
    size_t size;
    std::cin >> size;

    if (!std::cin.good() || size <= 0) {
        std::cerr << "wrong vector size";
        return EXIT_FAILURE;
    }

    std::vector<int> data(size, 0);

    for (size_t i = 0; i != size; ++i) {
        if (!(std::cin >> data[i])) {
            std::cerr << "WRONG GIVEN KEY\n";
            return EXIT_FAILURE;
        }
    }

    try {
        bitonic::sort(ocl_utils::Kernel_Names::fast, data);
    } catch (const std::runtime_error& e) {
        std::cerr << "Standard Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Unknown critical error!" << std::endl;
        return EXIT_FAILURE;
    }

    for (auto num : data)
        std::cout << num << " ";
    std::cout << std::endl;

    return EXIT_SUCCESS;
}