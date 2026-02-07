#include "bitonic_sort.hpp"
#include <iostream>
#include <vector>

int main() {
    int size;
    std::cin >> size;
    std::vector<int> data(size, 0);

    for (int i = 0; i != size; ++i) {
        std::cin >> data[i];
    }

    bitonic::bitonic_sort(data);

    for (auto num : data)
        std::cout << num << " ";
    std::cout << std::endl;

    return 0;
}