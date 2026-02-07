#pragma once
#include <vector>
#include <algorithm>
#include <iostream>

namespace bitonic {
template <typename ElemType>
void bitonic_sort(std::vector<ElemType>& sortingData) {
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
}