#pragma once

#include <vector>

#include <thrust/device_vector.h>

#include "util/vector.hpp"

template<typename T>
void cuda_acc_check(std::vector<T> &gt, thrust::device_vector<T> d_vec) {
    size_t size = gt.size();
    std::vector<float> actual(size);
    thrust::copy(d_vec.begin(), d_vec.end(), actual.begin());
    acc_check(gt, actual);
}
