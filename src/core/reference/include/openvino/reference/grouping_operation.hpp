#pragma once

#include <cmath>
#include <numeric>

#include "openvino/core/shape.hpp"


namespace ov {
namespace reference {

template <typename T>
void grouping(const T* features,
              const int32_t* indices,
              T* out,
              size_t batch_size,
              size_t channels,
              size_t num_points,
              size_t num_selected_points,
              size_t sample_size) {
    for (size_t b = 0; b < batch_size; ++b) {
        // Pointers for current batch
        const T* batch_features = features + b * channels * num_points;
        const int32_t* batch_indices = indices + b * num_selected_points * sample_size;
        T* batch_out = out + b * channels * num_selected_points * sample_size;

        // Initialize output to zero
        for (size_t i = 0; i < channels * num_selected_points * sample_size; ++i) {
            batch_out[i] = T{0};
        }

        // Perform grouping
        for (size_t c = 0; c < channels; ++c) {
            for (size_t np = 0; np < num_selected_points; ++np) {
                for (size_t s = 0; s < sample_size; ++s) {
                    int32_t idx = batch_indices[np * sample_size + s];
                    if (idx >= 0 && idx < static_cast<int32_t>(num_points)) {
                        size_t out_index = (c * num_selected_points + np) * sample_size + s;
                        size_t feat_index = c * num_points + idx;
                        batch_out[out_index] = batch_features[feat_index];
                    }
                }
            }
        }
    }
}
}  // namespace reference
}  // namespace ov