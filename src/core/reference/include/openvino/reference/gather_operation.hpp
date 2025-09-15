#pragma once

#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {

template <typename T>
void GatherOperation_Infer(const T* const features,
                        const int32_t* const idx,
                        T* out,
                        size_t b,
                        size_t c,
                        size_t n,
                        size_t npoint) {
    // Clear output buffer safely
    const size_t output_total = b * c * npoint;
    for (size_t t = 0; t < output_total; ++t) {
        out[t] = static_cast<T>(0);
    }

    // Iterate over batch, channels, and points to gather
    for (size_t batch_index = 0; batch_index < b; ++batch_index) {
        const T* batch_features = features + batch_index * c * n;
        const int32_t* batch_idx = idx + batch_index * npoint;
        T* batch_out = out + batch_index * c * npoint;

        for (size_t channel_index = 0; channel_index < c; ++channel_index) {
            for (size_t j = 0; j < npoint; ++j) {
                const int32_t a = batch_idx[j];
                if (a >= 0 && static_cast<size_t>(a) < n) {
                    const T val = batch_features[channel_index * n + static_cast<size_t>(a)];
                    // Sanitize NaN/Inf to zero for stability
                    const float fval = static_cast<float>(val);
                    batch_out[channel_index * npoint + j] = (std::isnan(fval) || std::isinf(fval))
                        ? static_cast<T>(0)
                        : val;
                } else {
                    batch_out[channel_index * npoint + j] = static_cast<T>(0);
                }
            }
        }
    }
}

}  // namespace reference
}  // namespace ov