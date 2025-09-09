#pragma once

#include <algorithm>

#include "openvino/core/shape.hpp"


namespace ov {
namespace reference {

template <typename T>

void ball_query_inference(const T* new_xyz,
                          const T* xyz,
                          int32_t* out, // Output type is int32_t
                          size_t b,
                          size_t n,
                          size_t npoint,
                          float radius,
                          int32_t nsample) {
    const float radius2 = radius * radius;

    // Initialize output to 0
    std::fill(out, out + b * npoint * nsample, 0);

    for (size_t batch_index = 0; batch_index < b; ++batch_index) {
        const T* current_new_xyz = new_xyz + batch_index * npoint * 3;
        const T* current_xyz = xyz + batch_index * n * 3;
        int32_t* current_batch_out = out + batch_index * npoint * nsample;

        for (size_t j = 0; j < npoint; ++j) {
            const T new_x = current_new_xyz[j * 3 + 0];
            const T new_y = current_new_xyz[j * 3 + 1];
            const T new_z = current_new_xyz[j * 3 + 2];
            int32_t cnt = 0;

            for (size_t k = 0; k < n && cnt < nsample; ++k) {
                const T x = current_xyz[k * 3 + 0];
                const T y = current_xyz[k * 3 + 1];
                const T z = current_xyz[k * 3 + 2];
                
                const float dx = static_cast<float>(new_x - x);
                const float dy = static_cast<float>(new_y - y);
                const float dz = static_cast<float>(new_z - z);
                
                const float d2 = dx * dx + dy * dy + dz * dz;

                if (d2 < radius2) {
                    if (cnt == 0) {
                        // If first neighbor found, fill the rest of the slots with its index
                        for (int32_t l = 0; l < nsample; ++l) {
                            current_batch_out[j * nsample + l] = static_cast<int32_t>(k);
                        }
                    }
                    current_batch_out[j * nsample + cnt] = static_cast<int32_t>(k);
                    ++cnt;
                }
            }
        }
    }
}
}  // namespace reference
}  // namespace ov