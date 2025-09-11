#pragma once

#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {

template <typename T>
void FurthestPointSampling_Infer(const T* const xyz,
                             const int32_t npoint,
                             int32_t* out,
                             size_t b,
                             size_t n,
                             size_t c) {
    // ==================Debug====================

    // std::cout<<"[OpenVINO] FurthestPointSampling_Infer"<<std::endl;
    // std::cout <<"[OpenVINO]  xyz : " << xyz << std::endl;
    // std::cout <<"[OpenVINO] b: "<< b << ",n: " << n << ",c: "<< c <<std::endl;

    // Safe sample prints to verify data without assuming Tensor APIs
    // if (b > 0 && n > 0 && c > 0) {
    //     size_t max_channels_to_print = std::min<size_t>(c, 3);
    //     std::cout <<"[OpenVINO] xyz[0,0,*] (up to 3 channels): ";
    //     for (size_t ch = 0; ch < max_channels_to_print; ++ch) {
    //         std::cout << static_cast<float>(xyz[0 * n * c + 0 * c + ch]) << (ch + 1 < max_channels_to_print ? ", " : "\n");
    //     }
    //     if (n > 1) {
    //         std::cout <<"[OpenVINO] xyz[0,1,*] (up to 3 channels): ";
    //         for (size_t ch = 0; ch < max_channels_to_print; ++ch) {
    //             std::cout << static_cast<float>(xyz[0 * n * c + 1 * c + ch]) << (ch + 1 < max_channels_to_print ? ", " : "\n");
    //         }
    //     }
    // }
    // ==================Debug====================

    for (size_t batch_index = 0; batch_index < b; ++batch_index) {
        // std::cout<<"[OpenVINO] batch_index : "<< batch_index <<std::endl;
        const T* current_dataset = xyz + static_cast<int64_t>(batch_index) * n * c;
        int32_t *current_idxs = out + static_cast<int64_t>(batch_index) * npoint;


        std::vector<float> temp(n, std::numeric_limits<float>::max());

        current_idxs[0] = 0;
        for (int j = 1; j < npoint; ++j) {
            // std::cout<<"[OpenVINO] j : "<< j <<std::endl;
            int besti = 0;
            float best = -std::numeric_limits<float>::max();
            float x1 = static_cast<float>(current_dataset[current_idxs[j - 1] * 3 + 0]);
            float y1 = static_cast<float>(current_dataset[current_idxs[j - 1] * 3 + 1]);
            float z1 = static_cast<float>(current_dataset[current_idxs[j - 1] * 3 + 2]);
            // std::cout<<"[OpenVINO] x1: "<< x1 << ",y1: " << y1 << ",z1: "<< z1 <<std::endl;

            for (size_t k = 0; k < n; ++k) {
                float x2 = static_cast<float>(current_dataset[k * 3 + 0]);
                float y2 = static_cast<float>(current_dataset[k * 3 + 1]);
                float z2 = static_cast<float>(current_dataset[k * 3 + 2]);

                float mag = x2 * x2 + y2 * y2 + z2 * z2;
                if (mag <= 1e-3f) continue;

                float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
                float d2 = std::min(d, temp[static_cast<size_t>(k)]);
                temp[static_cast<size_t>(k)] = d2;
                if (d2 > best) {
                    best = d2;
                    besti = static_cast<int>(k);
                }
            }

            current_idxs[j] = besti;
        }
    }
}

}  // namespace reference
}  // namespace ov