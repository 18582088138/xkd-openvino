// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#define MAX_ELEMS 2048

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

KERNEL(grouping_operation_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* xyz,          // (B, N, 3) 
    const __global INPUT1_TYPE* npoint_i,     // (npoint)
    __global OUTPUT_TYPE* output,             // OUTPUT0: [B, npoint] 
    __global float* temp_buf                  // INTERNAL: [B, N]
){
    int global_id = get_global_id(0);
    const int B = INPUT0_BATCH_NUM;                      // batch size
    const int N = INPUT0_FEATURE_NUM;                    // number of points
    const int npoint = INPUT1_BATCH_NUM;                 // number of sampled points (length of input1)
    // printf("[OV GPU] grouping_operation_ref : B=%d, N=%d, npoint=%d \n",B, N, npoint);

    if (global_id >= B) {
        return;
    }

    uint xyz_offset = global_id * N * 3;
    uint output_offset = global_id * npoint;
    uint temp_offset = global_id * N;

    // Pointers to current batch data
    __global const INPUT0_TYPE* current_dataset = xyz + xyz_offset;
    __global OUTPUT_TYPE* current_idxs = output + output_offset;
    __global float* temp = temp_buf + temp_offset;

    // Initialize temp array to 1e10 (matching torch GPU implementation)
    for (int i = 0; i < N; ++i) {
        temp[i] = 1e10f;
    }

    // Initialize first point
    current_idxs[0] = 0;
    for (int j = 1; j < npoint; ++j) {
        int besti = 0;
        float best = -1.0f; // Matching torch GPU implementation
        float x1 = current_dataset[current_idxs[j - 1] * 3 + 0];
        float y1 = current_dataset[current_idxs[j - 1] * 3 + 1];
        float z1 = current_dataset[current_idxs[j - 1] * 3 + 2];

        // Calculate distance for each point and find the farthest point
        for (int k = 0; k < N; ++k) {
            float x2 = current_dataset[k * 3 + 0];
            float y2 = current_dataset[k * 3 + 1];
            float z2 = current_dataset[k * 3 + 2];

            float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
            float d2 = fmin(d, temp[k]);
            temp[k] = d2;
            if (d2 > best) {
                best = d2;
                besti = k;
            }
        }

        // Update the next point to be selected
        current_idxs[j] = besti;
    }
}