// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

#define WORK_GROUP_SIZE 256
#define MAX_LOCAL_POINTS 8192

KERNEL(furthest_point_sampling_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* xyz,          // (B, N, 3) 
    const __global INPUT1_TYPE* npoint_i,     // (npoint)
    __global OUTPUT_TYPE* output,             // OUTPUT0: [B, npoint] 
    __global float* temp_buf                  // INTERNAL: [B, N]
){
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    
    const int B = INPUT0_BATCH_NUM;                      // batch size
    const int N = INPUT0_FEATURE_NUM;                    // number of points
    const int npoint = INPUT1_BATCH_NUM;                 // number of sampled points

    if (group_id >= B) {
        return;
    }

    // Local memory for reduction
    __local float local_dists[WORK_GROUP_SIZE];
    __local int local_indices[WORK_GROUP_SIZE];

    uint xyz_offset = group_id * N * 3;
    uint output_offset = group_id * npoint;
    uint temp_offset = group_id * N;

    // Pointers to current batch data
    __global const INPUT0_TYPE* current_dataset = xyz + xyz_offset;
    __global OUTPUT_TYPE* current_idxs = output + output_offset;
    __global float* temp = temp_buf + temp_offset;

    // Initialize temp array to 1e10
    for (int i = local_id; i < N; i += WORK_GROUP_SIZE) {
        temp[i] = 1e10f;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Initialize first point
    if (local_id == 0) {
        current_idxs[0] = 0;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int j = 1; j < npoint; ++j) {
        float best = -1.0f;
        int besti = 0;
        
        // Get coordinates of the last selected point
        float x1 = current_dataset[current_idxs[j - 1] * 3 + 0];
        float y1 = current_dataset[current_idxs[j - 1] * 3 + 1];
        float z1 = current_dataset[current_idxs[j - 1] * 3 + 2];

        // Each work-item processes a subset of points
        for (int k = local_id; k < N; k += WORK_GROUP_SIZE) {
            float x2 = current_dataset[k * 3 + 0];
            float y2 = current_dataset[k * 3 + 1];
            float z2 = current_dataset[k * 3 + 2];

            float mag = x2 * x2 + y2 * y2 + z2 * z2;
            if (mag <= 1e-3f) continue;

            float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
            float d2 = fmin(d, temp[k]);
            temp[k] = d2;
            
            if (d2 > best) {
                best = d2;
                besti = k;
            }
        }

        // Store local best in local memory
        local_dists[local_id] = best;
        local_indices[local_id] = besti;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Reduction to find global best
        for (int stride = WORK_GROUP_SIZE / 2; stride > 0; stride >>= 1) {
            if (local_id < stride) {
                if (local_dists[local_id + stride] > local_dists[local_id]) {
                    local_dists[local_id] = local_dists[local_id + stride];
                    local_indices[local_id] = local_indices[local_id + stride];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Store the result
        if (local_id == 0) {
            current_idxs[j] = local_indices[0];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
