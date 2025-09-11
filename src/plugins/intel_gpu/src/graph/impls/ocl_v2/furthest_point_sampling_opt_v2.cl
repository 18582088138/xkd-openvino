// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

#define MAX_WORK_GROUP_SIZE 512
#define VECTOR_SIZE 4

KERNEL(grouping_operation_opt_v2)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* xyz,          // (B, N, 3) 
    const __global INPUT1_TYPE* npoint_i,     // (npoint)
    __global OUTPUT_TYPE* output,             // OUTPUT0: [B, npoint] 
    __global float* temp_buf                  // INTERNAL: [B, N]
){
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int work_group_size = get_local_size(0);
    
    const int B = INPUT0_BATCH_NUM;                      // batch size
    const int N = INPUT0_FEATURE_NUM;                    // number of points
    const int npoint = INPUT1_BATCH_NUM;                 // number of sampled points

    if (group_id >= B) {
        return;
    }

    // Local memory for reduction
    __local float local_dists[MAX_WORK_GROUP_SIZE];
    __local int local_indices[MAX_WORK_GROUP_SIZE];

    uint xyz_offset = group_id * N * 3;
    uint output_offset = group_id * npoint;
    uint temp_offset = group_id * N;

    // Pointers to current batch data
    __global const INPUT0_TYPE* current_dataset = xyz + xyz_offset;
    __global OUTPUT_TYPE* current_idxs = output + output_offset;
    __global float* temp = temp_buf + temp_offset;

    // Vectorized initialization
    for (int i = local_id * VECTOR_SIZE; i < N; i += work_group_size * VECTOR_SIZE) {
        int end_idx = min(i + VECTOR_SIZE, N);
        for (int j = i; j < end_idx; ++j) {
            temp[j] = 1e10f;
        }
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

        // Vectorized distance calculation
        for (int k = local_id * VECTOR_SIZE; k < N; k += work_group_size * VECTOR_SIZE) {
            int end_idx = min(k + VECTOR_SIZE, N);
            for (int idx = k; idx < end_idx; ++idx) {
                float x2 = current_dataset[idx * 3 + 0];
                float y2 = current_dataset[idx * 3 + 1];
                float z2 = current_dataset[idx * 3 + 2];

                float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
                float d2 = fmin(d, temp[idx]);
                temp[idx] = d2;
                
                if (d2 > best) {
                    best = d2;
                    besti = idx;
                }
            }
        }

        // Store local best in local memory
        local_dists[local_id] = best;
        local_indices[local_id] = besti;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Optimized reduction - unroll small loops
        int active_size = work_group_size;
        while (active_size > 1) {
            int half = active_size >> 1;
            if (local_id < half && local_id + half < work_group_size) {
                if (local_dists[local_id + half] > local_dists[local_id]) {
                    local_dists[local_id] = local_dists[local_id + half];
                    local_indices[local_id] = local_indices[local_id + half];
                }
            }
            active_size = half;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Store the result
        if (local_id == 0) {
            current_idxs[j] = local_indices[0];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
