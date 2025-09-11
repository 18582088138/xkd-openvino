// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

#define MAX_WORK_GROUP_SIZE 512
#define VECTOR_SIZE 4
#define TILE_SIZE 2048

KERNEL(grouping_operation_opt_v4)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* xyz,          // (B, N, 3) 
    const __global INPUT1_TYPE* npoint_i,     // (npoint)
    __global OUTPUT_TYPE* output,             // OUTPUT0: [B, npoint] 
    __global float* temp_buf,                 // INTERNAL: [B, N]
    const int work_group_size                 // Dynamic work group size
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
    __local float local_dists[MAX_WORK_GROUP_SIZE];
    __local int local_indices[MAX_WORK_GROUP_SIZE];
    __local float tile_cache[TILE_SIZE * 3];

    uint xyz_offset = group_id * N * 3;
    uint output_offset = group_id * npoint;
    uint temp_offset = group_id * N;

    // Pointers to current batch data
    __global const INPUT0_TYPE* current_dataset = xyz + xyz_offset;
    __global OUTPUT_TYPE* current_idxs = output + output_offset;
    __global float* temp = temp_buf + temp_offset;

    // Vectorized initialization with unrolling
    int vec_loops = N / (work_group_size * VECTOR_SIZE);
    int vec_remainder = N % (work_group_size * VECTOR_SIZE);
    
    for (int v = 0; v < vec_loops; ++v) {
        int base_idx = v * work_group_size * VECTOR_SIZE + local_id * VECTOR_SIZE;
        #pragma unroll
        for (int i = 0; i < VECTOR_SIZE; ++i) {
            temp[base_idx + i] = 1e10f;
        }
    }
    
    // Handle remainder
    int base_idx = vec_loops * work_group_size * VECTOR_SIZE + local_id * VECTOR_SIZE;
    for (int i = 0; i < VECTOR_SIZE && base_idx + i < N; ++i) {
        temp[base_idx + i] = 1e10f;
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

        // Process in tiles for better cache utilization
        for (int tile_start = 0; tile_start < N; tile_start += TILE_SIZE) {
            int tile_end = min(tile_start + TILE_SIZE, N);
            int tile_size = tile_end - tile_start;
            
            // Load tile into local memory with vectorization
            int load_loops = tile_size / (work_group_size * VECTOR_SIZE);
            for (int v = 0; v < load_loops; ++v) {
                int base_idx = v * work_group_size * VECTOR_SIZE + local_id * VECTOR_SIZE;
                int global_base = tile_start + base_idx;
                #pragma unroll
                for (int i = 0; i < VECTOR_SIZE; ++i) {
                    int idx = global_base + i;
                    if (idx < tile_end) {
                        tile_cache[base_idx * 3 + i * 3 + 0] = current_dataset[idx * 3 + 0];
                        tile_cache[base_idx * 3 + i * 3 + 1] = current_dataset[idx * 3 + 1];
                        tile_cache[base_idx * 3 + i * 3 + 2] = current_dataset[idx * 3 + 2];
                    }
                }
            }
            
            // Handle remainder
            int base_idx = load_loops * work_group_size * VECTOR_SIZE + local_id * VECTOR_SIZE;
            int global_base = tile_start + base_idx;
            for (int i = 0; i < VECTOR_SIZE && global_base + i < tile_end; ++i) {
                int idx = global_base + i;
                tile_cache[base_idx * 3 + i * 3 + 0] = current_dataset[idx * 3 + 0];
                tile_cache[base_idx * 3 + i * 3 + 1] = current_dataset[idx * 3 + 1];
                tile_cache[base_idx * 3 + i * 3 + 2] = current_dataset[idx * 3 + 2];
            }
            
            barrier(CLK_LOCAL_MEM_FENCE);

            // Process tile with vectorization
            for (int k = local_id; k < tile_size; k += work_group_size) {
                int global_k = tile_start + k;
                if (global_k < N) {
                    float x2 = tile_cache[k * 3 + 0];
                    float y2 = tile_cache[k * 3 + 1];
                    float z2 = tile_cache[k * 3 + 2];

                    float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
                    float d2 = fmin(d, temp[global_k]);
                    temp[global_k] = d2;
                    
                    if (d2 > best) {
                        best = d2;
                        besti = global_k;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Store local best in local memory
        local_dists[local_id] = best;
        local_indices[local_id] = besti;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Optimized reduction with early exit
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
