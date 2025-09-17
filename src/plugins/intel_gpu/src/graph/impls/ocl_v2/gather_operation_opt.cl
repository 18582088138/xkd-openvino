// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

KERNEL(gather_operation_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* features,     // (B, C, N)
    const __global INPUT1_TYPE* idx,          // (B, NPOINT)
    __global OUTPUT_TYPE* output              // (B, C, NPOINT)
) {
    int gid = get_global_id(0);
    int total_threads = get_global_size(0);

    int B = INPUT0_BATCH_NUM;
    int C = INPUT0_FEATURE_NUM;
    int N = INPUT0_SIZE_Y;
    int NPOINT = INPUT1_FEATURE_NUM;

    int total_elements = B * C * NPOINT;

    // Grid-stride loop: support more threads than elements
    for (int elem = gid; elem < total_elements; elem += total_threads) {
        int j = elem % NPOINT;                 // point index
        int l = (elem / NPOINT) % C;           // channel index
        int i = elem / (C * NPOINT);           // batch index

        int src_idx = idx[i * NPOINT + j];
        float val = 0.0f;

        if (src_idx >= 0 && src_idx < N) {
            int in_offset = i * C * N + l * N + src_idx;
            float in_val = features[in_offset];
            val = (isnan(in_val) || isinf(in_val)) ? 0.0f : in_val;
        }

        int out_offset = i * C * NPOINT + l * NPOINT + j;
        output[out_offset] = val;
    }
}