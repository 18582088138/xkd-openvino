// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

KERNEL(grouping_operation_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* features,   // INPUT0: [B, C, N]  -> 通常 bfyx: B=b, F=c, Y=1, X=n
    const __global INPUT1_TYPE* indices,    // INPUT1: [B, npoint, nsample] -> 可能映射为 F=npoint 或 Y=npoint, X=nsample
    __global OUTPUT_TYPE* out               // OUTPUT0: [B, C, npoint, nsample] -> 通常 bfyx: B=b, F=c, Y=p, X=s
) {
    int global_id = get_global_id(0);

    int B = INPUT0_BATCH_NUM;
    int C = INPUT0_FEATURE_NUM;
    int N = INPUT0_SIZE_Y;
    int NPOINT = INPUT1_FEATURE_NUM;
    int NSAMPLE = INPUT1_SIZE_Y;

    int total = B * C * NPOINT * NSAMPLE;

    if (global_id >= total) return;

    // printf("global_id: %d \n",global_id);

    int batch = global_id / (C * NPOINT * NSAMPLE);
    int channel = (global_id / (NSAMPLE * NPOINT)) % C;
    int point = (global_id / NSAMPLE) % NPOINT;
    int sample = global_id % NSAMPLE;
    
    int idx_offset = batch * (NPOINT * NSAMPLE) + point * NSAMPLE + sample;
    int a = indices[idx_offset];

    float val = 0.0f;

    if (a >= 0 && a < N) {
        int input_offset = batch * (C * N) + channel * N + a;
        val = features[input_offset];

        if (isnan(val) || isinf(val)) {
            val = 0.0f;
        }
    }

    int out_offset = batch * (C * NPOINT * NSAMPLE) + channel * (NPOINT * NSAMPLE) + point * NSAMPLE + sample;
    out[out_offset] = val;

}