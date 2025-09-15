// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

KERNEL(gather_operation_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* features,     // (B, C, N)
    const __global INPUT1_TYPE* idx,          // (B, NPOINT)
    __global OUTPUT_TYPE* output){
    int gid = get_global_id(0);

    int B = INPUT0_BATCH_NUM;
    int C = INPUT0_FEATURE_NUM;
    int N = INPUT0_SIZE_Y;
    int NPOINT = INPUT1_FEATURE_NUM;
    
    printf("[OV GPU] B=%d, C=%d, N=%d, NPOINT=%d \n", B, C, N, NPOINT);

    int total = B * C * NPOINT;
    if (gid >= total) return;

    // int j = gid % NPOINT;                 // point index within npoints
    // int l = (gid / NPOINT) % C;           // channel index
    // int i = gid / (C * NPOINT);           // batch index

    // printf("[OV GPU] j=%d, l=%d, i=%d \n", j, l, i);

    // int a = idx[i * NPOINT + j];          // gather index from input idx

    // float out_val = 0.0f;
    // if (a >= 0 && a < N) {
		// int input_offset = i * (C * N) + l * N + a;
    //     float in_val = features[input_offset];
    //     // Minimal sanitization
    //     if (isnan(in_val) || isinf(in_val)) {
		// 	out_val = 0.0f;
		// } else {
		// 	out_val = in_val;
		// }
    // }

    // output[i * (C * NPOINT) + l * NPOINT + j] = out_val;
}