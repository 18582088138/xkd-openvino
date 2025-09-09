// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

KERNEL(ball_query_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* new_xyz,    // INPUT0: new_xyz [B, npoint, 3]  -> bfyx: B=b, F=c, Y=1, X=n
    const __global INPUT1_TYPE* xyz,        // INPUT1: xyz     [B, N, 3] -> 可能映射为 F=npoint 或 Y=npoint, 
    __global OUTPUT_TYPE* out               // OUTPUT0: [B, npoint, nsample] -> 通常 bfyx: B=b, F=c, Y=p, X=s
){
    uint batch_index = get_global_id(0);      // batch index
    uint point_index = get_global_id(1);      // point index (npoint)

    int B = INPUT0_BATCH_NUM;
    int NPOINT = INPUT0_FEATURE_NUM;
    int N = INPUT1_FEATURE_NUM;

    if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0 ){
            printf("======== [GPU ov ball_query] ======== \n");
            printf("batch_index: %d, point_index: %d, B: %d, N: %d, NPOINT: %d \n",batch_index, point_index, B, N, NPOINT);
        }
        
    float radius2 = radius * radius;
    int cnt = 0;

    uint inp_1_offset = batch_index * NPOINT * 3;
    uint inp_2_offset = batch_index * N * 3;
    uint output_offset = batch_index * NPOINT * nsample;

    INPUT0_TYPE new_x = new_xyz[inp_1_offset + point_index * 3 + 0];
    INPUT0_TYPE new_y = new_xyz[inp_1_offset + point_index * 3 + 1];
    INPUT0_TYPE new_z = new_xyz[inp_1_offset + point_index * 3 + 2];

    __local INPUT0_TYPE local_xyz[1024 * 3]; 
    int local_size = get_local_size(1);
    int local_id = get_local_id(1);

    for (int l = local_id; l < N; l += local_size) {
        local_xyz[l * 3 + 0] = xyz[inp_2_offset + l * 3 + 0];
        local_xyz[l * 3 + 1] = xyz[inp_2_offset + l * 3 + 1];
        local_xyz[l * 3 + 2] = xyz[inp_2_offset + l * 3 + 2];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k=0; k < N && cnt < nsample; ++k){
        INPUT1_TYPE x = local_xyz[k * 3 + 0];
        INPUT1_TYPE y = local_xyz[k * 3 + 1];
        INPUT1_TYPE z = local_xyz[k * 3 + 2];

        float dist = (new_x - x) * (new_x - x) +
                    (new_y - y) * (new_y - y) +
                    (new_z - z) * (new_z - z);


        if (dist < radius2) {
            if (cnt == 0) {
                for (int l = 0; l < nsample; ++l) {
                    out[output_offset + point_index * nsample + l] = k;
                    }
                }
            out[output_offset+ point_index * nsample + cnt] = k;
            ++cnt;
        }
    }

}