// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/furthest_point_sampling.hpp"
#include "registry.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_OCL
    #include "impls/ocl_v2/furthest_point_sampling_ref.hpp"
    #include "impls/ocl_v2/furthest_point_sampling_opt.hpp"
    // #include "impls/ocl_v2/furthest_point_sampling_opt_v2.hpp"
    // #include "impls/ocl_v2/furthest_point_sampling_opt_v3.hpp"
    // #include "impls/ocl_v2/furthest_point_sampling_bfyx_opt.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<furthest_point_sampling>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        // OV_GPU_CREATE_INSTANCE_OCL(ocl::FurthestPointSamplingOptV3, shape_types::any)
        // OV_GPU_CREATE_INSTANCE_OCL(ocl::FurthestPointSamplingOptV2, shape_types::any)
        OV_GPU_CREATE_INSTANCE_OCL(ocl::FurthestPointSamplingOpt, shape_types::any)
        OV_GPU_CREATE_INSTANCE_OCL(ocl::FurthestPointSamplingRef, shape_types::any)
        // OV_GPU_CREATE_INSTANCE_OCL(ocl::FurthestPointSamplingBfyxOpt, shape_types::any)
    };
    return impls;
}

} // namespace ov::intel_gpu