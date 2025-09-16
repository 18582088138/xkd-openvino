// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/custom_svd.hpp"
#include "registry.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_OCL
    #include "impls/ocl_v2/custom_svd_ref.hpp"
    // #include "impls/ocl_v2/custom_svd_opt.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<custom_svd>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_OCL(ocl::CustomSVDRef, shape_types::any)
        // OV_GPU_CREATE_INSTANCE_OCL(ocl::CustomSVDOpt, shape_types::any)
    };
    return impls;
}

} // namespace ov::intel_gpu