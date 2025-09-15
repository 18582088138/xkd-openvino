// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/svd.hpp"
#include "registry.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_OCL
    #include "impls/ocl_v2/svd_ref.hpp"
    // #include "impls/ocl_v2/svd_opt.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<svd>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_OCL(ocl::SVDRef, shape_types::any)
        // OV_GPU_CREATE_INSTANCE_OCL(ocl::SVDOpt, shape_types::any)
    };
    return impls;
}

} // namespace ov::intel_gpu