// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/svd.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/svd.hpp"

namespace ov::intel_gpu {

static void CreateSVDOp(ProgramBuilder& p, const std::shared_ptr<op::v0::SVD>& op) {
    auto inputs = p.GetInputInfo(op);

    const std::string layerName = layer_type_name_ID(op);
    const cldnn::svd svd_prim(layerName, inputs[0]);

    p.add_primitive(*op, svd_prim);
}

REGISTER_FACTORY_IMPL(v0, SVD);

}  // namespace ov::intel_gpu