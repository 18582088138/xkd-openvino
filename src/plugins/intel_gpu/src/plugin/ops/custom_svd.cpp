// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/custom_svd.hpp"

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/custom_svd.hpp"

namespace ov::intel_gpu {

static void CreateCustomSVDOp(ProgramBuilder& p, const std::shared_ptr<op::v0::CustomSVD>& op) {
    auto inputs = p.GetInputInfo(op);

    const std::string layerName = layer_type_name_ID(op);
    // const cldnn::custom_svd custom_svd_prim(layerName, inputs[0]);
    auto custom_svd_prim = cldnn::custom_svd(layerName, inputs[0]);

    custom_svd_prim.num_outputs = op->get_output_size();

    p.add_primitive(*op, custom_svd_prim);
}

REGISTER_FACTORY_IMPL(v0, CustomSVD);

}  // namespace ov::intel_gpu