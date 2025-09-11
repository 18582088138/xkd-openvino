// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/furthest_point_sampling.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/furthest_point_sampling.hpp"

namespace ov::intel_gpu {

static void CreateFurthestPointSamplingOp(ProgramBuilder& p, const std::shared_ptr<op::v0::FurthestPointSampling>& op) {
    auto inputs = p.GetInputInfo(op);

    const std::string layerName = layer_type_name_ID(op);
    const cldnn::furthest_point_sampling furthest_point_sampling_prim(layerName, inputs[0], inputs[1]);

    p.add_primitive(*op, furthest_point_sampling_prim);
}

REGISTER_FACTORY_IMPL(v0, FurthestPointSampling);

}  // namespace ov::intel_gpu