// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/gather_operation.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/gather_operation.hpp"

namespace ov::intel_gpu {

static void CreateGatherOperationOp(ProgramBuilder& p, const std::shared_ptr<op::v0::GatherOperation>& op) {
    auto inputs = p.GetInputInfo(op);

    const std::string layerName = layer_type_name_ID(op);
    const cldnn::gather_operation gather_operation_prim(layerName, inputs[0], inputs[1]);

    p.add_primitive(*op, gather_operation_prim);
}

REGISTER_FACTORY_IMPL(v0, GatherOperation);

}  // namespace ov::intel_gpu