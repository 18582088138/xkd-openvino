// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/furthest_point_sampling.hpp"

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "utils/common.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector furthest_point_sampling(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 2);

    auto ng_inputs = node.get_ov_inputs();
    auto xyz = ng_inputs[0];
    auto npoint = ng_inputs[1];

    auto furthest_point_sampling_node = std::make_shared<ov::op::v0::FurthestPointSampling>(xyz, npoint);

    return {furthest_point_sampling_node->outputs()};
}

ONNX_OP("FurthestPointSampling", OPSET_RANGE(1, 5), ai_onnx::opset_1::furthest_point_sampling);
}  // namespace opset_1


}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov