// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/ball_query.hpp"

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "utils/common.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector ball_query(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 2);

    auto ng_inputs = node.get_ov_inputs();
    auto new_xyz = ng_inputs[0];
    auto xyz = ng_inputs[1];

    float radius = node.get_attribute_value<float>("radius", 0.1f); 
    int nsample = node.get_attribute_value<int>("nsample", 64); 

    auto ball_query = std::make_shared<ov::op::v0::BallQuery>(new_xyz, xyz, radius, nsample);

    return {ball_query->outputs()};
}

ONNX_OP("BallQuery", OPSET_RANGE(1, 5), ai_onnx::opset_1::ball_query);
}  // namespace opset_1


}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
