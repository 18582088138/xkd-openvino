// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/grouping_operation.hpp"

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "utils/common.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector grouping_operation(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 2);

    auto ng_inputs = node.get_ov_inputs();
    auto features = ng_inputs[0];
    auto indices = ng_inputs[1];

    auto grouping = std::make_shared<ov::op::v0::GroupingOperation>(features, indices);

    return {grouping->outputs()};
}

ONNX_OP("GroupingOperation", OPSET_RANGE(1, 5), ai_onnx::opset_1::grouping_operation);
}  // namespace opset_1


}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
