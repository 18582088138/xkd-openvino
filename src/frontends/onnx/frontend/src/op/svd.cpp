// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/svd.hpp"

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "utils/common.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector svd(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 1);

    auto ng_inputs = node.get_ov_inputs();
    auto H  = ng_inputs[0];

    auto svd_node = std::make_shared<ov::op::v0::SVD>(H);

    return {svd_node->outputs()};
}

ONNX_OP("SVD", OPSET_RANGE(1, 5), ai_onnx::opset_1::svd);
}  // namespace opset_1


}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov