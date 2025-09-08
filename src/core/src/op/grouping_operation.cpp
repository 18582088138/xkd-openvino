// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/grouping_operation.hpp"
#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/except.hpp"
#include "generate_proposals_shape_inference.hpp"
#include "openvino/reference/grouping_operation.hpp"

namespace ov {
namespace op {

namespace grouping_operation {

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& features,
                             const Tensor& indices,
                             Tensor& out) {
        using T = typename element_type_traits<ET>::value_type;
        reference::grouping<T>(
            features.data<const T>(),
            indices.data<const int32_t>(),
            out.data<T>(),
            features.get_shape()[0],  // B
            features.get_shape()[1],  // C
            features.get_shape()[2],  // N
            indices.get_shape()[1],   // npoint
            indices.get_shape()[2]    // nsample
        );
        return true;
    }
};
}  // namespace grouping_operation


namespace v0 {

GroupingOperation::GroupingOperation(const Output<Node>& features, const Output<Node>& indices)
    : Op({features, indices}) {
    constructor_validate_and_infer_types();
}

bool GroupingOperation::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_GroupingOperation_visit_attributes);
    return true;
}

std::shared_ptr<Node> GroupingOperation::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_GroupingOperation_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<GroupingOperation>(new_args.at(0), new_args.at(1));
}

void GroupingOperation::validate_and_infer_types() {
    const auto& features_shape = get_input_partial_shape(0);
    const auto& indices_shape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(this,
        features_shape.rank().is_static() && features_shape.rank() == 3,
        "Features input must be 3D: [B, C, N]");

    NODE_VALIDATION_CHECK(this,
        indices_shape.rank().is_static() && indices_shape.rank() == 3,
        "Indices input must be 3D: [B, npoint, nsample]");

    PartialShape output_shape{
        features_shape[0], features_shape[1], indices_shape[1], indices_shape[2]
    };
    set_output_type(0, get_input_element_type(0), output_shape);
}

bool GroupingOperation::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_GroupingOperation_evaluate);
    OPENVINO_ASSERT(inputs.size() == 2 && outputs.size() == 1);

    const auto& features = inputs[0];
    const auto& indices = inputs[1];
    auto& out = outputs[0];

    const auto& f_shape = features.get_shape();
    const auto& idx_shape = indices.get_shape();

    // Validate input shapes
    NODE_VALIDATION_CHECK(this,
        f_shape.size() == 3,
        "Features input must be 3D: [B, C, N]");
    NODE_VALIDATION_CHECK(this,
        idx_shape.size() == 3,
        "Indices input must be 3D: [B, npoint, nsample]");
    NODE_VALIDATION_CHECK(this,
        f_shape[0] == idx_shape[0],
        "Batch size mismatch between features and indices");
    NODE_VALIDATION_CHECK(this,
        indices.get_element_type() == element::i32,
        "Indices must be of type int32");

    // Set output shape
    out.set_shape(Shape{f_shape[0], f_shape[1], idx_shape[1], idx_shape[2]});

    // Dispatch to reference implementation
    using namespace element;
    return IF_TYPE_OF(v0_GroupingOperation_evaluate,
                      OV_PP_ET_LIST(f32, f16, i32, i64),
                      grouping_operation::Evaluate,
                      features.get_element_type(),
                      features,
                      indices,
                      out);
}

bool GroupingOperation::has_evaluate() const {
    OV_OP_SCOPE(v0_GroupingOperation_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::f32:
    case element::f16:
    case element::i32:
    case element::i64:
        return true;
    default:
        return false;
    }
}

}  // namespace v0
}  // namespace op
}  // namespace ov
