// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/ball_query.hpp"
#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/except.hpp"
#include "generate_proposals_shape_inference.hpp"
#include "openvino/reference/ball_query.hpp"

namespace ov {
namespace op {

namespace ball_query {

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& new_xyz,
                             const Tensor& xyz,
                             Tensor& out,
                             float radius,
                             int nsample) { // 注意：添加了 radius 和 nsample
        using T = typename element_type_traits<ET>::value_type;
        reference::ball_query_inference<T>( // 调用正确的 reference 函数
            new_xyz.data<const T>(),
            xyz.data<const T>(),
            out.data<int32_t>(), // 输出类型是 int32_t
            new_xyz.get_shape()[0],  // B
            xyz.get_shape()[1],      // n
            new_xyz.get_shape()[1],  // npoint
            radius,
            nsample
        );
        return true;
    }
};
}  // namespace ball_query


namespace v0 {
BallQuery::BallQuery(const Output<Node>& new_xyz,
                     const Output<Node>& xyz,
                     float radius,
                     int nsample)
    : Op({new_xyz, xyz})
    , m_radius(radius)
    , m_nsample(nsample) {
    constructor_validate_and_infer_types();
}

bool BallQuery::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_BallQuery_visit_attributes);
    visitor.on_attribute("radius", m_radius);
    visitor.on_attribute("nsample", m_nsample);
    return true;
}

std::shared_ptr<Node> BallQuery::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_BallQuery_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BallQuery>(new_args.at(0), new_args.at(1), m_radius, m_nsample);
}

void BallQuery::validate_and_infer_types() {
    const auto& new_xyz_shape = get_input_partial_shape(0);

    PartialShape output_shape{new_xyz_shape[0], new_xyz_shape[1], m_nsample};
    set_output_type(0, element::i32, output_shape);
}

bool BallQuery::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_BallQuery_evaluate);
    OPENVINO_ASSERT(inputs.size() == 2 && outputs.size() == 1);

    const auto& new_xyz = inputs[0];
    const auto& xyz = inputs[1];
    auto& out = outputs[0];

    // Set output shape
    out.set_shape(Shape{
        new_xyz.get_shape()[0],
        new_xyz.get_shape()[1],
        static_cast<size_t>(m_nsample)
    });

    // Dispatch to reference implementation
    using namespace element;
    return IF_TYPE_OF(v0_BallQuery_evaluate,
                      OV_PP_ET_LIST(f32, f16),
                      ball_query::Evaluate,
                      new_xyz.get_element_type(),
                      new_xyz,
                      xyz,
                      out,
                      m_radius,
                      m_nsample);
}

bool BallQuery::has_evaluate() const {
    OV_OP_SCOPE(v0_BallQuery_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::f32:
    case element::f16:
        return true;
    default:
        return false;
    }
}

}  // namespace v0
}  // namespace op
}  // namespace ov
