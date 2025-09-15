// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/svd.hpp"
#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/except.hpp"
#include "generate_proposals_shape_inference.hpp"
#include "openvino/reference/svd.hpp"

namespace ov {
namespace op {

namespace svd {

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& H,
                             Tensor& U_output,
                             Tensor& S_output,
                             Tensor& V_output) {
        using T = typename element_type_traits<ET>::value_type;        
        reference::SVD_Infer<T>(
            H.data<const T>(),
            U_output.data<T>(),
            S_output.data<T>(),
            V_output.data<T>(),
            H.get_shape()[0],  // b
            H.get_shape()[1],  // m
            H.get_shape()[2]  // n
        );
        return true;
    }
};
}  // namespace svd


namespace v0 {

SVD::SVD(const Output<Node>& H)
    : Op({H}) {
    constructor_validate_and_infer_types();
}

bool SVD::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_SVD_visit_attributes);
    return true;
}

std::shared_ptr<Node> SVD::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_SVD_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<SVD>(new_args.at(0));
}

void SVD::validate_and_infer_types() {
    const auto& svd_shape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(this,
        H_shape.rank().is_static() && H_shape.rank() == 3,
        "Features input must be 3D: [B, X, Y]");

    auto rank = svd_shape.rank().is_static() ? svd_shape.rank().get_length() : 0;
    if (rank < 2) {
        throw std::runtime_error("CustomSVD input must have at least 2 dimensions (batch..., M, N)");
    }
    auto elem_type = get_input_element_type(0);
    
    // batch shape
    std::vector<ov::Dimension> batch_dims;
    for (size_t i = 0; i < rank - 2; ++i) batch_dims.push_back(svd_shape[i]);
    auto m = svd_shape[rank - 2];
    auto n = svd_shape[rank - 1];
    // U: (batch..., M, M), S: (batch..., min(M,N)), V: (batch..., N, N)
    std::vector<ov::Dimension> u_shape = batch_dims; u_shape.push_back(m); u_shape.push_back(m);
    std::vector<ov::Dimension> s_shape = batch_dims;
    // Fix: ov::Dimension does not have min, need to manually check
    if (m.is_static() && n.is_static()) {
        s_shape.push_back(std::min(m.get_length(), n.get_length()));
    } else {
        // When dynamic, use m conservatively
        s_shape.push_back(m);
    }
    std::vector<ov::Dimension> v_shape = batch_dims; v_shape.push_back(n); v_shape.push_back(n);
    set_output_type(0, elem_type, ov::PartialShape(u_shape)); // U
    set_output_type(1, elem_type, ov::PartialShape(s_shape)); // S
    set_output_type(2, elem_type, ov::PartialShape(v_shape)); // V
}


bool SVD::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_SVD_evaluate);
    OPENVINO_ASSERT(inputs.size() == 1 && outputs.size() == 3);

    const auto& input_pshape = get_input_partial_shape(0);
    if (input_pshape.is_dynamic()) {
        return false; 
    }

    const auto& H = inputs[0];
    auto& U_output = outputs[0];
    auto& S_output = outputs[1];
    auto& V_output = outputs[2];

    const auto& h_shape = H.get_shape();

    auto rank = h_shape.rank().is_static() ? h_shape.rank().get_length() : 0;
    if (rank < 2) {
        throw std::runtime_error("CustomSVD input must have at least 2 dimensions (batch..., M, N)");
    }
    auto m = h_shape[rank - 2];
    auto n = h_shape[rank - 1];

    ov::Shape u_shape(h_shape.begin(), h_shape.end() - 2);
    u_shape.insert(u_shape.end(), {m, m});

    ov::Shape s_shape(h_shape.begin(), h_shape.end() - 2);
    s_shape.push_back(std::min(m, n));

    ov::Shape v_shape(h_shape.begin(), h_shape.end() - 2);
    v_shape.insert(v_shape.end(), {n, n});

    U_output.set_shape(u_shape);
    S_output.set_shape(s_shape);
    V_output.set_shape(v_shape);

    // Dispatch to reference implementation
    using namespace element;
    return IF_TYPE_OF(v0_SVD_evaluate,
                      OV_PP_ET_LIST(f32, f16, i32, i64),
                      svd::Evaluate,
                      H.get_element_type(),
                      H,
                      U_output,
                      S_output,
                      V_output);
}

bool SVD::has_evaluate() const {
    OV_OP_SCOPE(v0_SVD_has_evaluate);
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