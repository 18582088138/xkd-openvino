// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather_operation.hpp"
#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/except.hpp"
#include "generate_proposals_shape_inference.hpp"
#include "openvino/reference/gather_operation.hpp"

namespace ov {
namespace op {

namespace gather_operation {

namespace {
std::string get_friendly_type_name(ov::element::Type_t et) {
    switch (et) {
        case ov::element::Type_t::f32: return "float32";
        case ov::element::Type_t::f16: return "float16";
        case ov::element::Type_t::i32: return "int32";
        case ov::element::Type_t::i64: return "int64";
        // Add more cases if needed for other types you support
        default: return "unknown";
    }
}
} 

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& feature,  // (B, C, N)      float32
                             const Tensor& idx,      // (B, npoint)    int32
                             Tensor& out) {          // (B, C, npoint) float32
        using T = typename element_type_traits<ET>::value_type;
        const auto& feature_shape = feature.get_shape();
        const auto& idx_shape = idx.get_shape();

        // std::cout << "[DEBUG] GatherOperation::Evaluate::visit" << std::endl;
        // std::cout << "  ET (enum value): " << static_cast<int>(ET) << std::endl;
        // std::cout << "  T (deduced type): " << get_friendly_type_name(ET) << std::endl;
        // std::cout << "  feature.get_element_type(): " << feature.get_element_type() << std::endl;
        // std::cout << "  feature.get_shape(): " << feature.get_shape() << std::endl;
        // std::cout << "  idx.get_shape(): " << idx.get_shape() << std::endl;
        reference::GatherOperation_Infer<T>(
            feature.data<const T>(),
            idx.data<const int32_t>(),
            out.data<T>(),
            feature_shape[0],
            feature_shape[1],
            feature_shape[2],
            idx_shape[1]
            );
        return true;
    }
};
}  // namespace gather_operation


namespace v0 {

GatherOperation::GatherOperation(const Output<Node>& feature, const Output<Node>& idx)
    : Op({feature, idx}) {
    constructor_validate_and_infer_types();
}

bool GatherOperation::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_GatherOperation_visit_attributes);
    return true;
}

std::shared_ptr<Node> GatherOperation::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_GatherOperation_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<GatherOperation>(new_args.at(0), new_args.at(1));
}

void GatherOperation::validate_and_infer_types() {
    const auto& feature_shape = get_input_partial_shape(0);
    const auto& idx_shape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(this,
        feature_shape.rank().is_static() && feature_shape.rank() == 3,
        "feature input must be 3D: [B, C, N]");

    PartialShape output_shape{feature_shape[0], feature_shape[1], idx_shape[1]};

    set_output_type(0, get_input_element_type(0), output_shape);
}

bool GatherOperation::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_GatherOperation_evaluate);
    OPENVINO_ASSERT(inputs.size() == 2 && outputs.size() == 1);

    const auto& feature = inputs[0];
    const auto& idx = inputs[1];
    auto& out = outputs[0];

    const auto& feature_shape = feature.get_shape();
    const auto& idx_shape = idx.get_shape();

    // Set output shape based on derived idx_count
    out.set_shape(Shape{feature_shape[0], feature_shape[1], idx_shape[1]});
    std::cout <<"[OpenVINO] GatherOperation::evaluate feature.get_element_type(): " << feature.get_element_type() << std::endl;

    // Dispatch to reference implementation
    using namespace element;
    return IF_TYPE_OF(v0_GatherOperation_evaluate,
                      OV_PP_ET_LIST(f32, f16, i32, i64),
                      gather_operation::Evaluate,
                      feature.get_element_type(),
                      feature,
                      idx,
                      out);
}

bool GatherOperation::has_evaluate() const {
    OV_OP_SCOPE(v0_GatherOperation_has_evaluate);
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