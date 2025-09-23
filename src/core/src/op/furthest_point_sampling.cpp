// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/furthest_point_sampling.hpp"
#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/except.hpp"
#include "generate_proposals_shape_inference.hpp"
#include "openvino/reference/furthest_point_sampling.hpp"

namespace ov {
namespace op {

namespace furthest_point_sampling {

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
    static result_type visit(const Tensor& xyz,
                             const Tensor& npoint,
                             Tensor& out) {
        using T = typename element_type_traits<ET>::value_type;
        // Derive npoint_count from shape if 1D, else from scalar value
        size_t npoint_count = 0;
        const auto& np_shape = npoint.get_shape();
        if (np_shape.size() == 1) {
            npoint_count = np_shape[0];
        } else {
            npoint_count = static_cast<size_t>(*npoint.data<const int32_t>());
        }
        // std::cout << "[DEBUG] FurthestPointSampling::Evaluate::visit" << std::endl;
        // std::cout << "  ET (enum value): " << static_cast<int>(ET) << std::endl;
        // std::cout << "  T (deduced type): " << get_friendly_type_name(ET) << std::endl;
        // std::cout << "  xyz.get_element_type(): " << xyz.get_element_type() << std::endl;
        // std::cout << "  xyz.get_shape(): " << xyz.get_shape() << std::endl;
        // std::cout << "  npoint.get_shape(): " << npoint.get_shape() << std::endl;
        reference::FurthestPointSampling_Infer<T>(
            xyz.data<const T>(),
            static_cast<int32_t>(npoint_count),  
            out.data<int32_t>(),
            xyz.get_shape()[0],
            xyz.get_shape()[1],
            xyz.get_shape()[2]
            );
        return true;
    }
};
}  // namespace furthest_point_sampling


namespace v0 {

FurthestPointSampling::FurthestPointSampling(const Output<Node>& xyz, const Output<Node>& npoint)
    : Op({xyz, npoint}) {
    constructor_validate_and_infer_types();
}

bool FurthestPointSampling::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_FurthestPointSampling_visit_attributes);
    return true;
}

std::shared_ptr<Node> FurthestPointSampling::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_FurthestPointSampling_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<FurthestPointSampling>(new_args.at(0), new_args.at(1));
}

void FurthestPointSampling::validate_and_infer_types() {
    const auto& xyz_shape = get_input_partial_shape(0);
    const auto& npoint_shape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(this,
        xyz_shape.rank().is_static() && xyz_shape.rank() == 3,
        "xyz input must be 3D: [B, N, 3]");

    // [Maybe] We cannot know npoint value at shape-infer time reliably; set dynamic second dim
    // PartialShape output_shape{xyz_shape[0], npoint_shape[0]};
    PartialShape output_shape{xyz_shape[0], Dimension::dynamic()};

    set_output_type(0, ov::element::i32, output_shape);
}

bool FurthestPointSampling::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_FurthestPointSampling_evaluate);
    OPENVINO_ASSERT(inputs.size() == 2 && outputs.size() == 1);

    const auto& xyz = inputs[0];
    const auto& npoint = inputs[1];
    auto& out = outputs[0];

    const auto& xyz_shape = xyz.get_shape();

    // Validate input shapes
    NODE_VALIDATION_CHECK(this,
        xyz_shape.size() == 3,
        "xyz input must be 3D: [B, N, 3]");

    // Validate channel dimension == 3
    NODE_VALIDATION_CHECK(this,
        xyz_shape[2] == 3,
        "xyz last dimension must be 3");

    // Derive npoint_count from npoint shape when rank==1, else from scalar value
    size_t npoint_count = 0;
    const auto& np_shape = npoint.get_shape();
    if (np_shape.size() == 1) {
        npoint_count = np_shape[0];
    } else {
        npoint_count = static_cast<size_t>(*npoint.data<const int32_t>());
    }

    NODE_VALIDATION_CHECK(this,
        npoint_count > 0 && npoint_count <= xyz_shape[1],
        "npoint must be in (0, N]");

    // Set output shape based on derived npoint_count
    out.set_shape(Shape{xyz_shape[0], static_cast<size_t>(npoint_count)});
    std::cout <<"[OpenVINO] FurthestPointSampling::evaluate xyz.get_element_type(): " << xyz.get_element_type() << std::endl;

    // Dispatch to reference implementation
    using namespace element;
    return IF_TYPE_OF(v0_FurthestPointSampling_evaluate,
                      OV_PP_ET_LIST(f32, f16),
                      furthest_point_sampling::Evaluate,
                      xyz.get_element_type(),
                      xyz,
                      npoint,
                      out);
}

bool FurthestPointSampling::has_evaluate() const {
    OV_OP_SCOPE(v0_FurthestPointSampling_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::f32:
    case element::f16:
    // case element::i32:
    // case element::i64:
        return true;
    default:
        return false;
    }
}

}  // namespace v0
}  // namespace op
}  // namespace ov