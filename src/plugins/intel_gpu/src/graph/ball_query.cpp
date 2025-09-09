// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ball_query_inst.h"
#include "json_object.h"
#include "primitive_type_base.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(ball_query);

template <typename ShapeType>
std::vector<layout> ball_query_inst::calc_output_layouts(const ball_query_node& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<ball_query>();
    auto new_xyz_layout = impl_param.get_input_layout(0);  //  new_xyz (B, npoint, 3)
    auto xyz_layout = impl_param.get_input_layout(1);      //  xyz     (B, N, 3)

    auto output_type = ov::element::i32;
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }

    const auto new_xyz_ps = new_xyz_layout.get_partial_shape();  // (B, npoint, 3)
    const auto xyz_ps = xyz_layout.get_partial_shape();          // (B, N, 3)

    ov::PartialShape output_shape = {new_xyz_ps[0], new_xyz_ps[1], desc->nsample};

    format output_format = format::adjust_to_rank(new_xyz_layout.format, output_shape.size());
    return {layout{output_shape, output_type, output_format}};
}

std::string ball_query_inst::to_string(const ball_query_node& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;
    node_info->dump(primitive_description);

    return primitive_description.str();
}

ball_query_inst::typed_primitive_inst(network& network, const ball_query_node& node) : parent(network, node) {}

}  // namespace cldnn