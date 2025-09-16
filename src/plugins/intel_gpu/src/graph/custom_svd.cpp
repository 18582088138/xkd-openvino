// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "json_object.h"
#include "custom_svd_inst.h"
#include "primitive_type_base.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(custom_svd);

template <typename ShapeType>
std::vector<layout> custom_svd_inst::calc_output_layouts(const custom_svd_node& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<custom_svd>();
    auto H_layout = impl_param.get_input_layout(0); // (B, M, N)
    auto output_type = H_layout.data_type;

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }

    const auto H_ps = H_layout.get_partial_shape(); // (B, M, N)

    ov::PartialShape U_output_shape = {H_ps[0], H_ps[1], H_ps[1]};
    ov::PartialShape S_output_shape = {H_ps[0], H_ps[1]};
    ov::PartialShape V_output_shape = {H_ps[0], H_ps[2], H_ps[2]};

    format U_format = format::adjust_to_rank(H_layout.format, U_output_shape.size());
    layout U_layout{U_output_shape, output_type, U_format};

    format S_format = format::adjust_to_rank(H_layout.format, S_output_shape.size()); // Rank 2
    layout S_layout{S_output_shape, output_type, S_format};

    format V_format = format::adjust_to_rank(H_layout.format, V_output_shape.size()); // Rank 3
    layout V_layout{V_output_shape, output_type, V_format};

    return {U_layout, S_layout, V_layout};
    // return {U_layout, V_layout};
}

std::string custom_svd_inst::to_string(const custom_svd_node& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;
    node_info->dump(primitive_description);

    return primitive_description.str();
}

custom_svd_inst::typed_primitive_inst(network& network, const custom_svd_node& node) : parent(network, node) {}

}  // namespace cldnn