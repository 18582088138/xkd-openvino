// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "json_object.h"
#include "gather_operation_inst.h"
#include "primitive_type_base.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(gather_operation);

template <typename ShapeType>
std::vector<layout> gather_operation_inst::calc_output_layouts(const gather_operation_node& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<gather_operation>();
    auto features_layout = impl_param.get_input_layout(0); // (B, C, N)
    auto idx_layout = impl_param.get_input_layout(1);      // (B, NPOINT)

    auto output_type = features_layout.data_type;
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }

    const auto features_ps = features_layout.get_partial_shape(); // (B, C, N)
    const auto idx_ps = idx_layout.get_partial_shape();           // (B, NPOINT)

    ov::PartialShape output_shape = {features_ps[0], idx_ps[1]};

    format output_format = format::adjust_to_rank(features_layout.format, output_shape.size());
    return {layout{output_shape, output_type, output_format}};
}

std::string gather_operation_inst::to_string(const gather_operation_node& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gather_operation_inst::typed_primitive_inst(network& network, const gather_operation_node& node) : parent(network, node) {}

}  // namespace cldnn