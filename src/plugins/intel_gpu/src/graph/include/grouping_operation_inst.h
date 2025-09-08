// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/grouping_operation.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<grouping_operation> : public typed_program_node_base<grouping_operation> {
    using parent = typed_program_node_base<grouping_operation>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using grouping_operation_node = typed_program_node<grouping_operation>;

template <>
class typed_primitive_inst<grouping_operation> : public typed_primitive_inst_base<grouping_operation> {
    using parent = typed_primitive_inst_base<grouping_operation>;
    using parent::parent;

public:
    template<typename ShapeType>
    // static std::vector<layout> calc_output_layouts(GroupingOperation_node const& /*node*/, const kernel_impl_params& impl_param) {
    //     return forward_input0_shape<ShapeType>(impl_param);
    // }
    // static layout calc_output_layout(GroupingOperation_node const& node, kernel_impl_params const& impl_param);

    static std::vector<layout> calc_output_layouts(const grouping_operation_node& /*node*/, const kernel_impl_params& impl_params);
    static layout calc_output_layout(const grouping_operation_node& node, const kernel_impl_params& impl_params) {
        return calc_output_layouts<ov::PartialShape>(node, impl_params)[0];
    }
    static std::string to_string(const grouping_operation_node& node);

    typed_primitive_inst(network& network, const grouping_operation_node& node);
};

using grouping_operation_inst = typed_primitive_inst<grouping_operation>;

}  // namespace cldnn
