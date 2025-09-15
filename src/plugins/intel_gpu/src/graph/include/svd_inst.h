// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/svd.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<svd> : public typed_program_node_base<svd> {
    using parent = typed_program_node_base<svd>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using svd_node = typed_program_node<svd>;

template <>
class typed_primitive_inst<svd> : public typed_primitive_inst_base<svd> {
    using parent = typed_primitive_inst_base<svd>;
    using parent::parent;

public:
    template<typename ShapeType>
    // static std::vector<layout> calc_output_layouts(GroupingOperation_node const& /*node*/, const kernel_impl_params& impl_param) {
    //     return forward_input0_shape<ShapeType>(impl_param);
    // }
    // static layout calc_output_layout(GroupingOperation_node const& node, kernel_impl_params const& impl_param);

    static std::vector<layout> calc_output_layouts(const svd_node& /*node*/, const kernel_impl_params& impl_params);
    static layout calc_output_layout(const svd_node& node, const kernel_impl_params& impl_params) {
        return calc_output_layouts<ov::PartialShape>(node, impl_params)[0];
    }
    static std::string to_string(const svd_node& node);

    typed_primitive_inst(network& network, const svd_node& node);
};

using svd_inst = typed_primitive_inst<svd>;

}  // namespace cldnn