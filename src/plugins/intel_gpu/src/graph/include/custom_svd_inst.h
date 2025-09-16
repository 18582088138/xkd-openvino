// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/custom_svd.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<custom_svd> : public typed_program_node_base<custom_svd> {
    using parent = typed_program_node_base<custom_svd>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using custom_svd_node = typed_program_node<custom_svd>;

template <>
class typed_primitive_inst<custom_svd> : public typed_primitive_inst_base<custom_svd> {
    using parent = typed_primitive_inst_base<custom_svd>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(const custom_svd_node& /*node*/, const kernel_impl_params& impl_params);
    // static layout calc_output_layout(const custom_svd_node& node, const kernel_impl_params& impl_params);
    static layout calc_output_layout(const custom_svd_node& node, const kernel_impl_params& impl_params) {
        return calc_output_layouts<ov::PartialShape>(node, impl_params)[0];
    }
    static std::string to_string(const custom_svd_node& node);

    typed_primitive_inst(network& network, const custom_svd_node& node);
};

using custom_svd_inst = typed_primitive_inst<custom_svd>;

}  // namespace cldnn