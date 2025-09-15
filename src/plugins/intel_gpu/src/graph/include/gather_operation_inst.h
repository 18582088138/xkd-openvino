// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/gather_operation.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<gather_operation> : public typed_program_node_base<gather_operation> {
    using parent = typed_program_node_base<gather_operation>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using gather_operation_node = typed_program_node<gather_operation>;

template <>
class typed_primitive_inst<gather_operation> : public typed_primitive_inst_base<gather_operation> {
    using parent = typed_primitive_inst_base<gather_operation>;
    using parent::parent;

public:
    template<typename ShapeType>

    static std::vector<layout> calc_output_layouts(const gather_operation_node& /*node*/, const kernel_impl_params& impl_params);
    static layout calc_output_layout(const gather_operation_node& node, const kernel_impl_params& impl_params) {
        return calc_output_layouts<ov::PartialShape>(node, impl_params)[0];
    }
    static std::string to_string(const gather_operation_node& node);

    typed_primitive_inst(network& network, const gather_operation_node& node);
};

using gather_operation_inst = typed_primitive_inst<gather_operation>;

}  // namespace cldnn