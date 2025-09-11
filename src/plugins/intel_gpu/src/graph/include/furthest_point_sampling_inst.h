// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/furthest_point_sampling.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<furthest_point_sampling> : public typed_program_node_base<furthest_point_sampling> {
    using parent = typed_program_node_base<furthest_point_sampling>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using furthest_point_sampling_node = typed_program_node<furthest_point_sampling>;

template <>
class typed_primitive_inst<furthest_point_sampling> : public typed_primitive_inst_base<furthest_point_sampling> {
    using parent = typed_primitive_inst_base<furthest_point_sampling>;
    using parent::parent;

public:
    template<typename ShapeType>

    static std::vector<layout> calc_output_layouts(const furthest_point_sampling_node& /*node*/, const kernel_impl_params& impl_params);
    static layout calc_output_layout(const furthest_point_sampling_node& node, const kernel_impl_params& impl_params) {
        return calc_output_layouts<ov::PartialShape>(node, impl_params)[0];
    }
    static std::string to_string(const furthest_point_sampling_node& node);

    typed_primitive_inst(network& network, const furthest_point_sampling_node& node);
};

using furthest_point_sampling_inst = typed_primitive_inst<furthest_point_sampling>;

}  // namespace cldnn