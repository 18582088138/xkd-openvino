// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "ball_query_base.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

struct BallQueryRef : public BallQueryBase {
    OV_GPU_PRIMITIVE_IMPL("ocl::ball_query::ref")
    explicit BallQueryRef(shape_types shape_type, ValidateFunc vf = nullptr) : BallQueryBase(shape_type, std::move(vf)) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_fmts = {format::bfyx, format::bfzyx};

        static constexpr std::array supported_types = {
            ov::element::f32,
            ov::element::f16,
            ov::element::i32,
            ov::element::i64,
        };

        // static constexpr std::array supported_xyz_types = {ov::element::i32, ov::element::i64};

        const auto& new_xyz_layout = node.get_input_layout(0);  // new_xyz (B, npoint, 3)
        const auto& xyz_layout = node.get_input_layout(1);      // xyz    (B, N, 3)
        const auto& out_layout = node.get_output_layout(0);     // output (B, npoint, nsample)

        if (!one_of(new_xyz_layout.format, supported_fmts) || !one_of(out_layout.format, supported_fmts)) {
            return false;
        }

        if (!one_of(xyz_layout.format, supported_fmts) || !one_of(xyz_layout.data_type, supported_types)) {
            return false;
        }

        if (!one_of(new_xyz_layout.data_type, supported_types) || !one_of(out_layout.data_type, supported_types)) {
            return false;
        }

        // [ToDo]fused_ops are not currently supported.

        // if (!fused_ops_are_one_of<eltwise, activation>(node.get_fused_primitives())) {
        //     return false;
        // }

        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
