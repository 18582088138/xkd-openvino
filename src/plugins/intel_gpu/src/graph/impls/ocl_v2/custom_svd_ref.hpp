// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "custom_svd_base.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

struct CustomSVDRef : public CustomSVDBase {
    OV_GPU_PRIMITIVE_IMPL("ocl::custom_svd::ref")
    explicit CustomSVDRef(shape_types shape_type, ValidateFunc vf = nullptr) : CustomSVDBase(shape_type, std::move(vf)) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_fmts = {format::bfyx, format::bfzyx};

        static constexpr std::array supported_types = {
            ov::element::f32,
            ov::element::f16,
        };

        // static constexpr std::array supported_idx_types = {ov::element::i32, ov::element::i64};

        const auto& H_layout = node.get_input_layout(0);       // H (B, M, N)
        const auto& U_out_layout = node.get_output_layout(0);  // Uoutput (B, C, npoint, nsample)
        const auto& S_out_layout = node.get_output_layout(1);
        const auto& V_out_layout = node.get_output_layout(2);

        if (!one_of(H_layout.format, supported_fmts) || !one_of(U_out_layout.format, supported_fmts) || !one_of(S_out_layout.format, supported_fmts) ||
            !one_of(V_out_layout.format, supported_fmts)) {
            return false;
        }

        if (!one_of(H_layout.data_type, supported_types) || !one_of(U_out_layout.data_type, supported_types) ||
            !one_of(S_out_layout.data_type, supported_types) || !one_of(V_out_layout.data_type, supported_types)) {
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