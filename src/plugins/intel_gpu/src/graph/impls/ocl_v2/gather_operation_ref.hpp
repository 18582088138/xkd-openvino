// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "gather_operation_base.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

struct GatherOperationRef : public GatherOperationBase {
    OV_GPU_PRIMITIVE_IMPL("ocl::gather_operation::ref")
    explicit GatherOperationRef(shape_types shape_type, ValidateFunc vf = nullptr) : GatherOperationBase(shape_type, std::move(vf)) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_fmts = {format::bfyx, format::bfzyx};

        static constexpr std::array supported_types = {
            ov::element::f32,
            ov::element::f16,
        };

        static constexpr std::array supported_idx_types = {ov::element::i32, ov::element::i64};

        const auto& features_layout = node.get_input_layout(0);  // features (B, C, N)
        const auto& idx_layout = node.get_input_layout(1);       //  idx (B, npoint）
        const auto& out_layout = node.get_output_layout(0);      // output (B, C, npoint)

        // std::cout << "[DEBUG] ov GatherOperationRef" << std::endl;

        // std::cout << "[DEBUG] ov features_layout.format = " << features_layout.format << std::endl;
        // std::cout << "[DEBUG] ov idx_layout.format = " << idx_layout.format << std::endl;
        // std::cout << "[DEBUG] ov out_layout.format = " << out_layout.format << std::endl;

        // std::cout << "[DEBUG] =======================" << std::endl;

        // std::cout << "[DEBUG] ov features_layout.data_type = " << features_layout.data_type << std::endl;
        // std::cout << "[DEBUG] ov idx_layout.data_type = " << idx_layout.data_type << std::endl;
        // std::cout << "[DEBUG] ov out_layout.data_type = " << out_layout.data_type << std::endl;

        if (!one_of(features_layout.format, supported_fmts) || !one_of(out_layout.format, supported_fmts)) {
            return false;
        }

        if (!one_of(idx_layout.format, supported_fmts) || !one_of(idx_layout.data_type, supported_idx_types)) {
            return false;
        }

        if (!one_of(features_layout.data_type, supported_types) || !one_of(out_layout.data_type, supported_types)) {
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