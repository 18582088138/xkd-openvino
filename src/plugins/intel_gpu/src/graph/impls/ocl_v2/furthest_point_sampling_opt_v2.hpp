// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "furthest_point_sampling_base.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

struct FurthestPointSamplingOptV2 : public FurthestPointSamplingBase {
    OV_GPU_PRIMITIVE_IMPL("ocl::furthest_point_sampling::opt_v2")
    explicit FurthestPointSamplingOptV2(shape_types shape_type, ValidateFunc vf = nullptr) : FurthestPointSamplingBase(shape_type, std::move(vf)) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_fmts = {format::bfyx, format::bfzyx};

        static constexpr std::array supported_types = {
            ov::element::f32,
            ov::element::f16,
            ov::element::i32,
            ov::element::i64,
        };

        static constexpr std::array supported_npoint_types = {ov::element::i32, ov::element::i64};

        const auto& xyz_layout = node.get_input_layout(0);     // xyz (B, N, 3)
        const auto& npoint_layout = node.get_input_layout(1);  // npoint (npoint)
        const auto& out_layout = node.get_output_layout(0);    // output (B, npoint)

        if (!one_of(xyz_layout.format, supported_fmts) || !one_of(out_layout.format, supported_fmts)) {
            return false;
        }

        if (!one_of(npoint_layout.format, supported_fmts) || !one_of(npoint_layout.data_type, supported_npoint_types)) {
            return false;
        }

        if (!one_of(xyz_layout.data_type, supported_types) || !one_of(out_layout.data_type, supported_types)) {
            return false;
        }

        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
