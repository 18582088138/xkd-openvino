// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

struct FurthestPointSamplingBase : public ImplementationManager {
    explicit FurthestPointSamplingBase(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] in_out_fmts_t query_formats(const program_node& node) const override {
        size_t input_count = node.get_dependencies().size();
        size_t output_count = node.get_outputs_count();

        std::vector<format::type> in_fmts(input_count, format::any);
        std::vector<format::type> out_fmts(output_count, format::any);

        return {in_fmts, out_fmts};
    }
};

}  // namespace ov::intel_gpu::ocl