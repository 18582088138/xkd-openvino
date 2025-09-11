// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "furthest_point_sampling_opt.hpp"

#include "common_utils/jitter.hpp"
#include "furthest_point_sampling_inst.h"
#include "intel_gpu/primitives/furthest_point_sampling.hpp"
#include "primitive_inst.h"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {

namespace {

using namespace ov::intel_gpu::ocl;

class FurthestPointSamplingGeneratorOpt : public KernelGenerator {
public:
    FurthestPointSamplingGeneratorOpt() : KernelGenerator("furthest_point_sampling_opt") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = KernelGenerator::get_jit_constants(params);
        return jit_constants;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        // Input 0: xyz  (B, N, 3) - FLOAT
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        // Input 1: npoint (npoint) - INT32
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        // Output 0: (B, npoint) - INT32
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        // Internal buffer: temp [B, N]
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* /*rt_params*/) {
        assert(!params.is_dynamic());
        auto& wgs = kd.params.workGroups;

        const auto& out_shape = params.output_layouts[0].get_shape();
        const size_t B = out_shape.size() > 0 ? static_cast<size_t>(out_shape[0]) : 1;

        // Launch one work-group per batch
        wgs.global = { B * 256, 1, 1 };  // 256 work-items per batch
        wgs.local  = { 256, 1, 1 };      // Work-group size 256
        }};
    }
};

class FurthestPointSamplingOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::FurthestPointSamplingOptImpl)
    Stage::Ptr grouping_stage = make_stage<FurthestPointSamplingGeneratorOpt>();

    FurthestPointSamplingOptImpl() : PrimitiveImplOCL(FurthestPointSamplingOpt::get_type_info_static()) {}
    FurthestPointSamplingOptImpl(const program_node& /*node*/, const RuntimeParams& params) : FurthestPointSamplingOptImpl() {
        add_stage(grouping_stage, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<FurthestPointSamplingOptImpl>(this);
    }

    // Internal buffer for temp distances
    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        const auto& pshape = params.input_layouts[0].get_partial_shape(); // [B,N,3]
        const auto max_shape = pshape.get_max_shape();
        const size_t B = max_shape.size() > 0 ? static_cast<size_t>(max_shape[0]) : 1;
        const size_t N = max_shape.size() > 1 ? static_cast<size_t>(max_shape[1]) : 1;
        return { BufferDescriptor{ B * N, ov::element::f32 } };
    }
};

}  // namespace

std::unique_ptr<primitive_impl> FurthestPointSamplingOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.get_dependencies().size() == 2);  // features, idx
    assert(node.get_outputs_count() == 1);        // output
    return std::make_unique<FurthestPointSamplingOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::FurthestPointSamplingOptImpl)
