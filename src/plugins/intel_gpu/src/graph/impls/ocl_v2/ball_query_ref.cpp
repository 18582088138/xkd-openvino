// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ball_query_ref.hpp"

#include "common_utils/jitter.hpp"
#include "ball_query_inst.h"
#include "intel_gpu/primitives/ball_query.hpp"
#include "primitive_inst.h"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {

namespace {

using namespace ov::intel_gpu::ocl;

class BallQueryGeneratorRef : public KernelGenerator {
public:
    BallQueryGeneratorRef() : KernelGenerator("ball_query_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<ball_query>();

        jit_constants.add({
            make_jit_constant("radius", desc->radius),
            make_jit_constant("nsample", desc->nsample),
        });
        return jit_constants;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        // Input 0: new_xyz (B, npoint, 3) - FLOAT
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        // Input 1: xyz (B, N, 3) - FLOAT
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        // Output 0: (B, npoint, nsample) - INT32
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        // If this kernel need internal buffer kernel, need this parameter
        // args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* /*rt_params*/) {
        assert(!params.is_dynamic());
        auto& wgs = kd.params.workGroups;

        // 输出形状: [B, npoint, nsample]
        const auto& out_shape = params.output_layouts[0].get_shape();
        const size_t B       = out_shape.size() > 0 ? static_cast<size_t>(out_shape[0]) : 1;
        const size_t npoint  = out_shape.size() > 1 ? static_cast<size_t>(out_shape[1]) : 1;
        const size_t nsample = out_shape.size() > 3 ? static_cast<size_t>(out_shape[2]) : 1;

        // 一维 GWS，匹配 kernel 的 total 线性索引
        const size_t total = B * npoint * nsample;
        wgs.global = { total, 1, 1 };

        const size_t max_wgs_total = params.get_device_info().max_work_group_size;
        size_t lws_x = std::min(static_cast<size_t>(256), max_wgs_total);
        // 让 LWS 整除 GWS，退化为 1 也要能执行
        while (lws_x > 1 && (total % lws_x) != 0) {
            lws_x >>= 1; // 减半，保证稳定
        }
        if (lws_x == 0) lws_x = 1;

        wgs.local = { lws_x, 1, 1 };
        }};
    }
};

class BallQueryRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::BallQueryRefImpl)
    Stage::Ptr grouping_stage = make_stage<BallQueryGeneratorRef>();

    BallQueryRefImpl() : PrimitiveImplOCL(BallQueryRef::get_type_info_static()) {}
    BallQueryRefImpl(const program_node& /*node*/, const RuntimeParams& params) : BallQueryRefImpl() {
        add_stage(grouping_stage, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<BallQueryRefImpl>(this);
    }

    // --- internal buffer descs (if need) ---
    // BallQuery does not require additional internal buffers to store intermediate statistics
    // [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {}
};

}  // namespace

std::unique_ptr<primitive_impl> BallQueryRef::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.get_dependencies().size() == 2);  // new_xyz, xyz
    assert(node.get_outputs_count() == 1);        // output
    return std::make_unique<BallQueryRefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

// BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ball_query)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::BallQueryRefImpl)
