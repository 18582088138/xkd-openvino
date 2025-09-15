// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "svd_ref.hpp"

#include "common_utils/jitter.hpp"
#include "svd_inst.h"
#include "intel_gpu/primitives/svd.hpp"
#include "primitive_inst.h"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {

namespace {

using namespace ov::intel_gpu::ocl;

class SVDGeneratorRef : public KernelGenerator {
public:
    SVDGeneratorRef() : KernelGenerator("svd_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = KernelGenerator::get_jit_constants(params);
        return jit_constants;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        // Input 0: H (B, M, N) - FLOAT
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});

        // Output U: (B, M, M) - FLOAT
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        // Output S: (B, min(M,N)) - FLOAT
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 1});

        // Output V: (B, N, N) - FLOAT
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 2});

        // If this kernel need internal buffer kernel, need this parameter
        // args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* /*rt_params*/) {
        assert(!params.is_dynamic());
        auto& wgs = kd.params.workGroups;

        // 获取输入 H 的形状 (B, M, N)
            const auto& input_shape = params.input_layouts[0].get_shape();
            const size_t input_rank = input_shape.size();

            // 假设输入至少是 2D (M, N)，批次维度是前面的所有维度的乘积
            size_t batch_size = 1;
            for (size_t i = 0; i < input_rank - 2; ++i) {
                batch_size *= input_shape[i];
            }
            // 如果输入是 2D (M, N)，batch_size 就是 1

            const size_t B = batch_size;
            // M 和 N 可以获取，但 dispatch 通常不直接用它们作为 GWS
            // const size_t M = input_shape[input_rank - 2];
            // const size_t N = input_shape[input_rank - 1];

            // Dispatch 策略：为每个批次分配一个或多个工作组
            // 简单起见，我们让 GWS.x 对应批次大小 B
            wgs.global = { B, 1, 1 }; // 每个批次一个全局工作项（可以扩展为多个）

            // 设置局部工作组大小 (LWS)
            const size_t max_wgs_total = params.get_device_info().max_work_group_size;
            // 选择一个合理的 LWS，通常是 2 的幂，且不超过 max_wgs_total
            // 并且最好能整除 GWS 的 x 维度 (B)
            size_t lws_x = 1;
            // 从一个较小的合理值开始向上找，但不超过 max_wgs_total
            // 并且是 B 的因子或能较好地处理 B
            const size_t preferred_lws = 64; // 常见的 LWS 大小
            if (B > 0) { // 防止除零
                 // 尝试 preferred_lws 或其小于 max_wgs_total 的最大因子
                size_t candidate_lws = preferred_lws;
                while (candidate_lws > max_wgs_total) {
                    candidate_lws /= 2;
                }
                if (candidate_lws > 0) {
                    // 找到 B 的最大因子，该因子 <= candidate_lws
                    for (lws_x = candidate_lws; lws_x > 1; --lws_x) {
                        if (B % lws_x == 0) {
                            break;
                        }
                    }
                    if (lws_x == 1) { // 如果没找到因子，就用 1
                         lws_x = 1;
                    }
                }
            }
            // 确保 lws_x 至少为 1 且不超过 max_wgs_total
            lws_x = std::max(static_cast<size_t>(1), std::min(lws_x, max_wgs_total));

            wgs.local = { lws_x, 1, 1 };
        }};
    }
};

class SVDRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::SVDRefImpl)
    Stage::Ptr grouping_stage = make_stage<SVDGeneratorRef>();

    SVDRefImpl() : PrimitiveImplOCL(SVDRef::get_type_info_static()) {}
    SVDRefImpl(const program_node& /*node*/, const RuntimeParams& params) : SVDRefImpl() {
        add_stage(grouping_stage, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<SVDRefImpl>(this);
    }

    // --- internal buffer descs (if need) ---
    // SVD does not require additional internal buffers to store intermediate statistics
    // [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {}
};

}  // namespace

std::unique_ptr<primitive_impl> SVDRef::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.get_dependencies().size() == 1);   // H
    assert(node.get_outputs_count() == 3);      // U, S, V
    return std::make_unique<SVDRefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

// BIND_BINARY_BUFFER_WITH_TYPE(cldnn::svd)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::SVDRefImpl)