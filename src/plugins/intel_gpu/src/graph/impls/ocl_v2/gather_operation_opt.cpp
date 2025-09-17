// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "gather_operation_opt.hpp"

#include "common_utils/jitter.hpp"
#include "gather_operation_inst.h"
#include "intel_gpu/primitives/gather_operation.hpp"
#include "primitive_inst.h"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {

namespace {

using namespace ov::intel_gpu::ocl;

class GatherOperationGeneratorOpt : public KernelGenerator {
public:
    GatherOperationGeneratorOpt() : KernelGenerator("gather_operation_opt") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = KernelGenerator::get_jit_constants(params);
        // 可添加额外的 JIT 宏，例如：
        // jit_constants.add_constant(MakeJitConstant("ENABLE_SANITIZATION", 1));
        return jit_constants;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            std::cout << "[OV debug] gather_operation_opt is_dynamic" << std::endl;
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        // Input 0: features (B, C, N)
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        // Input 1: idx (B, NPOINT)
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        // Output 0: (B, C, NPOINT)
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* /*rt_params*/) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            const auto& out_shape = params.output_layouts[0].get_shape();
            const size_t B = out_shape.size() > 0 ? static_cast<size_t>(out_shape[0]) : 1;
            const size_t C = out_shape.size() > 1 ? static_cast<size_t>(out_shape[1]) : 1;
            const size_t NPOINT = out_shape.size() > 2 ? static_cast<size_t>(out_shape[2]) : 1;

            const size_t total_elements = B * C * NPOINT;

            // 启发式：使用较大的 global size 提高并行度
            size_t max_wgs = params.get_device_info().max_work_group_size;
            size_t lws_x = std::min(static_cast<size_t>(256), max_wgs);

            // 调整 LWS 为 total_elements 的约数（或接近）
            while (lws_x > 1 && (total_elements % lws_x) != 0) {
                lws_x >>= 1;
            }
            if (lws_x == 0) lws_x = 1;

            // 🔥 关键优化：允许 global_size > total_elements（grid-stride）
            size_t preferred_global = next_power_of_two(std::max(total_elements, static_cast<size_t>(65536)));
            size_t global_x = round_up_to_multiple(preferred_global, lws_x);

            std::cout << "[OV get_dispatch_data_func OPT] total= " << total_elements
                      << " lws_x= " << lws_x
                      << " global_x= " << global_x << std::endl;

            wgs.global = {global_x, 1, 1};
            wgs.local = {lws_x, 1, 1};
        }};
    }

private:
    static size_t next_power_of_two(size_t n) {
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        return ++n;
    }

    static size_t round_up_to_multiple(size_t value, size_t multiple) {
        return ((value + multiple - 1) / multiple) * multiple;
    }
};

class GatherOperationOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::GatherOperationOptImpl)

    Stage::Ptr gather_stage = make_stage<GatherOperationGeneratorOpt>();

    GatherOperationOptImpl() : PrimitiveImplOCL(GatherOperationOpt::get_type_info_static()) {}
    GatherOperationOptImpl(const program_node& /*node*/, const RuntimeParams& params) : GatherOperationOptImpl() {
        add_stage(gather_stage, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GatherOperationOptImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> GatherOperationOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    std::cout << "[DEBUG] Creating GATHER_OPERATION_OPT_IMPL" << std::endl;
    assert(node.get_dependencies().size() == 2);  // features, idx
    assert(node.get_outputs_count() == 1);        // output
    return std::make_unique<GatherOperationOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

// 注册二进制序列化
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::GatherOperationOptImpl)