// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "gather_operation_ref.hpp"

#include "common_utils/jitter.hpp"
#include "gather_operation_inst.h"
#include "intel_gpu/primitives/gather_operation.hpp"
#include "primitive_inst.h"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {

namespace {

using namespace ov::intel_gpu::ocl;

class GatherOperationGeneratorRef : public KernelGenerator {
public:
    GatherOperationGeneratorRef() : KernelGenerator("gather_operation_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = KernelGenerator::get_jit_constants(params);
        return jit_constants;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        // Input 0: features  (B, C, N) - FLOAT
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        // Input 1: idx (B, NPOINT) - INT32
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        // Output 0: (B, C, NPOINT) - FLOAT
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* /*rt_params*/) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            // Output shape: [B, C, NPOINT]
            const auto& out_shape = params.output_layouts[0].get_shape();
            const size_t B = out_shape.size() > 0 ? static_cast<size_t>(out_shape[0]) : 1;
            const size_t C = out_shape.size() > 1 ? static_cast<size_t>(out_shape[1]) : 1;
            const size_t NPOINT = out_shape.size() > 2 ? static_cast<size_t>(out_shape[2]) : 1;

            const size_t total = B * C * NPOINT;
            // Choose LWS as a divisor of total, capped by device limit
            size_t max_wgs = params.get_device_info().max_work_group_size;
            size_t lws_x = std::min(static_cast<size_t>(256), max_wgs);
            while (lws_x > 1 && (total % lws_x) != 0) {
                lws_x >>= 1;
            }
            if (lws_x == 0)
                lws_x = 1;

            std::cout << "[OV get_dispatch_data_func] : " << " total= " << total << " lws_x= " << lws_x << std::endl;

            wgs.global = {total, 1, 1};
            // wgs.local = {lws_x, 1, 1};
        }};
    }
};

class GatherOperationRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::GatherOperationRefImpl)
    Stage::Ptr gather_stage = make_stage<GatherOperationGeneratorRef>();

    GatherOperationRefImpl() : PrimitiveImplOCL(GatherOperationRef::get_type_info_static()) {}
    GatherOperationRefImpl(const program_node& /*node*/, const RuntimeParams& params) : GatherOperationRefImpl() {
        add_stage(gather_stage, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GatherOperationRefImpl>(this);
    }

    // No internal buffers are required for gather
    // [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& /*params*/) const override { return {}; }
};

}  // namespace

std::unique_ptr<primitive_impl> GatherOperationRef::create_impl(const program_node& node, const RuntimeParams& params) const {
    std::cout << "[DEBUG] Creating GATHER_OPERATION_REF_IMPL" << std::endl;
    assert(node.get_dependencies().size() == 2);  // features, idx
    assert(node.get_outputs_count() == 1);        // output
    return std::make_unique<GatherOperationRefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

// BIND_BINARY_BUFFER_WITH_TYPE(cldnn::gather_operation)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::GatherOperationRefImpl)