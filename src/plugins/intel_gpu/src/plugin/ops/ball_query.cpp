// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/ball_query.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/ball_query.hpp"

namespace ov::intel_gpu {

static void CreateBallQueryOp(ProgramBuilder& p, const std::shared_ptr<op::v0::BallQuery>& op) {
    auto inputs = p.GetInputInfo(op);

    const std::string layerName = layer_type_name_ID(op);
    const cldnn::ball_query ball_query_prim(layerName, inputs[0], inputs[1], op->get_radius(), op->get_nsample());

    p.add_primitive(*op, ball_query_prim);
}

REGISTER_FACTORY_IMPL(v0, BallQuery);

}  // namespace ov::intel_gpu