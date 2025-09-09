// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/graph/program.hpp"

namespace cldnn {

/// @brief ball_query primitive for CLDNN
/// This primitive corresponds to the OpenVINO TemplateExtension::BallQuery Op.
struct ball_query : public primitive_base<ball_query> {
    CLDNN_DECLARE_PRIMITIVE(ball_query);

    /// @brief Default constructor (should generally not be used directly)
    ball_query() : primitive_base("", {}), radius(0.1f), nsample(64) {}

    /// @brief Constructs ball_query primitive
    /// @param id This primitive id
    /// @param inputs Inputs primitive ids. Expected: {new_xyz_input, xyz_input}
    ball_query(const primitive_id& id,
                      const input_info& input_new_xyz, // Represents the 'new_xyz' input
                      const input_info& input_xyz,
                      float radius,
                      int32_t nsample)  // Represents the 'xyz' input
        : primitive_base(id, {input_new_xyz, input_xyz}) 
        , radius(radius)
        , nsample(nsample) {}

    /// @brief Radius of the ball search
    float radius;
    /// @brief Maximum number of neighbors to return
    int32_t nsample;

    size_t hash() const override {
        size_t seed = primitive::hash(); // Hashes ID, inputs, etc.
        seed = hash_combine(seed, radius);
        seed = hash_combine(seed, nsample);
        return seed;
    }

    /// @brief Compares this primitive with another for equality
    bool operator==(const primitive& rhs) const override {
        // First compare common parameters (ID, inputs)
        if (!compare_common_params(rhs))
            return false;
         return true;
    }

    /// @brief Serializes this primitive to a binary buffer
    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<ball_query>::save(ob);
        ob << radius;
        ob << nsample;
    }

    /// @brief Deserializes this primitive from a binary buffer
    void load(BinaryInputBuffer& ib) override {
        primitive_base<ball_query>::load(ib);
        ib >> radius;
        ib >> nsample;
    }
};

} // namespace cldnn