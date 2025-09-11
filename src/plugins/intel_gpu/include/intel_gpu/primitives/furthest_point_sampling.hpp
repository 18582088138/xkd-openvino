// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/graph/program.hpp"

namespace cldnn {

/// @brief furthest_point_sampling primitive for CLDNN
/// This primitive corresponds to the OpenVINO TemplateExtension::FurthestPointSampling Op.
struct furthest_point_sampling : public primitive_base<furthest_point_sampling> {
    CLDNN_DECLARE_PRIMITIVE(furthest_point_sampling);

    /// @brief Default constructor (should generally not be used directly)
    furthest_point_sampling() : primitive_base("", {}) {}

    /// @brief Constructs furthest_point_sampling primitive
    /// @param id This primitive id
    /// @param inputs Inputs primitive ids. Expected: {xyz_input, npoint_input}
    furthest_point_sampling(const primitive_id& id,
                      const input_info& input_xyz, // Represents the 'xyz' input
                      const input_info& input_npoint)  // Represents the 'npoint' input
        : primitive_base(id, {input_xyz, input_npoint}) {}

    size_t hash() const override {
        size_t seed = primitive::hash(); // Hashes ID, inputs, etc.
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
        primitive_base<furthest_point_sampling>::save(ob);
    }

    /// @brief Deserializes this primitive from a binary buffer
    void load(BinaryInputBuffer& ib) override {
        primitive_base<furthest_point_sampling>::load(ib);
    }
};

} // namespace cldnn