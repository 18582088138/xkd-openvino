// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/graph/program.hpp"

namespace cldnn {

/// @brief gather_operation primitive for CLDNN
/// This primitive corresponds to the OpenVINO TemplateExtension::GatherOperation Op.
struct gather_operation : public primitive_base<gather_operation> {
    CLDNN_DECLARE_PRIMITIVE(gather_operation);

    /// @brief Default constructor (should generally not be used directly)
    gather_operation() : primitive_base("", {}) {}

    /// @brief Constructs gather_operation primitive
    /// @param id This primitive id
    /// @param inputs Inputs primitive ids. Expected: {features_input, idx_input}
    gather_operation(const primitive_id& id,
                      const input_info& input_features, // Represents the 'features' input
                      const input_info& input_idx)      // Represents the 'idx' input
        : primitive_base(id, {input_features, input_idx}) {}

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
        primitive_base<gather_operation>::save(ob);
    }

    /// @brief Deserializes this primitive from a binary buffer
    void load(BinaryInputBuffer& ib) override {
        primitive_base<gather_operation>::load(ib);
    }
};

} // namespace cldnn