// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/graph/program.hpp"

namespace cldnn {

/// @brief grouping_operation primitive for CLDNN
/// This primitive corresponds to the OpenVINO TemplateExtension::GroupingOperation Op.
struct grouping_operation : public primitive_base<grouping_operation> {
    CLDNN_DECLARE_PRIMITIVE(grouping_operation);

    /// @brief Default constructor (should generally not be used directly)
    grouping_operation() : primitive_base("", {}) {}

    /// @brief Constructs grouping_operation primitive
    /// @param id This primitive id
    /// @param inputs Inputs primitive ids. Expected: {features_input, idx_input}
    grouping_operation(const primitive_id& id,
                      const input_info& input_features, // Represents the 'features' input
                      const input_info& input_indices)  // Represents the 'idx' input
        : primitive_base(id, {input_features, input_indices}) {}

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
        primitive_base<grouping_operation>::save(ob);
    }

    /// @brief Deserializes this primitive from a binary buffer
    void load(BinaryInputBuffer& ib) override {
        primitive_base<grouping_operation>::load(ib);
    }
};

} // namespace cldnn