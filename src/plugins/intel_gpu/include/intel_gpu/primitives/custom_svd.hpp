// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/graph/program.hpp"

namespace cldnn {

/// @brief custom_svd primitive for CLDNN
/// This primitive corresponds to the OpenVINO TemplateExtension::CustomSVD Op.
struct custom_svd : public primitive_base<custom_svd> {
    CLDNN_DECLARE_PRIMITIVE(custom_svd);

    /// @brief Default constructor (should generally not be used directly)
    custom_svd() : primitive_base("", {}) {}

    /// @brief Constructs custom_svd primitive
    /// @param id This primitive id
    /// @param inputs Inputs primitive ids. Expected: {H_input}
    custom_svd(const primitive_id& id, const input_info& input_H) // Represents the 'H' input
        : primitive_base(id, {input_H}) {}

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
        primitive_base<custom_svd>::save(ob);
    }

    /// @brief Deserializes this primitive from a binary buffer
    void load(BinaryInputBuffer& ib) override {
        primitive_base<custom_svd>::load(ib);
    }
};

} // namespace cldnn