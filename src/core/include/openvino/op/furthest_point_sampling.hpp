
#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {

class OPENVINO_API FurthestPointSampling : public Op {
public:
    OPENVINO_OP("FurthestPointSampling", "opset1");

    FurthestPointSampling() = default;

    FurthestPointSampling(const Output<Node>& xyz, const Output<Node>& npoint);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

} // namespace v0
} // namespace op
} // namespace ov