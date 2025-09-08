// openvino/src/core/include/openvino/op/grouping_operation.hpp

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {

class OPENVINO_API GroupingOperation : public Op {
public:
    OPENVINO_OP("GroupingOperation", "opset1");

    GroupingOperation() = default;

    GroupingOperation(const Output<Node>& features, const Output<Node>& indices);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

} // namespace v0
} // namespace op
} // namespace ov