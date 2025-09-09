#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {

class OPENVINO_API BallQuery : public Op {
public:
    OPENVINO_OP("BallQuery", "opset1");

    BallQuery() = default;

    BallQuery(const Output<Node>& new_xyz, const Output<Node>& xyz,
              float radius_f = 0.0f, int nsample_i = 0);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    float get_radius() const { return m_radius; }
    int get_nsample() const { return m_nsample; }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;

private:
    float m_radius = 0.1f;
    int m_nsample = 64;
};

} // namespace v0
} // namespace op
} // namespace ov