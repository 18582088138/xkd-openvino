#pragma once

#include <limits>
#include <numeric>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

#include <Eigen/Dense>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {

template <typename T>
void SVD_Infer(const T* H, T* U_output, T* S_output, T* V_output,
               const int64_t batch, const int64_t m, const int64_t n) {
    // Use the smaller dimension for U/V size calculation to match thin SVD
    const int64_t k = std::min(m, n);
    const size_t in_mat_size = m * n;
    // --- Key Change 1: U size is now m * k ---
    const size_t u_mat_size = m * k;
    const size_t s_vec_size = k;
    // --- Key Change 2: V size is now k * n (matches thin V) ---
    const size_t v_mat_size = k * n;

    for (int64_t b = 0; b < batch; ++b) {
        const T* batch_data = H + b * in_mat_size;

        // Map input data to Eigen Matrix (assuming row-major layout)
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> input_map(batch_data, m, n);
        // Convert to float for Eigen SVD computation if T is not float
        Eigen::MatrixXf A_f = input_map.template cast<float>();

        // --- Key Change 3: Use ComputeThinU and ComputeThinV ---
        // This matches torch.svd(some=True) default behavior
        Eigen::BDCSVD<Eigen::MatrixXf> svd(A_f, Eigen::ComputeThinU | Eigen::ComputeThinV);

        // Get SVD results (as float)
        Eigen::MatrixXf U_f = svd.matrixU();    // Shape: [m, k]
        Eigen::VectorXf S_f = svd.singularValues(); // Shape: [k]
        Eigen::MatrixXf V_f = svd.matrixV();    // Shape: [n, k] - This is the V matrix itself

        // --- Write U ---
        T* current_U_output = U_output + b * u_mat_size; // Pointer to output block for this batch
        // --- Key Change 4: Map output U with correct dimensions [m, k] ---
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> U_map(current_U_output, m, k);
        U_map = U_f.template cast<T>();

        // --- Write S ---
        T* current_S_output = S_output + b * s_vec_size;
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> S_map(current_S_output, s_vec_size);
        S_map = S_f.template cast<T>();

        // --- Write V ---
        T* current_V_output = V_output + b * v_mat_size; // Pointer to output block for this batch
        // --- Key Change 5: Map output V with correct dimensions [k, n] ---
        // Eigen's matrixV() returns V. PyTorch's torch.svd also returns V (not V^T) when using ComputeThinV.
        // The output shape [k, n] directly matches.
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> V_map(current_V_output, k, n);
        V_map = V_f.template cast<T>();

    } // for batch
}

} // namespace reference
} // namespace ov