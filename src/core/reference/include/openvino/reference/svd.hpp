#pragma once

#include <limits>
#include <numeric>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

#include <Eigen/Dense>

#include "openvino/core/shape.hpp"

void ensure_svd_signs(Eigen::MatrixXf& U, Eigen::VectorXf& S, Eigen::MatrixXf& V) {
    // Ensure singular values are non-negative and sorted in descending order
    for (int i = 0; i < S.size(); ++i) {
        if (S(i) < 0) {
            S(i) = -S(i);
            U.col(i) = -U.col(i);
        }
    }
    
    // Sort singular values in descending order
    std::vector<std::pair<float, int>> s_indices;
    for (int i = 0; i < S.size(); ++i) {
        s_indices.push_back({S(i), i});
    }
    std::sort(s_indices.begin(), s_indices.end(), 
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                  return a.first > b.first;
              });
    
    // Reorder U, S, V according to sorted singular values
    Eigen::MatrixXf U_new = U;
    Eigen::MatrixXf V_new = V;
    Eigen::VectorXf S_new = S;
    
    for (int i = 0; i < S.size(); ++i) {
        int old_idx = s_indices[i].second;
        S_new(i) = s_indices[i].first;
        U_new.col(i) = U.col(old_idx);
        V_new.col(i) = V.col(old_idx);
    }
    
    U = U_new;
    S = S_new;
    V = V_new;
}

namespace ov {
namespace reference {

template <typename T>
void SVD_Infer(const T* const H,
            T* U_output,
            T* S_output,
            T* V_output,
            size_t batch,
            size_t m,
            size_t n){
    const float* data = H;

    size_t in_mat_size = m * n;
    size_t u_mat_size = m * m;
    size_t s_vec_size = std::min(m, n);
    size_t v_mat_size = n * n;
    
    for (size_t b = 0; b < batch; ++b) {
        const float* batch_data = data + b * in_mat_size;
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A(batch_data, m, n);
        
        // Use Eigen's BDCSVD for better numerical stability (similar to LAPACK)
        Eigen::BDCSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        
        // Get SVD results
        Eigen::MatrixXf U = svd.matrixU();
        Eigen::VectorXf S = svd.singularValues();
        Eigen::MatrixXf V = svd.matrixV();
        
        // Ensure proper signs and ordering (similar to PyTorch)
        ensure_svd_signs(U, S, V);
        
        // Write outputs
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(U_output + b * u_mat_size, m, m) = U;
        Eigen::Map<Eigen::VectorXf>(S_output + b * s_vec_size, s_vec_size) = S;
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(V_output + b * v_mat_size, n, n) = V;
    }

}

}  // namespace reference
}  // namespace ov