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
    const T* const data = H;

    size_t in_mat_size = m * n;
    size_t u_mat_size = m * m;
    size_t s_vec_size = std::min(m, n);
    size_t v_mat_size = n * n;
    
    for (size_t b = 0; b < batch; ++b) {
        const T* batch_data = data + b * in_mat_size;

        // --- FIX: Convert T* input to Eigen::MatrixXf ---
        // Create a temporary float matrix and copy/convert data
        Eigen::MatrixXf A_f(m, n);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                // Assuming row-major input layout for H
                A_f(i, j) = static_cast<float>(batch_data[i * n + j]);
            }
        }
        // Eigen::Map approach removed due to type mismatch T vs float

        // Use Eigen's BDCSVD for better numerical stability (similar to LAPACK)
        // Compute types should match the output requirements (Full matrices)
        Eigen::BDCSVD<Eigen::MatrixXf> svd(A_f, Eigen::ComputeFullU | Eigen::ComputeFullV);

        // Get SVD results (as float)
        Eigen::MatrixXf U_f = svd.matrixU();
        Eigen::VectorXf S_f = svd.singularValues();
        Eigen::MatrixXf V_f = svd.matrixV();

        // Ensure proper signs and ordering (similar to PyTorch)
        ensure_svd_signs(U_f, S_f, V_f);

        // --- FIX: Convert float results back to T* output ---
        // Write U
        T* current_U_output = U_output + b * u_mat_size;
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < m; ++j) {
                current_U_output[i * m + j] = static_cast<T>(U_f(i, j));
            }
        }

        // Write S
        T* current_S_output = S_output + b * s_vec_size;
        for (size_t i = 0; i < s_vec_size; ++i) {
             current_S_output[i] = static_cast<T>(S_f(i));
        }

        // Write V
        T* current_V_output = V_output + b * v_mat_size;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                current_V_output[i * n + j] = static_cast<T>(V_f(i, j));
            }
        }
        // Eigen::Map approach removed for output due to type mismatch T vs float
    }

}

}  // namespace reference
}  // namespace ov