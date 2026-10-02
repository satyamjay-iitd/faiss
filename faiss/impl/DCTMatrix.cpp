/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/DCTMatrix.h>

#include <algorithm>
#include <cmath>

namespace faiss {

namespace {
// Avoid relying on the M_PI macro (not guaranteed by the C++ standard,
// only by POSIX extensions some compilers gate behind feature-test
// macros).
constexpr double kPi = 3.14159265358979323846;
} // namespace

void dct2_ortho_basis(int N, float* C) {
    const double alpha0 = std::sqrt(1.0 / N);
    const double alphak = std::sqrt(2.0 / N);
    for (int k = 0; k < N; k++) {
        const double alpha = (k == 0) ? alpha0 : alphak;
        for (int n = 0; n < N; n++) {
            const double angle = kPi * (2.0 * n + 1.0) * k / (2.0 * N);
            C[k * N + n] = static_cast<float>(alpha * std::cos(angle));
        }
    }
}

void build_qtable(int rows, float quality, float* qtable) {
    float q = quality;
    q = std::max(1.0f, std::min(100.0f, q));
    const float scale = (q < 50.0f) ? (5000.0f / q) : (200.0f - 2.0f * q);
    for (int f = 0; f < rows; f++) {
        const float base = 1.0f + static_cast<float>(f);
        const float val = std::floor((base * scale + 50.0f) / 100.0f);
        qtable[f] = std::max(val, 1.0f);
    }
}

void dct_apply(
        int rows,
        int n_components,
        const float* C,
        bool transpose_C,
        const float* in,
        float* out) {
    for (int i = 0; i < rows; i++) {
        float* out_row = out + static_cast<size_t>(i) * n_components;
        std::fill(out_row, out_row + n_components, 0.0f);
        for (int j = 0; j < rows; j++) {
            const float c_val = transpose_C ? C[j * rows + i] : C[i * rows + j];
            if (c_val == 0.0f) {
                continue;
            }
            const float* in_row = in + static_cast<size_t>(j) * n_components;
            for (int c = 0; c < n_components; c++) {
                out_row[c] += c_val * in_row[c];
            }
        }
    }
}

} // namespace faiss
