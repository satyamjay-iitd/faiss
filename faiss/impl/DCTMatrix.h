/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace faiss {

/// Build the orthonormal N x N type-II DCT basis matrix, row-major:
///   C[k*N + n] = alpha(k) * cos(pi * (2n+1) * k / (2N))
///   alpha(0) = sqrt(1/N), alpha(k>0) = sqrt(2/N)
/// Matches scipy.fft.dct(x, type=2, norm="ortho") exactly. Because C is
/// orthonormal (C @ C^T == I), the same matrix transposed is the inverse
/// (type-III) transform under the same "ortho" scaling -- no separate IDCT
/// formula is needed, see dct_apply's transpose_C flag.
void dct2_ortho_basis(int N, float* C);

/// JPEG-style quantization step table: a linear ramp (finer steps at low
/// frequencies, coarser at high), scaled by JPEG's own quality-scaling
/// formula. `quality` is clamped to [1, 100]. Port of the Python reference's
/// build_qtable/jpeg_quality_scale (delta_vec/compress.py).
void build_qtable(int rows, float quality, float* qtable);

/// out = op(C) @ in, where C is rows x rows (row-major) and in/out are
/// rows x n_components (row-major). transpose_C=false applies the forward
/// DCT-II; transpose_C=true applies the inverse (DCT-III). Naive O(rows^2 *
/// n_components) reference implementation -- segments are small (tens to a
/// few hundred rows), so this is not a performance concern; kept as the
/// permanent implementation and as the test oracle for any future
/// BLAS-accelerated path.
void dct_apply(
        int rows,
        int n_components,
        const float* C,
        bool transpose_C,
        const float* in,
        float* out);

} // namespace faiss
