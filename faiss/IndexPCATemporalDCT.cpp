/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexPCATemporalDCT.h>

#include <algorithm>
#include <cmath>

#include <faiss/impl/DCTMatrix.h>
#include <faiss/impl/EntropyCoder.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

namespace faiss {

IndexPCATemporalDCT::IndexPCATemporalDCT(
        idx_t d_in,
        int n_components_in,
        float quality_in,
        float threshold_in)
        : Index(d_in, METRIC_L2),
          n_components(n_components_in),
          quality(quality_in),
          threshold(threshold_in),
          pca(static_cast<int>(d_in),
              n_components_in,
              /*eigen_power_in=*/0.f,
              /*random_rotation_in=*/false) {
    FAISS_THROW_IF_NOT_MSG(
            n_components_in > 0 && n_components_in <= d_in,
            "IndexPCATemporalDCT: n_components must be in (0, d]");
    FAISS_THROW_IF_NOT_MSG(
            quality_in >= 1.0f && quality_in <= 100.0f,
            "IndexPCATemporalDCT: quality must be in [1, 100]");
    FAISS_THROW_IF_NOT_MSG(
            threshold_in > 0.0f,
            "IndexPCATemporalDCT: threshold must be > 0");
    // Index(...) defaults is_trained=true; this index needs train() to
    // fit the PCA basis first.
    is_trained = false;
}

void IndexPCATemporalDCT::train(idx_t n, const float* x) {
    pca.train(n, x);
    is_trained = true;
}

void IndexPCATemporalDCT::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            is_trained,
            "IndexPCATemporalDCT::add: index is not trained, call train() first");
    if (n == 0) {
        return;
    }
    const idx_t id0 = ntotal;

    std::vector<float> projected(static_cast<size_t>(n) * n_components);
    pca.apply_noalloc(n, x, projected.data());

    // Content-adaptive segmentation, ported 1:1 from the validated Python
    // reference (delta_vec/compress.py::adaptive_segment_bounds): walk
    // through rows in order, maintaining a running mean (double precision,
    // matching the reference) of the current open segment; start a new
    // segment wherever a row's L2 deviation from that running mean exceeds
    // `threshold`. Operates on the *projected* sequence, per the class-level
    // API contract: this call's `n` rows are treated as one temporally-
    // contiguous sequence.
    std::vector<std::pair<idx_t, idx_t>> bounds;
    {
        std::vector<double> running_sum(n_components);
        for (int c = 0; c < n_components; c++) {
            running_sum[c] = projected[c];
        }
        idx_t count = 1;
        idx_t seg_start = 0;
        for (idx_t i = 1; i < n; i++) {
            double dev_sq = 0.0;
            const float* row = projected.data() + static_cast<size_t>(i) * n_components;
            for (int c = 0; c < n_components; c++) {
                const double running_mean_c =
                        running_sum[c] / static_cast<double>(count);
                const double diff = static_cast<double>(row[c]) - running_mean_c;
                dev_sq += diff * diff;
            }
            if (std::sqrt(dev_sq) > threshold) {
                bounds.emplace_back(seg_start, i);
                seg_start = i;
                for (int c = 0; c < n_components; c++) {
                    running_sum[c] = row[c];
                }
                count = 1;
            } else {
                for (int c = 0; c < n_components; c++) {
                    running_sum[c] += row[c];
                }
                count++;
            }
        }
        bounds.emplace_back(seg_start, n);
    }

    reduced_vectors.resize(
            (static_cast<size_t>(id0) + n) * n_components);

    for (const auto& bound : bounds) {
        const idx_t seg_start = bound.first;
        const idx_t seg_end = bound.second;
        const int rows = static_cast<int>(seg_end - seg_start);

        Segment seg;
        seg.start_id = id0 + seg_start;
        seg.rows = rows;
        seg.mean.assign(n_components, 0.0f);
        seg.stdev.assign(n_components, 0.0f);

        // Per-component mean/std over this segment (population std,
        // ddof=0 -- matches numpy's block.std(axis=0)), in double
        // precision for stability.
        std::vector<double> mean_d(n_components, 0.0);
        std::vector<double> var_d(n_components, 0.0);
        for (idx_t r = seg_start; r < seg_end; r++) {
            const float* row = projected.data() + static_cast<size_t>(r) * n_components;
            for (int c = 0; c < n_components; c++) {
                mean_d[c] += row[c];
            }
        }
        for (int c = 0; c < n_components; c++) {
            mean_d[c] /= rows;
        }
        for (idx_t r = seg_start; r < seg_end; r++) {
            const float* row = projected.data() + static_cast<size_t>(r) * n_components;
            for (int c = 0; c < n_components; c++) {
                const double diff = row[c] - mean_d[c];
                var_d[c] += diff * diff;
            }
        }
        for (int c = 0; c < n_components; c++) {
            const double std_c = std::sqrt(var_d[c] / rows);
            seg.mean[c] = static_cast<float>(mean_d[c]);
            seg.stdev[c] = (std_c > 1e-8) ? static_cast<float>(std_c) : 1.0f;
        }

        // z-score normalize
        std::vector<float> normalized(static_cast<size_t>(rows) * n_components);
        for (int r = 0; r < rows; r++) {
            const float* row = projected.data() +
                    static_cast<size_t>(seg_start + r) * n_components;
            float* nrow = normalized.data() + static_cast<size_t>(r) * n_components;
            for (int c = 0; c < n_components; c++) {
                nrow[c] = (row[c] - seg.mean[c]) / seg.stdev[c];
            }
        }

        const std::vector<float>& C = get_dct_basis(rows);
        std::vector<float> coeffs(static_cast<size_t>(rows) * n_components);
        dct_apply(
                rows,
                n_components,
                C.data(),
                /*transpose_C=*/false,
                normalized.data(),
                coeffs.data());

        const std::vector<float>& qtable = get_qtable(rows);
        std::vector<int8_t> quantized_i8(static_cast<size_t>(rows) * n_components);
        for (int r = 0; r < rows; r++) {
            const float q_step = qtable[r];
            const float* crow = coeffs.data() + static_cast<size_t>(r) * n_components;
            int8_t* qrow = quantized_i8.data() + static_cast<size_t>(r) * n_components;
            for (int c = 0; c < n_components; c++) {
                float v = std::nearbyint(crow[c] / q_step);
                // Clamp before the int8 cast: unlike numpy's wraparound
                // .astype(int8), a float outside the int8 range cast via
                // static_cast is undefined behavior in C++. (Measured
                // empirically across the real target corpus at every
                // quality 1-100: coefficient magnitudes never approach
                // +-127, so this clamp is not expected to ever bind in
                // practice -- it exists for safety on other data.)
                v = std::max(-128.0f, std::min(127.0f, v));
                qrow[c] = static_cast<int8_t>(v);
            }
        }

        // Trailing-zero truncation: find the last row with any nonzero
        // quantized coefficient and only keep rows up through it. Exact,
        // not lossy -- rows after this point are already all-zero (see
        // Segment::stored_rows's docs for why this holds at typical
        // qualities), reconstruction zero-fills them back in below.
        int stored_rows = 0;
        for (int r = rows - 1; r >= 0; r--) {
            const int8_t* qrow = quantized_i8.data() + static_cast<size_t>(r) * n_components;
            bool any_nonzero = false;
            for (int c = 0; c < n_components; c++) {
                if (qrow[c] != 0) {
                    any_nonzero = true;
                    break;
                }
            }
            if (any_nonzero) {
                stored_rows = r + 1;
                break;
            }
        }
        seg.stored_rows = stored_rows;
        seg.compressed_quantized = zlib_compress(
                reinterpret_cast<const uint8_t*>(quantized_i8.data()),
                static_cast<size_t>(stored_rows) * n_components);

        // Dequantize + inverse-DCT this segment immediately and append
        // the decoded rows to reduced_vectors, so search() and
        // reconstruct()/reconstruct_n() always scan/read the fully
        // decoded (lossy) representation without needing to redo this
        // work per query. Decoded from the actual compressed bytes (not
        // the pre-compression buffer still in scope above) so this path
        // genuinely exercises -- and is validated by -- the stored
        // representation every time add() runs.
        std::vector<int8_t> decoded_i8(static_cast<size_t>(stored_rows) * n_components);
        zlib_decompress(
                seg.compressed_quantized.data(),
                seg.compressed_quantized.size(),
                reinterpret_cast<uint8_t*>(decoded_i8.data()),
                decoded_i8.size());

        std::vector<float> dequantized(static_cast<size_t>(rows) * n_components, 0.0f);
        for (int r = 0; r < stored_rows; r++) {
            const float q_step = qtable[r];
            const int8_t* qrow = decoded_i8.data() + static_cast<size_t>(r) * n_components;
            float* drow = dequantized.data() + static_cast<size_t>(r) * n_components;
            for (int c = 0; c < n_components; c++) {
                drow[c] = static_cast<float>(qrow[c]) * q_step;
            }
        }
        // rows [stored_rows, rows) are left at 0 (zero-initialized above).
        std::vector<float> recon_normalized(static_cast<size_t>(rows) * n_components);
        dct_apply(
                rows,
                n_components,
                C.data(),
                /*transpose_C=*/true,
                dequantized.data(),
                recon_normalized.data());

        for (int r = 0; r < rows; r++) {
            const float* rn_row =
                    recon_normalized.data() + static_cast<size_t>(r) * n_components;
            float* out_row = reduced_vectors.data() +
                    static_cast<size_t>(seg.start_id + r) * n_components;
            for (int c = 0; c < n_components; c++) {
                out_row[c] = rn_row[c] * seg.stdev[c] + seg.mean[c];
            }
        }

        segments.push_back(std::move(seg));
    }

    ntotal += n;
}

void IndexPCATemporalDCT::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "IndexPCATemporalDCT::search: index is not trained");
    FAISS_THROW_IF_NOT(k > 0);
    const IDSelector* sel = params ? params->sel : nullptr;

    std::vector<float> xq_reduced(static_cast<size_t>(n) * n_components);
    pca.apply_noalloc(n, x, xq_reduced.data());

    // Brute-force L2 in the n_components-dim reduced space, reusing
    // FAISS's own BLAS-accelerated, multi-threaded, IDSelector-aware
    // kernel directly (this is exactly what IndexFlat::search() itself
    // calls internally) rather than wrapping a nested IndexFlatL2. See
    // class-level docs for why this ranking is exactly equal to a
    // full-d search against the reconstructed corpus.
    knn_L2sqr(
            xq_reduced.data(),
            reduced_vectors.data(),
            static_cast<size_t>(n_components),
            static_cast<size_t>(n),
            static_cast<size_t>(ntotal),
            static_cast<size_t>(k),
            distances,
            labels,
            nullptr,
            sel);

    // Optional polish (does not affect ranking, already exact): fold in
    // the per-query orthogonal-complement constant so the reported
    // distances equal the true full-d squared L2 to the reconstructed
    // corpus vector, matching how other lossy FAISS indices report
    // distances to the decoded approximation.
    for (idx_t i = 0; i < n; i++) {
        const float full_sq = fvec_L2sqr(x + i * d, pca.mean.data(), d);
        const float proj_sq = fvec_norm_L2sqr(
                xq_reduced.data() + i * n_components, n_components);
        float orth_sq = full_sq - proj_sq;
        if (orth_sq < 0) {
            orth_sq = 0; // guard tiny negative values from float error
        }
        for (idx_t j = 0; j < k; j++) {
            if (labels[i * k + j] >= 0) {
                distances[i * k + j] += orth_sq;
            }
        }
    }
}

void IndexPCATemporalDCT::pca_unproject(
        idx_t reduced_row0,
        idx_t nrows,
        float* out) const {
    for (idx_t r = 0; r < nrows; r++) {
        const float* reduced_row = reduced_vectors.data() +
                static_cast<size_t>(reduced_row0 + r) * n_components;
        float* out_row = out + static_cast<size_t>(r) * d;
        for (int j = 0; j < d; j++) {
            out_row[j] = pca.mean[j];
        }
        for (int i = 0; i < n_components; i++) {
            const float coeff = reduced_row[i];
            const float* A_row = pca.A.data() + static_cast<size_t>(i) * d;
            for (int j = 0; j < d; j++) {
                out_row[j] += coeff * A_row[j];
            }
        }
    }
}

void IndexPCATemporalDCT::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(
            key >= 0 && key < ntotal,
            "IndexPCATemporalDCT::reconstruct: id out of range");
    pca_unproject(key, 1, recons);
}

void IndexPCATemporalDCT::reconstruct_n(idx_t i0, idx_t ni, float* recons)
        const {
    FAISS_THROW_IF_NOT_MSG(
            i0 >= 0 && ni >= 0 && i0 + ni <= ntotal,
            "IndexPCATemporalDCT::reconstruct_n: range out of bounds");
    // reduced_vectors already holds the eagerly-decoded reduced-space
    // representation for every id, in dense id order (populated at
    // add() time -- see add()), so un-projecting the whole requested
    // range is already one batched call; no need to additionally group
    // by segment_for_id() here.
    pca_unproject(i0, ni, recons);
}

void IndexPCATemporalDCT::reset() {
    segments.clear();
    reduced_vectors.clear();
    ntotal = 0;
}

size_t IndexPCATemporalDCT::storage_bytes() const {
    size_t total = pca.mean.size() * sizeof(float) + pca.A.size() * sizeof(float);
    for (const auto& seg : segments) {
        total += seg.mean.size() * sizeof(float);
        total += seg.stdev.size() * sizeof(float);
        total += seg.compressed_quantized.size();
    }
    return total;
}

const std::vector<float>& IndexPCATemporalDCT::get_dct_basis(int rows) const {
    auto it = dct_basis_cache.find(rows);
    if (it != dct_basis_cache.end()) {
        return it->second;
    }
    std::vector<float> C(static_cast<size_t>(rows) * rows);
    dct2_ortho_basis(rows, C.data());
    auto res = dct_basis_cache.emplace(rows, std::move(C));
    return res.first->second;
}

const std::vector<float>& IndexPCATemporalDCT::get_qtable(int rows) const {
    auto it = qtable_cache.find(rows);
    if (it != qtable_cache.end()) {
        return it->second;
    }
    std::vector<float> qt(rows);
    build_qtable(rows, quality, qt.data());
    auto res = qtable_cache.emplace(rows, std::move(qt));
    return res.first->second;
}

const IndexPCATemporalDCT::Segment& IndexPCATemporalDCT::segment_for_id(
        idx_t key,
        idx_t* row_within_segment) const {
    FAISS_THROW_IF_NOT_MSG(
            !segments.empty(),
            "IndexPCATemporalDCT::segment_for_id: index is empty");
    // segments are in strictly increasing start_id order (appended in
    // that order by add(), across possibly multiple add() calls) --
    // binary search for the last segment with start_id <= key.
    auto it = std::upper_bound(
            segments.begin(),
            segments.end(),
            key,
            [](idx_t k, const Segment& s) { return k < s.start_id; });
    FAISS_THROW_IF_NOT_MSG(
            it != segments.begin(),
            "IndexPCATemporalDCT::segment_for_id: id out of range");
    --it;
    if (row_within_segment) {
        *row_within_segment = key - it->start_id;
    }
    return *it;
}

} // namespace faiss
