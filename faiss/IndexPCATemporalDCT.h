/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <faiss/Index.h>
#include <faiss/VectorTransform.h>

namespace faiss {

/** Index that compresses temporally-contiguous sequences of embeddings
 * (e.g. per-video frame embeddings) via:
 *   1. a corpus-wide PCA projection (faiss::PCAMatrix, unwhitened, no
 *      random rotation) to n_components dims -- exploits cross-dimensional
 *      redundancy,
 *   2. content-adaptive segmentation of each add()'d sequence in the
 *      projected space (a new segment starts wherever a frame's L2
 *      deviation from its segment's running mean exceeds `threshold`),
 *   3. per-segment, per-component z-score normalization + orthonormal
 *      type-II DCT along the time axis + JPEG-style quantization at a
 *      single fixed `quality` in [1, 100] -- exploits temporal redundancy,
 *   4. three storage-shrinking steps on top of the quantized coefficients,
 *      all exact (none add error beyond what step 3's quantization already
 *      introduced) -- see Segment below for the reasoning behind each:
 *      int8 (not int16) storage, per-segment trailing-zero truncation, and
 *      zlib entropy coding of the truncated bytes.
 *
 * IMPORTANT API CONTRACT: each call to add(n, x) is treated as ONE
 * temporally-contiguous sequence (e.g. one video), in frame order.
 * Do not concatenate unrelated sequences into a single add() call, and
 * do not split one logical sequence across multiple add() calls --
 * segmentation only ever sees the rows within a single add() call.
 *
 * search() is performed entirely in the n_components-dim reduced space.
 * This gives EXACTLY the same nearest-neighbor ranking as a full-d search
 * against the reconstructed corpus would: for query q and any
 * reconstructed corpus point c_hat = mean + p_c @ A (A's rows orthonormal),
 *   ||q - c_hat||^2 = ||(q-mean)_proj - p_c||^2 + ||(q-mean)_orth||^2
 * The first term is what a reduced-space L2 search computes; the second
 * depends only on q, identically across every candidate, so it can never
 * change the top-k ranking. reconstruct()/reconstruct_n() are the only
 * paths that pay the PCA un-projection cost back to d dims.
 *
 * Only METRIC_L2 is supported. add_with_ids/remove_ids/merge_from are not
 * supported (inherited throwing Index:: defaults are used), because
 * per-vector ids would break the sequential id <-> segment mapping this
 * index relies on. IO/serialization is not yet implemented.
 */
struct IndexPCATemporalDCT : Index {
    // Fixed at construction; add() relies on these not changing once
    // segments exist.
    int n_components; ///< PCA output dimensionality
    float quality;     ///< JPEG-style quality in [1, 100]
    float threshold;   ///< adaptive segmentation L2 threshold, in PCA space

    /// PCA transform: mean (d floats) + A (n_components * d floats,
    /// descending eigenvalue order). Only train() and forward projection
    /// (apply_noalloc(), via pca.mean/pca.A) are used -- reverse_transform()
    /// is deliberately NOT used for un-projection, see pca_unproject().
    PCAMatrix pca;

    /// One temporally-contiguous compressed block (segment).
    struct Segment {
        idx_t start_id = 0;  ///< global id of this segment's first row
        int rows = 0;        ///< number of frames in this segment (needed
                              ///< for the inverse DCT -- see stored_rows)
        std::vector<float> mean;   ///< size n_components
        // Named `stdev`, not `std` -- shadowing the `std` namespace with a
        // member name of the same spelling confused SWIG's C++ parser
        // (syntax error on a later, unrelated std::unordered_map member
        // declaration) even though it's legal C++.
        std::vector<float> stdev;  ///< size n_components (std_safe, >= 1e-8)

        /// Number of quantized rows actually stored, <= rows. Measured
        /// empirically across the real corpus: quantized DCT coefficients
        /// are heavily concentrated in the low-frequency (early) rows, and
        /// at typical qualities every row from stored_rows..rows-1 is
        /// *exactly* all-zero for essentially every segment -- so instead
        /// of storing (and later zlib-compressing) that guaranteed-zero
        /// tail, it's simply not stored at all; reconstruction zero-fills
        /// rows stored_rows..rows-1 before the inverse DCT. This is exact,
        /// not lossy -- it never discards a nonzero coefficient.
        int stored_rows = 0;

        /// quantized[row*n_components + c], for row in [0, stored_rows),
        /// stored as int8 (not int16: measured empirically across the real
        /// corpus, at every quality from 1-100, coefficient magnitudes
        /// never approach int8's +-127 range -- z-scored data through an
        /// orthonormal transform, divided by a qtable step that's always
        /// >=1, simply doesn't produce large values) and then zlib-
        /// compressed on top (deflate's LZ77+Huffman combo exploits the
        /// remaining run-heavy zero structure within the kept rows).
        std::vector<uint8_t> compressed_quantized;
    };
    /// segments in increasing start_id order (== order added)
    std::vector<Segment> segments;

    /// fully decoded (dequantized + inverse-DCT'd) reduced-space vectors,
    /// n_components per row, in id order 0..ntotal-1. This is what
    /// search() runs a brute-force L2 scan against.
    std::vector<float> reduced_vectors;

    /// caches keyed by segment length (rows): threshold-based segmentation
    /// produces many distinct lengths across a long sequence, and both the
    /// DCT basis and the qtable depend only on rows (qtable also depends
    /// on `quality`, but that is fixed per-instance). A `using` alias (not
    /// an inline nested-template `mutable` member declaration) sidesteps a
    /// SWIG parser issue triggered specifically in the context of the full
    /// swigfaiss.swig include chain (reproduced only there, not when this
    /// header is parsed standalone).
    using FloatArrayByLength = std::unordered_map<int, std::vector<float>>;
    mutable FloatArrayByLength dct_basis_cache; ///< rows -> rows*rows
    mutable FloatArrayByLength qtable_cache;    ///< rows -> rows

    IndexPCATemporalDCT(
            idx_t d,
            int n_components,
            float quality,
            float threshold);

    /// Fit the PCA basis (mean + top n_components eigenvectors of the
    /// covariance) on a representative corpus. Must be called before
    /// add().
    void train(idx_t n, const float* x) override;

    /// Encode one temporally-contiguous sequence: project -> adaptively
    /// segment -> per-segment normalize+DCT+quantize -> append decoded
    /// reduced-space rows to reduced_vectors. See the class-level API
    /// contract above.
    void add(idx_t n, const float* x) override;

    /// Brute-force L2 search entirely in the n_components reduced space
    /// (see class-level docs for the residual-constant argument for why
    /// this gives the exact same ranking as full-space search).
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// Reconstruct one vector to full d dims (PCA un-projection of its
    /// segment's already-decoded reduced-space row).
    void reconstruct(idx_t key, float* recons) const override;

    /// Batched reconstruct_n, grouping the requested [i0, i0+ni) range by
    /// owning segment so PCA un-projection runs once per segment slice
    /// rather than once per row.
    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    void reset() override;

    /// Actual in-memory footprint of the compressed representation: the
    /// one-time PCA basis (mean + components) plus, per segment, its
    /// mean/stdev vectors and its zlib-compressed quantized bytes (the
    /// post-truncation, post-entropy-coding size -- i.e. what a
    /// hypothetical serialized form of this index would need to write).
    /// Does NOT include reduced_vectors, the eagerly-decoded in-memory
    /// search cache -- that's a derived, reconstructible-from-segments
    /// convenience, not part of the "storage" this index is compressing.
    size_t storage_bytes() const;

    // --- internal helpers ---

    /// Returns the cached (building it on first use) rows x rows
    /// orthonormal DCT-II basis matrix, row-major.
    const std::vector<float>& get_dct_basis(int rows) const;

    /// Returns the cached (building it on first use) length-`rows`
    /// quantization step table for this index's `quality`.
    const std::vector<float>& get_qtable(int rows) const;

    /// Un-project `nrows` contiguous reduced-space rows starting at
    /// reduced_vectors row `reduced_row0` into `out` (nrows * d floats):
    /// out = reduced @ pca.A + pca.mean. Deliberately does not use
    /// PCAMatrix::reverse_transform() -- see class-level docs.
    void pca_unproject(idx_t reduced_row0, idx_t nrows, float* out) const;

    /// Locate the segment owning global id `key`; if row_within_segment is
    /// non-null, also returns key's 0-based row offset within that
    /// segment.
    const Segment& segment_for_id(idx_t key, idx_t* row_within_segment = nullptr)
            const;
};

} // namespace faiss
