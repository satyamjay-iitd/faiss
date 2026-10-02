/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <faiss/IndexPCATemporalDCT.h>
#include <faiss/impl/DCTMatrix.h>
#include <faiss/impl/EntropyCoder.h>
#include <faiss/utils/random.h>

namespace {

std::vector<float> make_random(size_t n, size_t d, long seed) {
    std::vector<float> x(n * d);
    faiss::float_randn(x.data(), x.size(), seed);
    return x;
}

} // namespace

// ---------------------------------------------------------------------
// DCTMatrix: pure math, no Index dependency.
// ---------------------------------------------------------------------

TEST(DCTMatrix, ForwardInverseRoundTrip) {
    // Because the basis is orthonormal (C @ C^T == I), applying the
    // forward transform (transpose_C=false) followed by the inverse
    // (transpose_C=true) must recover the original input exactly, up to
    // float rounding -- this is the DCT-II/DCT-III duality under "ortho"
    // scaling the whole quantization scheme depends on.
    for (int N : {1, 2, 5, 17, 64, 200}) {
        const int n_components = 3;
        std::vector<float> C(static_cast<size_t>(N) * N);
        faiss::dct2_ortho_basis(N, C.data());

        std::vector<float> x = make_random(N, n_components, 1000 + N);
        std::vector<float> coeffs(x.size()), recon(x.size());
        faiss::dct_apply(N, n_components, C.data(), false, x.data(), coeffs.data());
        faiss::dct_apply(N, n_components, C.data(), true, coeffs.data(), recon.data());

        for (size_t i = 0; i < x.size(); i++) {
            EXPECT_NEAR(x[i], recon[i], 1e-4) << "N=" << N << " i=" << i;
        }
    }
}

TEST(DCTMatrix, BasisIsOrthonormal) {
    // C @ C^T == I, the property the round-trip test above relies on.
    for (int N : {1, 3, 8, 33}) {
        std::vector<float> C(static_cast<size_t>(N) * N);
        faiss::dct2_ortho_basis(N, C.data());
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double dot = 0.0;
                for (int k = 0; k < N; k++) {
                    dot += static_cast<double>(C[i * N + k]) * C[j * N + k];
                }
                EXPECT_NEAR(dot, (i == j) ? 1.0 : 0.0, 1e-4)
                        << "N=" << N << " i=" << i << " j=" << j;
            }
        }
    }
}

TEST(DCTMatrix, QtableMonotonicAndAtLeastOne) {
    for (int rows : {1, 8, 64}) {
        for (float quality : {1.0f, 25.0f, 50.0f, 90.0f, 100.0f}) {
            std::vector<float> qt(rows);
            faiss::build_qtable(rows, quality, qt.data());
            for (int f = 0; f < rows; f++) {
                EXPECT_GE(qt[f], 1.0f) << "rows=" << rows << " quality=" << quality;
                if (f > 0) {
                    EXPECT_GE(qt[f], qt[f - 1])
                            << "rows=" << rows << " quality=" << quality << " f=" << f;
                }
            }
        }
    }
}

TEST(DCTMatrix, QtableHigherQualityIsFiner) {
    // Higher quality -> smaller (or equal) step sizes everywhere, i.e.
    // less aggressive quantization.
    const int rows = 32;
    std::vector<float> qt_low(rows), qt_high(rows);
    faiss::build_qtable(rows, 10.0f, qt_low.data());
    faiss::build_qtable(rows, 90.0f, qt_high.data());
    for (int f = 0; f < rows; f++) {
        EXPECT_LE(qt_high[f], qt_low[f]) << "f=" << f;
    }
}

// ---------------------------------------------------------------------
// EntropyCoder: pure zlib wrapper, no Index dependency.
// ---------------------------------------------------------------------

TEST(EntropyCoder, RoundTrip) {
    for (size_t size : {size_t(0), size_t(1), size_t(7), size_t(1000)}) {
        std::vector<uint8_t> data(size);
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<uint8_t>((i * 37) % 256);
        }
        auto compressed = faiss::zlib_compress(data.data(), data.size());
        std::vector<uint8_t> decoded(size);
        faiss::zlib_decompress(
                compressed.data(), compressed.size(), decoded.data(), size);
        EXPECT_EQ(data, decoded) << "size=" << size;
    }
}

TEST(EntropyCoder, CompressesRunsOfZeros) {
    // The actual property this module is relied on for: a byte buffer
    // that's mostly zero (as quantized coefficients at typical qualities
    // are) should compress to much less than its raw size.
    std::vector<uint8_t> data(1000, 0);
    for (int i = 0; i < 20; i++) {
        data[i * 37 % 1000] = static_cast<uint8_t>(i + 1);
    }
    auto compressed = faiss::zlib_compress(data.data(), data.size());
    EXPECT_LT(compressed.size(), data.size() / 4);
}

// ---------------------------------------------------------------------
// IndexPCATemporalDCT
// ---------------------------------------------------------------------

TEST(IndexPCATemporalDCT, ConstructorValidatesArgsAndStartsUntrained) {
    EXPECT_NO_THROW(faiss::IndexPCATemporalDCT(16, 4, 90.0f, 1.0f));
    EXPECT_THROW(
            faiss::IndexPCATemporalDCT(16, 0, 90.0f, 1.0f), faiss::FaissException);
    EXPECT_THROW(
            faiss::IndexPCATemporalDCT(16, 17, 90.0f, 1.0f), faiss::FaissException);
    EXPECT_THROW(
            faiss::IndexPCATemporalDCT(16, 4, 0.0f, 1.0f), faiss::FaissException);
    EXPECT_THROW(
            faiss::IndexPCATemporalDCT(16, 4, 101.0f, 1.0f), faiss::FaissException);
    EXPECT_THROW(
            faiss::IndexPCATemporalDCT(16, 4, 90.0f, 0.0f), faiss::FaissException);

    faiss::IndexPCATemporalDCT index(16, 4, 90.0f, 1.0f);
    EXPECT_FALSE(index.is_trained);
    EXPECT_EQ(index.ntotal, 0);
}

TEST(IndexPCATemporalDCT, TrainSetsIsTrained) {
    const int d = 16, n_components = 4;
    faiss::IndexPCATemporalDCT index(d, n_components, 90.0f, 1.0f);
    auto train = make_random(200, d, 1);
    index.train(200, train.data());
    EXPECT_TRUE(index.is_trained);
    EXPECT_EQ(index.pca.mean.size(), static_cast<size_t>(d));
    EXPECT_EQ(index.pca.A.size(), static_cast<size_t>(n_components) * d);
}

TEST(IndexPCATemporalDCT, AddBeforeTrainThrows) {
    faiss::IndexPCATemporalDCT index(16, 4, 90.0f, 1.0f);
    auto x = make_random(10, 16, 2);
    EXPECT_THROW(index.add(10, x.data()), faiss::FaissException);
}

TEST(IndexPCATemporalDCT, SingleRowSegmentsReconstructExactly) {
    // With a vanishingly small threshold, essentially every step exceeds
    // it, forcing every segment down to a single row. A single-row
    // segment's per-component std is exactly 0 (only one sample), so
    // std_safe clamps to 1.0 and normalized = (row - mean)/1 = 0
    // (mean == the row itself for a 1-row segment) -- every DCT
    // coefficient is then exactly 0, quantizes to exactly 0, and
    // dequantizes back to exactly 0, so the DCT+quantization path
    // reconstructs its (PCA-projected) input exactly, independent of
    // `quality`. This is a fully deterministic, closed-form case, good
    // for pinning down exact behavior without needing an external
    // numerical oracle.
    //
    // n_components == d here specifically so PCA itself is a lossless
    // (full-rank, orthonormal) rotation with no separate truncation
    // error -- isolating the DCT/quantization-path claim above. With
    // n_components < d, full-d reconstruct() would *also* carry the
    // PCA truncation's own separate, unavoidable loss on top of this.
    const int d = 12, n_components = 12, n = 40;
    faiss::IndexPCATemporalDCT index(d, n_components, 50.0f, /*threshold=*/1e-6f);
    auto train = make_random(200, d, 3);
    index.train(200, train.data());

    auto seq = make_random(n, d, 4);
    index.add(n, seq.data());

    EXPECT_EQ(static_cast<int>(index.segments.size()), n)
            << "expected every segment to collapse to a single row";
    for (auto& seg : index.segments) {
        EXPECT_EQ(seg.rows, 1);
        // every coefficient is exactly 0 (see the comment above), so the
        // trailing-zero-truncation logic should find no nonzero row at
        // all and store nothing.
        EXPECT_EQ(seg.stored_rows, 0);
        EXPECT_TRUE(seg.compressed_quantized.empty());
    }

    std::vector<float> recon(n * d);
    index.reconstruct_n(0, n, recon.data());
    for (int i = 0; i < n * d; i++) {
        EXPECT_NEAR(recon[i], seq[i], 1e-3) << "i=" << i;
    }
}

TEST(IndexPCATemporalDCT, SegmentationBoundaryAtDeliberateJump) {
    // A smoothly-varying run of rows followed by a hard jump should
    // produce a segment boundary exactly at the jump, for a threshold
    // set between the smooth-region step size and the jump size.
    const int d = 20, n_components = 4, n_smooth = 30, n_jump_region = 20;
    const int n = n_smooth + n_jump_region;

    faiss::IndexPCATemporalDCT index(d, n_components, 90.0f, /*threshold=*/3.0f);
    auto train = make_random(300, d, 5);
    index.train(300, train.data());

    // Smoothly drifting sequence (random walk, small steps) followed by
    // a large constant offset -- matches how real video embeddings
    // behave (small frame-to-frame drift, occasional scene change).
    std::vector<float> seq(n * d, 0.0f);
    auto steps = make_random(n, d, 6);
    for (int i = 1; i < n_smooth; i++) {
        for (int j = 0; j < d; j++) {
            seq[i * d + j] = seq[(i - 1) * d + j] + 0.05f * steps[i * d + j];
        }
    }
    for (int i = n_smooth; i < n; i++) {
        for (int j = 0; j < d; j++) {
            seq[i * d + j] = seq[(n_smooth - 1) * d + j] + 50.0f;
        }
    }

    index.add(n, seq.data());

    bool boundary_at_jump = false;
    for (auto& seg : index.segments) {
        if (seg.start_id == n_smooth) {
            boundary_at_jump = true;
        }
        // no segment should straddle the jump
        EXPECT_FALSE(seg.start_id < n_smooth && seg.start_id + seg.rows > n_smooth);
    }
    EXPECT_TRUE(boundary_at_jump);
    // the smooth region (small steps well under the threshold) should
    // have merged into segment(s) longer than 1 row
    EXPECT_GT(index.segments.front().rows, 1);
}

TEST(IndexPCATemporalDCT, TruncationAndEntropyCodingShrinkStorage) {
    // A long, smoothly-drifting (temporally correlated, not iid) segment
    // at a middling quality should have its DCT energy concentrated in
    // the early (low-frequency) rows -- the real-corpus property
    // Segment::stored_rows's docs describe -- so trailing rows should be
    // exactly zero and go unstored, and what *is* stored should compress
    // well below its dense size.
    const int d = 24, n_components = 6, n = 200;
    faiss::IndexPCATemporalDCT index(d, n_components, 50.0f, /*threshold=*/1e6f);
    auto train = make_random(300, d, 20);
    index.train(300, train.data());

    std::vector<float> seq(n * d, 0.0f);
    auto steps = make_random(n, d, 21);
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < d; j++) {
            seq[i * d + j] = seq[(i - 1) * d + j] + 0.05f * steps[i * d + j];
        }
    }
    index.add(n, seq.data());

    ASSERT_EQ(index.segments.size(), 1u) << "threshold set high enough that nothing should split";
    const auto& seg = index.segments.front();
    EXPECT_EQ(seg.rows, n);
    EXPECT_LT(seg.stored_rows, seg.rows)
            << "expected a nonzero trailing-zero region to be truncated away";

    const size_t dense_bytes = static_cast<size_t>(seg.rows) * n_components; // int8
    EXPECT_LT(seg.compressed_quantized.size(), dense_bytes / 2)
            << "truncation + zlib together should beat dense int8 storage by a solid margin here";
}

TEST(IndexPCATemporalDCT, SearchReturnsSelfAsTopMatch) {
    // Uses the same single-row-segment (lossless in the *reduced* space)
    // construction as above, but with n_components < d (the realistic,
    // truncating case), to test the ranking-correctness claim from the
    // class docs in its general form. Reduced-space single-row segments
    // still decode to exactly their own projected value (see the
    // reconstruction test above for why), so a query taken from the
    // corpus's own row i is always at reduced-space distance exactly 0
    // from reduced_vectors[i] -- the minimum possible -- so it must rank
    // first, REGARDLESS of PCA truncation. The reported *distance*,
    // however, is NOT ~0 here (unlike the n_components==d case): search()
    // adds back ||q-mean||^2 - ||xq_reduced||^2, the part of q's own
    // energy that falls outside the PCA subspace, which is generally
    // nonzero whenever n_components < d. That residual is exactly what a
    // direct reconstruct()-based full-d L2 computation gives, so check
    // the reported distance against that instead of against 0.
    const int d = 10, n_components = 4, n = 25;
    faiss::IndexPCATemporalDCT index(d, n_components, 50.0f, /*threshold=*/1e-6f);
    auto train = make_random(200, d, 7);
    index.train(200, train.data());
    auto seq = make_random(n, d, 8);
    index.add(n, seq.data());

    const int k = 5;
    std::vector<float> distances(k);
    std::vector<faiss::idx_t> labels(k);
    const int query_row = 12;
    const float* query = seq.data() + query_row * d;
    index.search(1, query, k, distances.data(), labels.data());

    EXPECT_EQ(labels[0], query_row);

    std::vector<float> recon(d);
    index.reconstruct(query_row, recon.data());
    double expected_dist_sq = 0.0;
    for (int j = 0; j < d; j++) {
        double diff = query[j] - recon[j];
        expected_dist_sq += diff * diff;
    }
    EXPECT_NEAR(distances[0], expected_dist_sq, 1e-2);

    for (int j = 1; j < k; j++) {
        EXPECT_GE(distances[j], distances[j - 1]);
    }
}

TEST(IndexPCATemporalDCT, ReconstructBatchMatchesPerRow) {
    const int d = 14, n_components = 5, n = 50;
    faiss::IndexPCATemporalDCT index(d, n_components, 75.0f, 1.0f);
    auto train = make_random(200, d, 9);
    index.train(200, train.data());
    auto seq = make_random(n, d, 10);
    index.add(n, seq.data());

    std::vector<float> batch(n * d);
    index.reconstruct_n(0, n, batch.data());
    for (int i = 0; i < n; i++) {
        std::vector<float> single(d);
        index.reconstruct(i, single.data());
        for (int j = 0; j < d; j++) {
            EXPECT_FLOAT_EQ(single[j], batch[i * d + j]);
        }
    }
}

TEST(IndexPCATemporalDCT, ResetClearsState) {
    const int d = 12, n_components = 3, n = 20;
    faiss::IndexPCATemporalDCT index(d, n_components, 90.0f, 1.0f);
    auto train = make_random(200, d, 11);
    index.train(200, train.data());
    auto seq = make_random(n, d, 12);
    index.add(n, seq.data());
    ASSERT_GT(index.ntotal, 0);

    index.reset();
    EXPECT_EQ(index.ntotal, 0);
    EXPECT_TRUE(index.segments.empty());
    EXPECT_TRUE(index.reduced_vectors.empty());
    // training state survives reset() (matches Index::reset()'s
    // "removes all elements from the database" contract, not "untrain")
    EXPECT_TRUE(index.is_trained);

    // index is usable again after reset
    EXPECT_NO_THROW(index.add(n, seq.data()));
    EXPECT_EQ(index.ntotal, n);
}

TEST(IndexPCATemporalDCT, StorageBytesTracksAddedData) {
    const int d = 16, n_components = 4;
    faiss::IndexPCATemporalDCT index(d, n_components, 90.0f, 1e6f);
    auto train = make_random(200, d, 30);
    index.train(200, train.data());

    const size_t trained_only = index.storage_bytes();
    // the PCA basis alone (mean + components) accounts for this exactly
    EXPECT_EQ(trained_only, (d + static_cast<size_t>(n_components) * d) * sizeof(float));

    auto seq1 = make_random(20, d, 31);
    index.add(20, seq1.data());
    const size_t after_first_add = index.storage_bytes();
    EXPECT_GT(after_first_add, trained_only);

    auto seq2 = make_random(20, d, 32);
    index.add(20, seq2.data());
    EXPECT_GT(index.storage_bytes(), after_first_add);

    index.reset();
    EXPECT_EQ(index.storage_bytes(), trained_only)
            << "reset() clears segments but keeps the (already-counted) PCA basis";
}

TEST(IndexPCATemporalDCT, TwoAddCallsAreIndependentSequences) {
    // Per the class-level API contract, each add() call is its own
    // temporally-contiguous sequence -- segmentation must not treat the
    // last row of one add() call as adjacent to the first row of the
    // next.
    const int d = 10, n_components = 3, n1 = 15, n2 = 15;
    faiss::IndexPCATemporalDCT index(d, n_components, 90.0f, 1e-6f);
    auto train = make_random(200, d, 13);
    index.train(200, train.data());

    auto seq1 = make_random(n1, d, 14);
    auto seq2 = make_random(n2, d, 15);
    index.add(n1, seq1.data());
    index.add(n2, seq2.data());

    EXPECT_EQ(index.ntotal, n1 + n2);
    // with threshold ~0, every segment should still be a single row
    // (had the two calls been merged, a boundary could have been
    // skipped by coincidence, but each call's *first* row always opens
    // its own fresh segment regardless)
    bool found_seg_at_n1 = false;
    for (auto& seg : index.segments) {
        if (seg.start_id == n1) {
            found_seg_at_n1 = true;
        }
    }
    EXPECT_TRUE(found_seg_at_n1);
}
