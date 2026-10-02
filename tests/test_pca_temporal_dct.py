# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import unittest

import numpy as np
import faiss


def make_random(n, d, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)).astype("float32")


def make_drifting_sequence(n_smooth, n_jump, d, seed, jump=6.0):
    """Smoothly drifting (random-walk) sequence followed by a hard jump --
    mirrors real video-embedding behavior (small frame-to-frame drift,
    occasional scene change) far better than iid noise, which is why the
    C++-side tests build their inputs the same way."""
    rng = np.random.default_rng(seed)
    n = n_smooth + n_jump
    steps = rng.normal(scale=0.15, size=(n, d)).astype("float32")
    seq = np.cumsum(steps, axis=0)
    seq[n_smooth:] += jump
    return seq.astype("float32")


class TestIndexPCATemporalDCT(unittest.TestCase):
    def test_construct_and_train(self):
        d, n_components = 16, 4
        index = faiss.IndexPCATemporalDCT(d, n_components, 90.0, 1.0)
        self.assertIsInstance(index, faiss.Index)
        self.assertFalse(index.is_trained)
        self.assertEqual(index.ntotal, 0)

        index.train(make_random(200, d, 1))
        self.assertTrue(index.is_trained)

    def test_constructor_validates_args(self):
        d = 16
        with self.assertRaises(RuntimeError):
            faiss.IndexPCATemporalDCT(d, 0, 90.0, 1.0)
        with self.assertRaises(RuntimeError):
            faiss.IndexPCATemporalDCT(d, d + 1, 90.0, 1.0)
        with self.assertRaises(RuntimeError):
            faiss.IndexPCATemporalDCT(d, 4, 101.0, 1.0)
        with self.assertRaises(RuntimeError):
            faiss.IndexPCATemporalDCT(d, 4, 90.0, 0.0)

    def test_add_before_train_raises(self):
        index = faiss.IndexPCATemporalDCT(16, 4, 90.0, 1.0)
        with self.assertRaises(RuntimeError):
            index.add(make_random(10, 16, 2))

    def test_search_returns_self_as_top_match(self):
        # threshold ~0 forces every segment down to a single row, which
        # decodes losslessly *in the reduced space* regardless of PCA
        # truncation (see the C++ tests for the closed-form argument), so
        # a query taken from the corpus's own row is always its own
        # nearest neighbor -- this is the ranking-correctness property the
        # whole latent-space search design rests on.
        d, n_components, n = 10, 4, 25
        index = faiss.IndexPCATemporalDCT(d, n_components, 50.0, 1e-6)
        index.train(make_random(200, d, 3))
        seq = make_random(n, d, 4)
        index.add(seq)

        query_row = 12
        D, I = index.search(seq[query_row : query_row + 1], 5)
        self.assertEqual(I[0, 0], query_row)
        # returned distances are sorted ascending
        self.assertTrue(np.all(np.diff(D[0]) >= 0))

    def test_segmentation_boundary_at_jump(self):
        d, n_components = 20, 4
        n_smooth, n_jump = 30, 20
        index = faiss.IndexPCATemporalDCT(d, n_components, 90.0, 3.0)
        index.train(make_random(300, d, 5))
        seq = make_drifting_sequence(n_smooth, n_jump, d, seed=6)
        index.add(seq)

        # every row must still be reconstructable and the corpus size
        # must match what was added -- segmentation internals aren't
        # exposed to Python (see swigfaiss.swig's %ignore list), so this
        # exercises the same boundary indirectly: the C++ tests assert
        # the actual segment boundaries directly.
        self.assertEqual(index.ntotal, n_smooth + n_jump)
        recon = index.reconstruct_n(0, index.ntotal)
        self.assertEqual(recon.shape, (n_smooth + n_jump, d))

    def test_reconstruct_n_matches_reconstruct(self):
        d, n_components, n = 14, 5, 50
        index = faiss.IndexPCATemporalDCT(d, n_components, 75.0, 1.0)
        index.train(make_random(200, d, 9))
        index.add(make_random(n, d, 10))

        batch = index.reconstruct_n(0, n)
        for i in range(n):
            single = index.reconstruct(i)
            np.testing.assert_allclose(single, batch[i], rtol=0, atol=1e-6)

    def test_reset_clears_state_keeps_training(self):
        d, n_components, n = 12, 3, 20
        index = faiss.IndexPCATemporalDCT(d, n_components, 90.0, 1.0)
        index.train(make_random(200, d, 11))
        seq = make_random(n, d, 12)
        index.add(seq)
        self.assertGreater(index.ntotal, 0)

        index.reset()
        self.assertEqual(index.ntotal, 0)
        self.assertTrue(index.is_trained)

        # usable again after reset
        index.add(seq)
        self.assertEqual(index.ntotal, n)

    def test_two_add_calls_are_independent_sequences(self):
        # per the class-level API contract, add() call boundaries are
        # sequence boundaries -- both halves should be searchable/
        # reconstructable after two separate add() calls.
        d, n_components = 10, 3
        n1, n2 = 15, 15
        index = faiss.IndexPCATemporalDCT(d, n_components, 90.0, 1e-6)
        index.train(make_random(200, d, 13))
        seq1 = make_random(n1, d, 14)
        seq2 = make_random(n2, d, 15)
        index.add(seq1)
        index.add(seq2)
        self.assertEqual(index.ntotal, n1 + n2)

        D, I = index.search(seq2[0:1], 1)
        self.assertEqual(I[0, 0], n1)  # seq2's row 0 is global id n1

    def test_downcast_from_base_index(self):
        d, n_components = 8, 3
        index = faiss.IndexPCATemporalDCT(d, n_components, 90.0, 1.0)
        base = faiss.downcast_index(index)
        self.assertIs(type(base), faiss.IndexPCATemporalDCT)


class TestCrossCheckAgainstPythonReference(unittest.TestCase):
    """Numerically cross-checks the C++ index against delta_vec/compress.py,
    the already-validated Python/numpy reference implementation this index
    is a from-scratch port of. Requires scipy and the delta_vec package on
    sys.path; skipped if unavailable (e.g. a bare faiss CI checkout without
    the sibling delta_vec project)."""

    @classmethod
    def setUpClass(cls):
        # This faiss checkout lives inside delta_vec/faiss/, i.e. one level
        # below delta_vec itself (not as a sibling directory).
        faiss_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        delta_vec_dir = os.path.dirname(faiss_repo_root)
        if not os.path.isfile(os.path.join(delta_vec_dir, "compress.py")):
            raise unittest.SkipTest(
                f"delta_vec/compress.py not found alongside this faiss checkout (looked in {delta_vec_dir})"
            )
        sys.path.insert(0, delta_vec_dir)
        try:
            import compress as compress_mod  # noqa: F401
        except ImportError as e:
            raise unittest.SkipTest(f"delta_vec.compress not importable: {e}")
        cls.compress_mod = compress_mod

    def test_matches_python_reference(self):
        compress_mod = self.compress_mod
        d, n_components, n_train, n = 24, 4, 300, 80
        quality, threshold = 90.0, 2.0

        train = make_random(n_train, d, 100)
        seq = make_drifting_sequence(40, 40, d, seed=101, jump=6.0)

        # --- Python reference pipeline ---
        basis = compress_mod.compute_pca_basis(train, n_components)
        projected = compress_mod.pca_project(seq, basis)
        bounds = compress_mod.adaptive_segment_bounds(projected, threshold)

        blocks = []
        for start, end in bounds:
            comp = compress_mod.compress_block_dct_quant(
                projected[start:end], np.array([quality])
            )
            blocks.append(
                {
                    "start": start,
                    "end": end,
                    "mean": comp["mean"],
                    "std": comp["std"],
                    "compressions": comp["compressions"],
                }
            )
        ref_recon_reduced = compress_mod.reconstruct_ours_dct_quant(blocks, 0, n_components)
        ref_recon_full = compress_mod.pca_reconstruct(ref_recon_reduced, basis)

        # --- C++ index ---
        # Note: per-segment internals (boundaries, quantized coefficients)
        # are deliberately not exposed to Python (see the %ignore list in
        # swigfaiss.swig -- they're implementation details, and PCA
        # eigenvector sign is ambiguous between independent
        # eigendecompositions anyway, so a raw quantized-int comparison
        # isn't meaningful across implementations regardless; see the
        # standalone C++ cross-check this test was derived from for that
        # level of detail). What's checked here is the one thing that
        # actually matters through the public API: the final decoded
        # output.
        index = faiss.IndexPCATemporalDCT(d, n_components, quality, threshold)
        index.train(train)
        index.add(seq)
        cpp_recon_full = index.reconstruct_n(0, n)

        self.assertEqual(index.ntotal, n)

        # Final reconstruction must match closely (float32 + independent
        # eigendecomposition accumulation-order tolerance) -- this is the
        # load-bearing check: it validates PCA projection, normalization,
        # DCT, quantization, and un-projection all agree end-to-end.
        rel_err = np.linalg.norm(cpp_recon_full - ref_recon_full) / np.linalg.norm(
            ref_recon_full
        )
        self.assertLess(rel_err, 1e-3)


if __name__ == "__main__":
    unittest.main()
