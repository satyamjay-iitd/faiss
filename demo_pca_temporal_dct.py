"""
Standalone demo/smoke-test for faiss.IndexPCATemporalDCT.

This is the fastest way to try the new index interactively -- it doesn't
require the delta_vec project or its own faiss-cpu install; it only needs
the freshly-built faiss module produced by this checkout's own build.

Setup (one-time, ~2-5 min):
    cd faiss
    mkdir -p build && cd build
    cmake .. -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON \
        -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=generic
        # (needs a `swig` binary on PATH; `pip install swig` works if your
        # system doesn't have one and you can't `apt install swig`)
    make -j"$(nproc)" swigfaiss
    cd faiss/python && python3 setup.py build   # produces build/lib/faiss/

Run this script with that built package on PYTHONPATH, e.g. from the faiss
repo root:
    PYTHONPATH=build/faiss/python/build/lib python3 demo_pca_temporal_dct.py

(No LD_LIBRARY_PATH tricks needed -- `ldd _swigfaiss.so` has no unresolved
dependencies once built this way.)
"""

import numpy as np

import faiss


def make_drifting_sequence(n_smooth, n_jump, d, seed, jump=6.0):
    """A smoothly drifting (random-walk) run of frames followed by a hard
    jump -- stands in for a real video: small frame-to-frame drift most of
    the time, with an occasional scene change."""
    rng = np.random.default_rng(seed)
    n = n_smooth + n_jump
    steps = rng.normal(scale=0.15, size=(n, d)).astype("float32")
    seq = np.cumsum(steps, axis=0)
    seq[n_smooth:] += jump
    return seq.astype("float32")


def main():
    d = 64
    n_components = 12
    quality = 90.0
    threshold = 2.0

    print(f"faiss module: {faiss.__file__}")
    print(f"d={d} n_components={n_components} quality={quality} threshold={threshold}")
    print()

    # --- construct + train ---
    index = faiss.IndexPCATemporalDCT(d, n_components, quality, threshold)
    print(f"constructed: is_trained={index.is_trained}, ntotal={index.ntotal}")

    rng = np.random.default_rng(0)
    train = rng.normal(size=(1000, d)).astype("float32")
    index.train(train)
    print(f"trained:     is_trained={index.is_trained}")
    print()

    # --- add one "video" (one temporally-contiguous sequence) ---
    n_smooth, n_jump = 150, 100
    seq = make_drifting_sequence(n_smooth, n_jump, d, seed=1)
    index.add(seq)
    print(f"added {seq.shape[0]} frames -> ntotal={index.ntotal}")
    print()

    # --- search: query with one of the corpus's own frames ---
    query_row = 37
    k = 5
    D, I = index.search(seq[query_row : query_row + 1], k)
    print(f"search(query=row {query_row}), top-{k}:")
    for rank, (idx, dist) in enumerate(zip(I[0], D[0])):
        tag = "  <- query itself" if idx == query_row else ""
        print(f"  rank {rank}: id={idx:4d}  dist={dist:.4f}{tag}")
    print()

    # --- reconstruct: compare decoded vs. original ---
    recon = index.reconstruct_n(0, index.ntotal)
    err_per_row = np.linalg.norm(recon - seq, axis=1)
    print("reconstruction error (L2 per frame):")
    print(f"  mean={err_per_row.mean():.4f}  min={err_per_row.min():.4f}  max={err_per_row.max():.4f}")
    print()

    # --- a quick recall@k check against brute-force ground truth ---
    gt_index = faiss.IndexFlatL2(d)
    gt_index.add(seq)
    n_query = 30
    query_idx = rng.choice(seq.shape[0], size=n_query, replace=False)
    queries = seq[query_idx]

    gt_D, gt_I = gt_index.search(queries, k)
    our_D, our_I = index.search(queries, k)

    hits = sum(len(set(gt_I[i]) & set(our_I[i])) for i in range(n_query))
    recall = hits / (n_query * k)
    print(f"recall@{k} vs. brute-force ground truth, {n_query} random queries: {recall:.3f}")

    print()
    print("DEMO DONE")


if __name__ == "__main__":
    main()
