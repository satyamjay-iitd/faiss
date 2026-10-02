"""
Benchmark: RaBitQ vs. IndexPCATemporalDCT on the real Argoverse embeddings
corpus (72,505 frames x 1024 dims, 227 driving-clip logs).

Compares indexing time (train + add) and query time (single-query, the
realistic one-at-a-time usage pattern, plus a batch figure for reference),
alongside a quick recall@k sanity check so the timings have context.

Needs the freshly-built faiss module from this checkout (not delta_vec's
pip-installed faiss-cpu) -- see demo_pca_temporal_dct.py's docstring for the
one-time build steps. Run from the faiss repo root:

    PYTHONPATH=build/faiss/python/build/lib python3 \
        bench_pca_temporal_dct_vs_rabitq.py
"""

import json
import time
from pathlib import Path

import numpy as np

import faiss

ROOT = Path(__file__).resolve().parent.parent.parent  # .../vecdb-temporal-redundancy
EMBEDDINGS_PATH = ROOT / "argoverse_embeddings.npy"
META_PATH = ROOT / "argoverse_embeddings_meta.json"

N_COMPONENTS = 119  # ~90% of corpus variance in this dataset (see delta_vec EDA)
QUALITY = 100.0
THRESHOLD = 0.25  # in PCA-projected space; matches delta_vec's adaptive+PCA sweep

K = 10
N_QUERIES = 100
SEED = 42


def log_row_ranges(meta: list[dict]) -> list[tuple[int, int]]:
    """Contiguous [start, end) row range per log, in the order logs first
    appear -- each range is one add() call's worth of temporally-
    contiguous frames, per IndexPCATemporalDCT's API contract."""
    ranges = []
    start = 0
    prev = meta[0]["log_id"]
    for i, m in enumerate(meta):
        if m["log_id"] != prev:
            ranges.append((start, i))
            start = i
            prev = m["log_id"]
    ranges.append((start, len(meta)))
    return ranges


def time_single_queries(search_fn, queries, k, warmup=10):
    n = queries.shape[0]
    for i in range(warmup):
        search_fn(queries[i % n : i % n + 1], k)
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        search_fn(queries[i : i + 1], k)
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    return times.mean(), np.median(times), times.min()


def recall_at_k(gt_ids, pred_ids):
    hits = sum(len(set(gt_ids[i]) & set(pred_ids[i])) for i in range(len(gt_ids)))
    return hits / (len(gt_ids) * gt_ids.shape[1])


def main():
    print(f"faiss module: {faiss.__file__}")

    t0 = time.perf_counter()
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    meta = json.loads(META_PATH.read_text())
    t_load = time.perf_counter() - t0
    n, d = embeddings.shape
    print(f"loaded {n:,} frames x {d} dims from {EMBEDDINGS_PATH.name} in {t_load:.2f}s")
    print(f"(this load time is common to both methods below and excluded from the comparison)")
    print()

    ranges = log_row_ranges(meta)
    print(f"{len(ranges)} logs (temporally-contiguous sequences)")
    print()

    rng = np.random.default_rng(SEED)
    query_idx = rng.choice(n, size=N_QUERIES, replace=False)
    queries = embeddings[query_idx]

    # --- ground truth (for the recall sanity check only) ---
    gt_index = faiss.IndexFlatL2(d)
    gt_index.add(embeddings)
    _, gt_ids = gt_index.search(queries, K)

    results = {}

    # ============================== RaBitQ ==============================
    print("=" * 60)
    print("RaBitQ")
    print("=" * 60)
    t0 = time.perf_counter()
    rabitq_index = faiss.IndexRaBitQ(d)
    rabitq_index.train(embeddings)
    t_rabitq_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    rabitq_index.add(embeddings)
    t_rabitq_add = time.perf_counter() - t0

    print(f"  train: {t_rabitq_train:.3f}s")
    print(f"  add:   {t_rabitq_add:.3f}s  ({rabitq_index.ntotal:,} vectors, one call)")
    print(f"  index time (train+add): {t_rabitq_train + t_rabitq_add:.3f}s")

    m, md, mn = time_single_queries(rabitq_index.search, queries, K)
    print(f"  single-query search: mean={m*1000:.4f}ms  median={md*1000:.4f}ms  min={mn*1000:.4f}ms")

    t0 = time.perf_counter()
    _, rabitq_pred_ids = rabitq_index.search(queries, K)
    t_batch = time.perf_counter() - t0
    print(f"  batch({N_QUERIES}) search: {t_batch*1000:.2f}ms total, {t_batch/N_QUERIES*1000:.4f}ms/query")

    recall = recall_at_k(gt_ids, rabitq_pred_ids)
    print(f"  recall@{K}: {recall:.4f}")

    rabitq_bytes = rabitq_index.ntotal * rabitq_index.code_size + d * 4  # + centroid
    original_bytes = n * d * 4
    rabitq_ratio = original_bytes / rabitq_bytes
    print(f"  storage: {rabitq_bytes:,} bytes -> {rabitq_ratio:.2f}x compression")

    results["rabitq"] = {
        "train_s": t_rabitq_train, "add_s": t_rabitq_add,
        "query_mean_ms": m * 1000, "query_median_ms": md * 1000,
        "batch_ms_per_query": t_batch / N_QUERIES * 1000, "recall": recall,
        "bytes": rabitq_bytes, "ratio": rabitq_ratio,
    }
    print()

    # ======================= IndexPCATemporalDCT =======================
    print("=" * 60)
    print(f"IndexPCATemporalDCT (n_components={N_COMPONENTS}, quality={QUALITY}, threshold={THRESHOLD})")
    print("=" * 60)
    ours_index = faiss.IndexPCATemporalDCT(d, N_COMPONENTS, QUALITY, THRESHOLD)

    t0 = time.perf_counter()
    ours_index.train(embeddings)
    t_ours_train = time.perf_counter() - t0

    # One add() call per log, respecting the "one call = one temporally-
    # contiguous sequence" API contract (unlike RaBitQ, which has no such
    # constraint and can ingest the whole corpus in one call).
    t0 = time.perf_counter()
    for start, end in ranges:
        ours_index.add(embeddings[start:end])
    t_ours_add = time.perf_counter() - t0

    print(f"  train: {t_ours_train:.3f}s")
    print(f"  add:   {t_ours_add:.3f}s  ({ours_index.ntotal:,} vectors, {len(ranges)} calls -- one per log)")
    print(f"  index time (train+add): {t_ours_train + t_ours_add:.3f}s")

    m, md, mn = time_single_queries(ours_index.search, queries, K)
    print(f"  single-query search: mean={m*1000:.4f}ms  median={md*1000:.4f}ms  min={mn*1000:.4f}ms")

    t0 = time.perf_counter()
    _, ours_pred_ids = ours_index.search(queries, K)
    t_batch = time.perf_counter() - t0
    print(f"  batch({N_QUERIES}) search: {t_batch*1000:.2f}ms total, {t_batch/N_QUERIES*1000:.4f}ms/query")

    recall = recall_at_k(gt_ids, ours_pred_ids)
    print(f"  recall@{K}: {recall:.4f}")

    ours_bytes = ours_index.storage_bytes()
    original_bytes = n * d * 4
    ours_ratio = original_bytes / ours_bytes
    print(f"  storage: {ours_bytes:,} bytes -> {ours_ratio:.2f}x compression")

    results["ours"] = {
        "train_s": t_ours_train, "add_s": t_ours_add,
        "query_mean_ms": m * 1000, "query_median_ms": md * 1000,
        "batch_ms_per_query": t_batch / N_QUERIES * 1000, "recall": recall,
        "bytes": ours_bytes, "ratio": ours_ratio,
    }
    print()

    # ============================== summary ==============================
    print("=" * 60)
    print("summary")
    print("=" * 60)
    r, o = results["rabitq"], results["ours"]
    print(f"{'':30s}{'RaBitQ':>15s}{'Ours':>15s}")
    print(f"{'index time (train+add) s':30s}{r['train_s']+r['add_s']:>15.3f}{o['train_s']+o['add_s']:>15.3f}")
    print(f"{'  train s':30s}{r['train_s']:>15.3f}{o['train_s']:>15.3f}")
    print(f"{'  add s':30s}{r['add_s']:>15.3f}{o['add_s']:>15.3f}")
    print(f"{'single-query mean ms':30s}{r['query_mean_ms']:>15.4f}{o['query_mean_ms']:>15.4f}")
    print(f"{'single-query median ms':30s}{r['query_median_ms']:>15.4f}{o['query_median_ms']:>15.4f}")
    print(f"{'batch ms/query':30s}{r['batch_ms_per_query']:>15.4f}{o['batch_ms_per_query']:>15.4f}")
    print(f"{'recall@'+str(K):30s}{r['recall']:>15.4f}{o['recall']:>15.4f}")
    print(f"{'storage bytes':30s}{r['bytes']:>15,}{o['bytes']:>15,}")
    print(f"{'compression ratio':30s}{r['ratio']:>14.2f}x{o['ratio']:>14.2f}x")


if __name__ == "__main__":
    main()
