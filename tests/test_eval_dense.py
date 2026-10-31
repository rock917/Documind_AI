import faiss
import numpy as np

from documind.retrieval.eval_dense import dense_order, hit_array, rrf_order


def test_dense_order_ranks_by_inner_product():
    vecs = np.array([[1, 0], [0, 1], [0.7071, 0.7071]], dtype=np.float32)
    index = faiss.IndexFlatIP(2)
    index.add(vecs)
    assert dense_order(index, np.array([0.0, 1.0], dtype=np.float32)) == [1, 2, 0]
    assert dense_order(index, np.array([1.0, 0.0], dtype=np.float32)) == [0, 2, 1]


def test_dense_order_returns_every_chunk():
    index = faiss.IndexFlatIP(3)
    index.add(np.eye(3, dtype=np.float32))
    assert sorted(dense_order(index, np.array([1.0, 1.0, 1.0], dtype=np.float32))) == [0, 1, 2]


def test_rrf_prefers_chunks_ranked_high_in_both():
    assert rrf_order([[2, 0, 1], [2, 1, 0]])[0] == 2


def test_rrf_exact_order_with_tie():
    # chunks 0 and 2 each rank first in one list and last in the other: 1/61 + 1/63 = 0.032266 (tied, lower index first)
    # chunk 1 is second in both: 2/62 = 0.032258, slightly lower
    assert rrf_order([[0, 1, 2], [2, 1, 0]]) == [0, 2, 1]


def test_rrf_contains_all_items_once():
    out = rrf_order([[3, 1, 0, 2], [0, 2, 3, 1]])
    assert sorted(out) == [0, 1, 2, 3]


def test_hit_array():
    assert hit_array([1, 5, None, 6], 5).tolist() == [1, 1, 0, 0]