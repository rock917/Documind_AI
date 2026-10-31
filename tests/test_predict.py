import numpy as np
import torch
from transformers import BertConfig, BertForQuestionAnswering

from documind.eval.predict import window_logits


def tiny_model():
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=64)
    return BertForQuestionAnswering(cfg).eval()


def make_window(length, seed):
    rng = np.random.default_rng(seed)
    ids = [2] + [int(x) for x in rng.integers(5, 100, size=length - 2)] + [3]
    return {"input_ids": ids, "token_type_ids": [0] * 4 + [1] * (length - 4)}


def test_batched_logits_match_single_window_logits_despite_padding():
    model = tiny_model()
    wins = [make_window(n, i) for i, n in enumerate([6, 11, 9, 14, 7])]
    batched = window_logits(model, wins, batch_size=3)
    for w, (bs, be) in zip(wins, batched):
        (ss, se), = window_logits(model, [w], batch_size=1)
        assert len(bs) == len(w["input_ids"]) == len(ss)
        np.testing.assert_allclose(bs, ss, atol=1e-4)
        np.testing.assert_allclose(be, se, atol=1e-4)


def test_one_result_per_window_in_order():
    model = tiny_model()
    wins = [make_window(n, i) for i, n in enumerate([8, 5, 12])]
    out = window_logits(model, wins, batch_size=2)
    assert [len(s) for s, _ in out] == [8, 5, 12]