"""Offline smoke test of the QA fine-tuning stack. No downloads needed.

Checks the code paths the project depends on:
  fast tokenizer sliding windows (stride + overflow), offset mapping round-trip,
  SQuAD-v2 style no-answer windows (CLS target), datasets Dataset, Trainer, inference decode.
"""
import random, sys, tempfile, pathlib
import torch, transformers, datasets, tokenizers, numpy as np
from datasets import Dataset
from transformers import (BertConfig, BertForQuestionAnswering, BertTokenizerFast,
                          Trainer, TrainingArguments, default_data_collator)

print("python", sys.version.split()[0], "| torch", torch.__version__, "| transformers", transformers.__version__,
      "| datasets", datasets.__version__, "| tokenizers", tokenizers.__version__, "| numpy", np.__version__)
print("cuda available:", torch.cuda.is_available())

random.seed(0)
torch.manual_seed(0)
words = ("the agreement shall be governed by laws of state delaware this party may terminate for convenience "
         "upon thirty days notice liability fees paid term renewal indemnify and hold harmless").split()
letters = list("abcdefghijklmnopqrstuvwxyz")
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + letters + ["##" + c for c in letters] + words + ["?", "."]
tmp = pathlib.Path(tempfile.mkdtemp())
(tmp / "vocab.txt").write_text("\n".join(dict.fromkeys(vocab)), encoding="utf-8")
tok = BertTokenizerFast(vocab_file=str(tmp / "vocab.txt"), do_lower_case=True)
assert tok.is_fast

# ---- toy examples: long contexts (forces overflow), some with answers, some without
examples = []
for i in range(24):
    ctx_words = [random.choice(words) for _ in range(300)]
    has_ans = i % 3 != 0
    ans_text, ans_start = "", -1
    if has_ans:
        pos = random.randrange(20, 250)
        ctx_words[pos:pos + 4] = ["governed", "by", "laws", "of"]
        context = " ".join(ctx_words)
        ans_text = "governed by laws of"
        ans_start = context.index(ans_text)
    else:
        context = " ".join(ctx_words)
        context = context.replace("governed", "agreement")
    examples.append({"question": "governing law ?", "context": context,
                     "answer_text": ans_text, "answer_start": ans_start})

MAXLEN, STRIDE = 64, 16


def prepare(exs):
    enc = tok([e["question"] for e in exs], [e["context"] for e in exs], truncation="only_second",
              max_length=MAXLEN, stride=STRIDE, return_overflowing_tokens=True,
              return_offsets_mapping=True, padding="max_length")
    smap = enc.pop("overflow_to_sample_mapping")
    offs = enc.pop("offset_mapping")
    sp, ep = [], []
    for i, off in enumerate(offs):
        ex = exs[smap[i]]
        seq = enc.sequence_ids(i)
        cls = 0
        if ex["answer_start"] < 0:
            sp.append(cls)
            ep.append(cls)
            continue
        s_char = ex["answer_start"]
        e_char = s_char + len(ex["answer_text"])
        ts = seq.index(1)
        te = len(seq) - 1 - seq[::-1].index(1)
        if not (off[ts][0] <= s_char and off[te][1] >= e_char):
            sp.append(cls)
            ep.append(cls)
            continue  # answer not inside this window -> no-answer target
        while ts < len(off) and seq[ts] == 1 and off[ts][0] <= s_char:
            ts += 1
        sp.append(ts - 1)
        while seq[te] == 1 and off[te][1] >= e_char:
            te -= 1
        ep.append(te + 1)
    enc["start_positions"] = sp
    enc["end_positions"] = ep
    return enc, smap, offs


enc, smap, offs = prepare(examples)
n_feat = len(enc["input_ids"])
assert n_feat > len(examples), "sliding window did not produce overflow windows"
n_noans = sum(1 for s in enc["start_positions"] if s == 0)
n_ans = n_feat - n_noans
assert n_ans > 0 and n_noans > 0
print(f"windows: {len(examples)} examples -> {n_feat} features ({n_ans} with answer span, {n_noans} no-answer)")

# ---- offset round-trip: decoded span must equal the gold answer text exactly
checked = 0
for i in range(n_feat):
    s, e = enc["start_positions"][i], enc["end_positions"][i]
    if s == 0:
        continue
    ex = examples[smap[i]]
    got = ex["context"][offs[i][s][0]: offs[i][e][1]]
    assert got == ex["answer_text"], (got, ex["answer_text"])
    checked += 1
print(f"offset round-trip OK on {checked} answer windows")

# ---- Trainer
ds = Dataset.from_dict({k: v for k, v in enc.items() if k in
                        ("input_ids", "token_type_ids", "attention_mask", "start_positions", "end_positions")})
cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                 intermediate_size=64, max_position_embeddings=MAXLEN)
model = BertForQuestionAnswering(cfg)
args = TrainingArguments(output_dir=str(tmp / "out"), max_steps=4, per_device_train_batch_size=4,
                         gradient_accumulation_steps=2, learning_rate=1e-3, logging_steps=1, report_to="none",
                         save_strategy="no", eval_strategy="no", dataloader_num_workers=0, seed=42,
                         fp16=False, use_cpu=not torch.cuda.is_available())
tr = Trainer(model=model, args=args, train_dataset=ds, data_collator=default_data_collator)
out = tr.train()
loss = out.training_loss
assert np.isfinite(loss)
print("Trainer OK, training_loss =", round(loss, 4), "| device:", model.device)

# ---- inference decode (no-answer score from the CLS position)
model.eval()
batch = {k: torch.tensor(ds[:4][k]).to(model.device) for k in ("input_ids", "token_type_ids", "attention_mask")}
with torch.no_grad():
    o = model(**batch)
assert o.start_logits.shape == (4, MAXLEN) and o.end_logits.shape == (4, MAXLEN)
null_score = (o.start_logits[:, 0] + o.end_logits[:, 0]).tolist()
print("inference OK; no-answer scores (CLS start+end):", [round(x, 3) for x in null_score])
print("TRAIN SMOKE TEST PASSED")