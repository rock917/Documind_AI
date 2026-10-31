"""Show the highest-scoring mistakes for one category, to see WHY the reader goes wrong.

Lists present clauses where the predicted span is wrong (wrong place, or right place but too short/long),
most confident first, with gold and predicted text and where each sits in the contract.
Refuses test files (look at errors on val only, so the test set stays a clean final check).

Run from the project root (E:\\Documind) with the virtual environment active:
    python -m documind.eval.show_errors --pred data\\predictions\\legalbert_qa_val.jsonl --category "Effective Date" --max 6
"""
import argparse
import json

from documind.eval.error_analysis import classify_present, guard_not_test, shorten

NEG_INF = -1e9
ERROR_TYPES = ("wrong_place", "overlap_too_short", "overlap_too_long")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def select_errors(rows, category, error_types=ERROR_TYPES):
    """(error_type, row) pairs for present clauses of one category with a wrong span, most confident first."""
    found = []
    for r in rows:
        if r["category"] != category or not r["answers"]:
            continue
        t = classify_present(r)
        if t in error_types:
            found.append((t, r))
    found.sort(key=lambda tr: -(tr[1]["present_score"] if tr[1]["present_score"] is not None else NEG_INF))
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--category", required=True)
    p.add_argument("--max", type=int, default=6)
    args = p.parse_args()
    guard_not_test(args.pred)

    rows = read_jsonl(args.pred)
    categories = sorted({r["category"] for r in rows})
    if args.category not in categories:
        raise ValueError(f"Unknown category {args.category!r}. Available: {categories}")
    present = [r for r in rows if r["category"] == args.category and r["answers"]]
    errors = select_errors(rows, args.category)
    print(f"category: {args.category} | present clauses: {len(present)} | wrong spans: {len(errors)} | showing {min(args.max, len(errors))} most confident")
    for t, r in errors[: args.max]:
        print(f"\n[{t}] score={r['present_score']:.2f} contract={shorten(r['title'], 70)}")
        print(f"  predicted (chars {r['char_start']}-{r['char_end']}): {shorten(r['pred_text'])}")
        for a in r["answers"][:2]:
            print(f"  gold      (chars {a['start']}-{a['start'] + len(a['text'])}): {shorten(a['text'])}")
        if len(r["answers"]) > 2:
            print(f"  ... and {len(r['answers']) - 2} more gold spans")


if __name__ == "__main__":
    main()