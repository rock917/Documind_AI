import pytest

from documind.retrieval.bm25 import BM25, tokenize


def test_tokenize_lowercases_drops_punctuation_and_stopwords():
    assert tokenize("The Governing LAW of this Agreement, Delaware.") == ["governing", "law", "agreement", "delaware"]


def test_rare_words_get_higher_idf_than_common_words():
    corpus = [tokenize(t) for t in ["payment due", "payment terms", "payment schedule", "arbitration clause"]]
    bm = BM25().fit(corpus)
    assert bm.idf["arbitration"] > bm.idf["payment"]
    assert bm.default_idf > bm.idf["arbitration"]


def test_documents_with_query_words_outrank_documents_without():
    docs = [tokenize(t) for t in ["payment is due monthly", "governing law is delaware", "notice by email"]]
    bm = BM25().fit(docs)
    ranked = bm.rank(tokenize("which law governs delaware"), docs)
    assert ranked[0][0] == 1
    assert ranked[0][1] > ranked[1][1] >= ranked[2][1]


def test_shorter_document_wins_with_equal_term_frequency():
    short = tokenize("delaware law")
    long = tokenize("delaware law " + "filler words about unrelated topics " * 10)
    bm = BM25().fit([short, long, tokenize("other text entirely")])
    ranked = bm.rank(tokenize("delaware law"), [short, long])
    assert ranked[0][0] == 0 and ranked[0][1] > ranked[1][1]


def test_ties_keep_document_order_and_unseen_words_score_zero_when_absent():
    docs = [tokenize("alpha beta"), tokenize("alpha beta")]
    bm = BM25().fit(docs)
    assert [i for i, _ in bm.rank(tokenize("alpha"), docs)] == [0, 1]
    assert bm.rank(tokenize("zzzunseen"), docs)[0][1] == 0.0


def test_fit_on_empty_corpus_fails_clearly():
    with pytest.raises(ValueError):
        BM25().fit([])