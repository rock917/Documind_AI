"""BM25 ranking written from scratch, with corpus-level IDF.

fit() learns word rarity (IDF) and the average chunk length from a corpus of tokenized chunks, for example the
chunks of all training contracts. rank() then scores the chunks of ANY single contract against a query using that IDF.
"""
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be been by for from has have in is it its of on or that the this to was were will with "
    "shall may any such if not no but which who whom than then".split()
)


def tokenize(text):
    """Lowercase, split into alphanumeric tokens, drop stopwords."""
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.idf = {}
        self.default_idf = 0.0
        self.avgdl = 1.0

    def fit(self, tokenized_docs):
        n = 0
        total_len = 0
        df = Counter()
        for doc in tokenized_docs:
            n += 1
            total_len += len(doc)
            df.update(set(doc))
        if n == 0:
            raise ValueError("Cannot fit BM25 on an empty corpus")
        self.avgdl = max(total_len / n, 1.0)
        self.idf = {t: math.log(1.0 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.default_idf = math.log(1.0 + (n + 0.5) / 0.5)  # a word never seen in the corpus
        return self

    def score(self, query_tokens, doc_counts, doc_len):
        """BM25 score of one document, given its term counts and length."""
        total = 0.0
        norm = self.k1 * (1.0 - self.b + self.b * doc_len / self.avgdl)
        for t in set(query_tokens):
            tf = doc_counts.get(t, 0)
            if tf:
                total += self.idf.get(t, self.default_idf) * tf * (self.k1 + 1.0) / (tf + norm)
        return total

    def rank(self, query_tokens, tokenized_docs):
        """Indices of the documents, best first, with their scores: [(index, score), ...]."""
        scored = []
        for i, doc in enumerate(tokenized_docs):
            scored.append((i, self.score(query_tokens, Counter(doc), len(doc))))
        scored.sort(key=lambda item: -item[1])  # stable: ties keep document order
        return scored