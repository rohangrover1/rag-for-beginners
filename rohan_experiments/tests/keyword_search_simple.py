import os
import pickle
from typing import List
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt")

class PersistentBM25Retriever:
    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        self.index_path = os.path.join(persist_dir, "bm25.pkl")
        self.docs_path = os.path.join(persist_dir, "documents.pkl")

        self.documents: List[str] = []
        self.tokenized_docs: List[List[str]] = []
        self.bm25: BM25Okapi | None = None

        os.makedirs(persist_dir, exist_ok=True)

        if os.path.exists(self.index_path):
            self._load()

    # ------------------------
    # Internal helpers
    # ------------------------
    def _tokenize(self, text: str) -> List[str]:
        return word_tokenize(text.lower())

    def _build_index(self):
        self.bm25 = BM25Okapi(self.tokenized_docs)

    # ------------------------
    # Persistence
    # ------------------------
    def _save(self):
        with open(self.index_path, "wb") as f:
            pickle.dump(self.bm25, f)

        with open(self.docs_path, "wb") as f:
            pickle.dump(self.documents, f)

    def _load(self):
        with open(self.index_path, "rb") as f:
            self.bm25 = pickle.load(f)

        with open(self.docs_path, "rb") as f:
            self.documents = pickle.load(f)

        self.tokenized_docs = [self._tokenize(doc) for doc in self.documents]

    # ------------------------
    # Public API
    # ------------------------
    def add_documents(self, new_docs: List[str]):
        """
        Adds new documents and rebuilds BM25 index.
        """
        for doc in new_docs:
            self.documents.append(doc)
            self.tokenized_docs.append(self._tokenize(doc))

        self._build_index()
        self._save()

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        if not self.bm25:
            raise ValueError("BM25 index not initialized")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        return [self.documents[i] for i in top_indices]
    

if __name__ == "__main__":
    persist_dir = "./bm25_store_test"

    retriever = PersistentBM25Retriever(persist_dir)

    # Initial documents
    retriever.add_documents([
        "BM25 is a ranking function used by search engines.",
        "Retrieval Augmented Generation combines search with LLMs.",
        "Vector databases use embeddings for semantic search."
    ])

    # Query
    results = retriever.retrieve("search ranking algorithm", k=2)
    print(results)