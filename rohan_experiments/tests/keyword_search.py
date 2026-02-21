from langchain_community.retrievers import BM25Retriever
import os
import json
import pickle
import hashlib
from typing import List, Dict, Set

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from pydantic import Field
from nltk.tokenize import word_tokenize
import nltk
from langchain_core.documents import Document


nltk.download("punkt")

# Deterministic Hash Utilities
def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def hash_document(doc: Document) -> str:
    payload = {
        "page_content": doc.page_content,
        "metadata": doc.metadata or {}
    }
    return sha256(canonical_json(payload))

# Chunk Hash Map
def build_chunk_map(docs: List[Document]) -> Dict[str, str]:
    chunk_map = {}
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id")
        if not chunk_id:
            raise ValueError("Document missing required metadata: chunk_id")
        chunk_map[chunk_id] = hash_document(doc)
    return chunk_map

# Diff Detection
def diff_chunk_maps(
    old: Dict[str, str],
    new: Dict[str, str]
) -> tuple[Set[str], Set[str], Set[str]]:
    added = new.keys() - old.keys()
    removed = old.keys() - new.keys()
    modified = {k for k in new.keys() & old.keys() if new[k] != old[k]}
    return added, removed, modified


# The Retriever Subclass
class HashAwareBM25Retriever(BaseRetriever):
    persist_dir: str = Field(...)
    k: int = Field(default=5)

    _bm25: BM25Retriever | None = None
    _chunk_map: Dict[str, str] = {}

    # -------------------------
    # Initialization
    # -------------------------
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.makedirs(self.persist_dir, exist_ok=True)
        self._load_if_exists()

    # -------------------------
    # LangChain Required Method
    # -------------------------
    def _get_relevant_documents(self, query: str) -> List[Document]:
        if not self._bm25:
            raise ValueError("BM25 retriever not initialized")
        return self._bm25.get_relevant_documents(query)

    # -------------------------
    # Tokenization
    # -------------------------
    @staticmethod
    def preprocess(text: str) -> List[str]:
        return word_tokenize(text.lower())

    def preprocess_hash(self) -> str:
        config = {
            "tokenizer": "nltk.word_tokenize",
            "lowercase": True
        }
        return sha256(canonical_json(config))

    # -------------------------
    # Public API
    # -------------------------
    def add_documents(self, docs: List[Document]):
        """
        Idempotent document ingestion with rebuild detection.
        """
        new_chunk_map = build_chunk_map(docs)

        old_chunk_map = self._chunk_map
        added, removed, modified = diff_chunk_maps(old_chunk_map, new_chunk_map)

        change_ratio = (
            len(added) + len(removed) + len(modified)
        ) / max(1, len(old_chunk_map))

        preprocess_changed = (
            self._load_preprocess_hash() != self.preprocess_hash()
        )

        if not preprocess_changed and change_ratio < 0.02:
            # Skip rebuild
            return

        # Rebuild BM25
        self._bm25 = BM25Retriever.from_documents(
            docs,
            preprocess_func=self.preprocess
        )
        self._bm25.k = self.k

        # Persist
        self._chunk_map = new_chunk_map
        self._persist_all()

    # -------------------------
    # Persistence
    # -------------------------
    def _persist_all(self):
        with open(self._bm25_path(), "wb") as f:
            pickle.dump(self._bm25, f)

        with open(self._chunk_map_path(), "w") as f:
            json.dump(self._chunk_map, f, sort_keys=True)

        with open(self._corpus_hash_path(), "w") as f:
            f.write(sha256("".join(sorted(self._chunk_map.values()))))

        with open(self._preprocess_hash_path(), "w") as f:
            f.write(self.preprocess_hash())

    def _load_if_exists(self):
        if os.path.exists(self._bm25_path()):
            with open(self._bm25_path(), "rb") as f:
                self._bm25 = pickle.load(f)

        if os.path.exists(self._chunk_map_path()):
            with open(self._chunk_map_path(), "r") as f:
                self._chunk_map = json.load(f)

    # -------------------------
    # Paths
    # -------------------------
    def _bm25_path(self) -> str:
        return os.path.join(self.persist_dir, "bm25.pkl")

    def _chunk_map_path(self) -> str:
        return os.path.join(self.persist_dir, "chunk_map.json")

    def _corpus_hash_path(self) -> str:
        return os.path.join(self.persist_dir, "corpus.sha256")

    def _preprocess_hash_path(self) -> str:
        return os.path.join(self.persist_dir, "preprocess.sha256")

    def _load_preprocess_hash(self) -> str | None:
        path = self._preprocess_hash_path()
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return f.read().strip()

def make_doc(text, source, idx):
    return Document(
        page_content=text,
        metadata={
            "source_id": source,
            "chunk_index": idx,
            "chunk_id": f"{source}::chunk::{idx}"
        }
    )


if __name__ == "__main__":
    try:
        docs = [
            make_doc("BM25 is a ranking algorithm", "doc1", 0),
            make_doc("RAG combines retrieval and generation", "doc1", 1)
        ]

        retriever = HashAwareBM25Retriever(
            persist_dir="./bm25_store",
            k=3
        )

        retriever.add_documents(docs)

        results = retriever.invoke("ranking algorithm")
        for d in results:
            print(d.page_content)
    except Exception as e:
        print(f"❌ An error occurred: {e}")

