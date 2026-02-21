import os
from whoosh.fields import Schema, ID, TEXT, KEYWORD
from whoosh.analysis import StemmingAnalyzer
from whoosh.index import create_in, open_dir, exists_in
from whoosh.writing import AsyncWriter
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from whoosh.qparser import QueryParser
from pydantic import Field
from typing import Any

sample_chunks = [
    {
        "doc_id": "doc-1",
        "text": "Our enterprise pricing plan includes priority support and SLA guarantees.",
        "metadata": {
            "source": "pricing.md",
            "tags": ["pricing", "enterprise"]
        }
    },
    {
        "doc_id": "doc-2",
        "text": "The free tier allows up to 10,000 API calls per month with community support.",
        "metadata": {
            "source": "pricing.md",
            "tags": ["pricing", "free"]
        }
    },
    {
        "doc_id": "doc-3",
        "text": "Authentication is handled using OAuth 2.0 with support for refresh tokens.",
        "metadata": {
            "source": "auth.md",
            "tags": ["auth", "security"]
        }
    },
    {
        "doc_id": "doc-4",
        "text": "Rate limits are enforced per API key and reset every 60 seconds.",
        "metadata": {
            "source": "limits.md",
            "tags": ["limits", "api"]
        }
    },
]

new_chunks = [
    {
        "doc_id": "doc-5",
        "text": "All API responses are returned in JSON format, and endpoints follow REST conventions.",
        "metadata": {
            "source": "api.md",
            "tags": ["api", "format", "rest"]
        }
    },
    {
        "doc_id": "doc-6",
        "text": "Data retention is 90 days by default, with options for enterprise customers to extend retention periods.",
        "metadata": {
            "source": "privacy.md",
            "tags": ["data", "retention", "enterprise"]
        }
    },
    {
        "doc_id": "doc-7",
        "text": "Notifications can be sent via email or webhook, depending on user preferences.",
        "metadata": {
            "source": "notifications.md",
            "tags": ["notifications", "email", "webhook"]
        }
    },
]

def get_schema():
    return Schema(
        doc_id=ID(stored=True, unique=True),
        text=TEXT(analyzer=StemmingAnalyzer()),
        source=ID(stored=True),
        tags=KEYWORD(stored=True, commas=True, lowercase=True),
    )

def get_or_create_index(index_dir: str):
    os.makedirs(index_dir, exist_ok=True)

    if exists_in(index_dir):
        return open_dir(index_dir)

    return create_in(index_dir, get_schema())

def index_chunks(index_dir: str, chunks: list[dict]):
    ix = get_or_create_index(index_dir)
    writer = AsyncWriter(ix)

    for chunk in chunks:
        writer.update_document(
            doc_id=chunk["doc_id"],
            text=chunk["text"],
            source=chunk["metadata"].get("source", ""),
            tags=",".join(chunk["metadata"].get("tags", [])),
        )

    writer.commit()


class WhooshBM25Retriever(BaseRetriever):
    """
    LangChain-compatible BM25 retriever backed by a Whoosh index.
    """
    
    index_dir: str = Field(...)
    k: int = Field(default=5)
    ix: Any = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.ix = open_dir(self.index_dir)

    def chunk_count(self) -> int:
        with self.ix.searcher() as searcher:
            return searcher.doc_count()


    def _get_relevant_documents(self, query: str) -> List[Document]:
        with self.ix.searcher() as searcher:
            parser = QueryParser("text", self.ix.schema)
            q = parser.parse(query)

            results = searcher.search(q, limit=self.k)

            docs = []
            for r in results:
                docs.append(
                    Document(
                        page_content="",  # fetched later from vector DB
                        metadata={
                            "doc_id": r["doc_id"],
                            "bm25_score": float(r.score),
                            "source": r["source"],
                            "tags": r["tags"],
                        },
                    )
                )

            return docs


if __name__ == "__main__":
    INDEX_DIR = "./whoosh_test_index"

    # 1. Build the index
    index_chunks(INDEX_DIR, new_chunks)

    # 2. Create retriever
    retriever = WhooshBM25Retriever(
        index_dir=INDEX_DIR,
        k=3
    )
    print("Chunks indexed:", retriever.chunk_count())


    # # 3. Run test queries
    # queries = [
    #     "enterprise pricing support",
    #     "API rate limits",
    #     "OAuth authentication",
    # ]

    # for q in queries:
    #     print(f"\nQuery: {q}")
    #     results = retriever._get_relevant_documents(q)

    #     for i, doc in enumerate(results, 1):
    #         print(
    #             f"{i}. doc_id={doc.metadata['doc_id']} | "
    #             f"score={doc.metadata['bm25_score']:.3f} | "
    #             f"source={doc.metadata['source']} | "
    #             f"tags={doc.metadata['tags']}"
    #         )