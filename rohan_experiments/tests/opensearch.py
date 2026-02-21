from opensearchpy import OpenSearch
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from typing import Any, List


sample_chunks = [
    {"doc_id": "doc-1", "text": "Enterprise pricing includes SLA and support", "source": "pricing.md", "tags": ["enterprise", "pricing"]},
    {"doc_id": "doc-2", "text": "Free tier allows up to 10,000 API calls per month", "source": "pricing.md", "tags": ["free", "pricing"]},
    {"doc_id": "doc-3", "text": "OAuth 2.0 is used for authentication", "source": "auth.md", "tags": ["auth", "security"]},
    {"doc_id": "doc-4", "text": "API rate limits reset every 60 seconds", "source": "limits.md", "tags": ["limits", "api"]},
]

def index_documents(sample_chunks, index_name: str) -> str:
# opensrach BM225 Index
    try:
        client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])

        if not client.indices.exists(index=index_name):
            client.indices.create(
                index=index_name,
                body={
                    "mappings": {
                        "properties": {
                            "doc_id": {"type": "keyword"},
                            "text": {"type": "text"},
                            "source": {"type": "keyword"},
                            "tags": {"type": "keyword"},
                        }
                    }
                },
            )

        # Index chunks
        for chunk in sample_chunks:
            client.index(index=index_name, id=chunk["doc_id"], body=chunk)
    except Exception as e:
        print(f"Error indexing documents: {e}")
        return None

class OpenSearchBM25Retriever(BaseRetriever):
    host: str = Field(...)
    port: int = Field(default=9200)
    index_name: str = Field(...)
    k: int = Field(default=5)
    client: Any = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.client = OpenSearch(hosts=[{"host": self.host, "port": self.port}])

    def _get_relevant_documents(self, query: str) -> List[Document]:
        body = {"size": self.k, "query": {"match": {"text": {"query": query}}}}
        resp = self.client.search(index=self.index_name, body=body)
        docs = []
        for hit in resp["hits"]["hits"]:
            source = hit["_source"]
            docs.append(
                Document(
                    #page_content="",  # can fetch from vector DB
                    page_content=source.get("text"),  
                    metadata={
                        "doc_id": source.get("doc_id"),
                        "bm25_score": float(hit["_score"]),
                        "source": source.get("source"),
                        "tags": source.get("tags"),
                    },
                )
            )
        return docs

if __name__ == "__main__":
    # Create index if not exists
    index_name = "documents"
    
    # indiex the documents
    # index_documents(sample_chunks, index_name)
    # print(f"Indexed {len(sample_chunks)} documents into OpenSearch index '{index_name}'")

    # test the retriever
    #query = "enterprise support SLA"
    query = "Who is Rohan Grover?"
    bm25_retriever = OpenSearchBM25Retriever(host="localhost", port=9200, index_name=index_name, k=5)
    sparse_results = bm25_retriever._get_relevant_documents(query)
    print(f"Retrieved {len(sparse_results)} documents for query: '{query}'")
    for i, doc in enumerate(sparse_results):
        print(f"Document {i}: Metadata {doc.metadata}")
        print(f"Text: {doc.page_content}")

