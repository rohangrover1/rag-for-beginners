from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from typing import Any, List
from langchain_core.documents import Document
from opensearchpy import OpenSearch
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain_classic.retrievers import EnsembleRetriever
from pydantic import Field, BaseModel, field_validator
from langchain_cohere import CohereRerank
import logging
import time

_ = load_dotenv(find_dotenv())

# ------------------------------------------------------------------ #
#  Module-level logger                                                #
# ------------------------------------------------------------------ #

logger = logging.getLogger("default_logger")

# --------------------------------------------------
# Custom Exceptions
# --------------------------------------------------

class RetrievalError(Exception):
    pass


class OpenSearchRetrievalError(RetrievalError):
    pass


class LLMQueryGenerationError(RetrievalError):
    pass


class RerankError(RetrievalError):
    pass


class ValidationError(RetrievalError):
    pass


class BM25Retriever(BaseRetriever):
    """
    LangChain-compatible retriever BM25 keyword search using OpenSearch.
    """
    
    host: str = Field(...)                  # ... makes this a required field
    port: int = Field(default=9200)
    index_name: str = Field(...)
    k: int = Field(default=5)
    llm_model: str = Field(default="gpt-4o")
    bm25client: Any = Field(default=None, exclude=True)

    @field_validator("k")
    @classmethod
    def validate_k(cls, v):
        if v <= 0:
            raise ValueError("k must be > 0")
        return v

    def __init__(self, **data):
        '''Initialize the OpenSearch client'''
        super().__init__(**data)
        object.__setattr__(
            self, 'bm25client',
            OpenSearch(hosts=[{"host": self.host, "port": self.port}])
        )


    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> List[Document]:
        '''Retrieve relevant documents using BM25 keyword search'''

        if not query or not query.strip():
            logger.error("Query cannot be empty.")
            raise ValidationError("Query cannot be empty")

        logger.info("Executing BM25 search | index=%s | k=%s", self.index_name, self.k)


        try:
            body = {"size": self.k, "query": {"match": {"text": {"query": query}}}}
            resp = self.bm25client.search(index=self.index_name, body=body)
        except Exception as e:
            logger.exception("OpenSearch BM25 query failed")
            raise OpenSearchRetrievalError(str(e))
        
        hits = resp.get("hits", {}).get("hits", [])
        logger.info("BM25 returned %d documents", len(hits))
        
        docs = []
        for hit in resp["hits"]["hits"]:
            source = hit["_source"]
            docs.append(
                Document(
                    #page_content="",  # can fetch from vector DB
                    page_content=source.get("text", ""),  
                    metadata={
                        "doc_id": source.get("doc_id"),
                        "bm25_score": float(hit["_score"]),
                        "document_name": source.get("document_name"),
                        "raw_text": source.get("raw_text"),
                        "tables_html": source.get("tables_html"),
                        "images_base64": source.get("images_base64"),
                    },
                )
            )
        return docs

# --------------------------------------------------
# Retry Utility
# --------------------------------------------------
def retry_call(func, retries=3, delay=1, backoff=2):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            logger.warning("Attempt %d failed: %s", attempt + 1, e)
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= backoff



def create_vector_retriever(query: str, persistent_directory, embedding_model_name, k: int = 3) -> List[Document]:
    """
    Retrieve relevant documents using vector similarity search.
    """
    embedding_model = OpenAIEmbeddings(model=embedding_model_name)
    vectorstore = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return vector_retriever

# --------------------------------------------------
# Query Reformulation
# --------------------------------------------------

# Pydantic model for structured output
class QueryVariations(BaseModel):
    queries: List[str]

def create_multiple_reformulated_queries(query: str, num_reformulated_queries: int, llm_model: str) -> List[str]:
    """
    Generate multiple reformulated queries using a simple heuristic or an LLM.
    For simplicity, this example just appends a number to the original query.
    In production, you would use an LLM to generate semantically diverse reformulations.
    """

    if num_reformulated_queries <= 0:
        logger.error("num_reformulated_queries must be > 0")
        raise ValidationError("num_reformulated_queries must be > 0")

    if not query.strip():
        logger.error("Query cannot be empty.")
        raise ValidationError("Query cannot be empty")

    logger.info("Generating %d reformulated queries", num_reformulated_queries)

    llm = ChatOpenAI(model=llm_model, temperature=0)
    llm_with_tools = llm.with_structured_output(QueryVariations)
    prompt = f"""Generate {num_reformulated_queries} different variations of this query that would help retrieve relevant documents:
        Original query: {query} 
        Return {num_reformulated_queries} alternative queries that rephrase or approach the same question from different angles."""

    response = llm_with_tools.invoke(prompt)
    query_variations = response.queries
    print("Original Query:", query)
    print("Generated Query Variations:")
    for i, variation in enumerate(query_variations, 1):
        print(f"{i}. {variation}")
    return query_variations

def rerank_documents(query: str, documents: List[Document], num_reranked: int) -> List[Document]:
    """
    Rerank retrieved documents based on semantic similarity to the query.
    """
    # Initialize Cohere reranker
    print(f"Reranking {len(documents)} documents based on relevance to the query...")
    reranker = CohereRerank(model="rerank-english-v3.0", top_n=num_reranked)
    # Rerank the retrieved documents
    reranked_docs = reranker.compress_documents(documents, query)
    return reranked_docs


def HybridRetriever(query: str, persistent_directory, embedding_model_name, bm25_index_name, k, 
                    num_reformulated_queries, multi_query_reformulation) -> List[Document]:
    """
    Retrieve relevant documents using a hybrid approach combining BM25 and vector similarity.
    """
    # Step 1: Get BM25 results
    bm25_retriever = BM25Retriever(host="localhost", port=9200, index_name=bm25_index_name, k=k)
    
    # Step 2: Get vector similarity results
    vector_retriever = create_vector_retriever(query, persistent_directory, embedding_model_name, k)

    # Step 3: Create the Hybrid Retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]  # Weight to vector and keyword search
    )

    if multi_query_reformulation:
        # Optional: Implement multi-query reformulation logic here to generate additional queries
        # For simplicity, we will just use the original query in this example
        query_variations = create_multiple_reformulated_queries(query, num_reformulated_queries)
        docs = []
        for q in query_variations:
            docs.extend(hybrid_retriever.invoke(q))
        # Rerank the combined results from all query variations
        docs = rerank_documents(query, docs, num_reranked=k)

    else:
        # If not using multi-query reformulation, we can directly combine results from both retrievers
        docs = hybrid_retriever.invoke(query)
    

    
    return docs