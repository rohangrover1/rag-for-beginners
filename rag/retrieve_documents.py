from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_cohere import CohereRerank
from opensearchpy import OpenSearch
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv, find_dotenv
from typing import Any, List, Dict
import logging
import time
import json

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


# --------------------------------------------------
# Pydantic model for structured LLM output
# --------------------------------------------------

class QueryVariations(BaseModel):
    queries: List[str]


# ============================================================
# Retry Utility (Sync)
# ============================================================

def retry_call(func, retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Retry a callable up to `retries` times with exponential back-off.
    Raises the last exception if all attempts fail.
    """
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            logger.warning("Attempt %d/%d failed: %s", attempt + 1, retries, e)
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= backoff


# ============================================================
# Hybrid Retriever (Sync, LangChain-compatible)
# ============================================================

class HybridRetriever(BaseRetriever):
    """
    LangChain-compatible retriever that combines:
      - Dense vector search (Chroma + OpenAI embeddings)
      - Sparse keyword search (BM25 via OpenSearch)
    Results are merged with Reciprocal Rank Fusion (RRF) and
    optionally re-ranked with Cohere Rerank.

    Optional multi-query reformulation generates semantically
    diverse query variations before retrieval.
    """

    # -- Vector store config --
    persist_directory: str = Field(...)
    embedding_model_name: str = Field(...)

    # -- LLM config (for query reformulation) --
    llm_model_name: str = Field(...)

    # -- Reranker config --
    reranker_model_name: str = Field(...)

    # -- BM25 / OpenSearch config --
    bm25_host: str = Field(...)
    bm25_port: int = Field(default=9200)
    bm25_index_name: str = Field(...)

    # -- Retrieval hyper-parameters --
    k: int = Field(default=5)
    rrf_k: int = Field(default=60)
    rerank_top_k: int = Field(default=5)
    enable_rerank: bool = Field(default=True)
    multi_query_reformulation: bool = Field(default=False)
    num_reformulated_queries: int = Field(default=3)
    min_docs_threshold: int = Field(default=3)  # Minimum acceptable docs after reranking

    # -- Internal objects (excluded from serialisation) --
    vector_retriever: Any = Field(default=None, exclude=True)
    bm25_client: Any = Field(default=None, exclude=True)
    reranker: Any = Field(default=None, exclude=True)

    # --------------------------------------------------------
    # Validators
    # --------------------------------------------------------

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("k must be > 0")
        return v

    @field_validator("num_reformulated_queries")
    @classmethod
    def validate_num_reformulated_queries(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("num_reformulated_queries must be > 0")
        return v

    # --------------------------------------------------------
    # Post-init: build internal clients
    # --------------------------------------------------------

    def model_post_init(self, __context: Any) -> None:
        # Vector retriever
        embeddings = OpenAIEmbeddings(model=self.embedding_model_name)
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings,
        )
        self.vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": self.k}
        )

        # BM25 client
        self.bm25_client = OpenSearch(
            hosts=[{"host": self.bm25_host, "port": self.bm25_port}]
        )

        # Optional Cohere reranker
        if self.enable_rerank:
            self.reranker = CohereRerank(
                model=self.reranker_model_name,
                top_n=self.rerank_top_k,
            )

    # --------------------------------------------------------
    # Main retrieval entry-point (LangChain interface)
    # --------------------------------------------------------

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:

        if not query.strip():
            logger.error("Query cannot be empty.")
            raise ValidationError("Query cannot be empty")

        logger.info("Starting hybrid retrieval | query=%s", query)

        # --- 1. Optionally expand the query into multiple variations ---
        if self.multi_query_reformulation:
            try:
                query_variations = self._create_multiple_reformulated_queries(query)
            except Exception as e:
                logger.warning(
                    "Query reformulation failed (%s); falling back to original query.", e
                )
                query_variations = [query]
        else:
            query_variations = [query]

        # --- 2. Retrieve & fuse across all query variations ---
        # Accumulate RRF scores into a single shared dict so that a document
        # that surfaces for multiple variations gets a higher combined score.
        combined: Dict[str, Dict] = {}

        for i, q in enumerate(query_variations, 1):
            logger.info(
                "Processing query variation %d/%d: %s", i, len(query_variations), q
            )

            vector_docs = retry_call(lambda q=q: self.vector_retriever.invoke(q))
            bm25_docs = retry_call(lambda q=q: self._bm25_search(q))

            # *** UPDATED: filter out LLM-generated chunks before fusing ***
            vector_docs = self._filter_valid_chunks(vector_docs)
            bm25_docs = self._filter_valid_chunks(bm25_docs)

            self._accumulate_rrf_scores(vector_docs, bm25_docs, combined)

        fused_docs = [
            v["doc"]
            for v in sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        ]
        logger.info("RRF combined %d unique documents", len(fused_docs))

        # --- 3. Optional reranking with fallback ---
        if self.enable_rerank:
            # *** UPDATED: fallback to top RRF docs if reranker returns too few ***
            reranked_docs = retry_call(lambda: self._rerank(query, fused_docs))
            if len(reranked_docs) < self.min_docs_threshold:
                logger.warning(
                    "Reranker returned only %d docs (threshold=%d). "
                    "Padding with top RRF-scored docs.",
                    len(reranked_docs),
                    self.min_docs_threshold,
                )
                reranked_ids = {
                    doc.metadata.get("doc_id") or hash(doc.page_content)
                    for doc in reranked_docs
                }
                fallback_docs = [
                    doc for doc in fused_docs
                    if (doc.metadata.get("doc_id") or hash(doc.page_content))
                    not in reranked_ids
                ]
                slots_needed = self.min_docs_threshold - len(reranked_docs)
                fused_docs = reranked_docs + fallback_docs[:slots_needed]
            else:
                fused_docs = reranked_docs

        logger.info(
            "Hybrid retrieval completed, returning %d documents", len(fused_docs)
        )
        return fused_docs

    # --------------------------------------------------------
    # *** UPDATED FUNCTION: Filter out LLM-generated chunks ***
    # --------------------------------------------------------

    def _filter_valid_chunks(self, docs: List[Document]) -> List[Document]:
        """
        Remove documents that were generated by an LLM rather than sourced
        directly from the vector or keyword store.

        A chunk is considered LLM-generated (and therefore invalid) if:
          - It has no metadata at all, OR
          - It is missing both `doc_id` and `chunk_index` (the two fields
            that every indexed chunk must carry), OR
          - Its metadata explicitly flags it as synthetic via
            `is_llm_generated=True`.

        Valid chunks are those that originated from the ingestion pipeline
        and therefore always carry `doc_id`, `chunk_index`, and
        `document_name`.
        """
        valid, dropped = [], []

        for doc in docs:
            meta = doc.metadata or {}

            explicitly_generated = meta.get("is_llm_generated", False)
            missing_origin_fields = not meta.get("doc_id") and meta.get("chunk_index") is None

            if explicitly_generated or missing_origin_fields:
                dropped.append(doc)
            else:
                valid.append(doc)

        if dropped:
            logger.warning(
                "_filter_valid_chunks: dropped %d LLM-generated / untagged doc(s). "
                "Retained %d valid doc(s).",
                len(dropped),
                len(valid),
            )

        return valid

    # --------------------------------------------------------
    # BM25 Search
    # --------------------------------------------------------

    def _bm25_search(self, query: str) -> List[Document]:
        body = {
            "size": self.k,
            "query": {"match": {"text": {"query": query}}},
        }

        try:
            resp = self.bm25_client.search(
                index=self.bm25_index_name,
                body=body,
            )
        except Exception as e:
            logger.error("BM25 search failed | index=%s | error=%s", self.bm25_index_name, e)
            raise OpenSearchRetrievalError(str(e)) from e

        hits = resp.get("hits", {}).get("hits", [])
        docs = []
        for rank, hit in enumerate(hits):
            source = hit["_source"]
            docs.append(
                Document(
                    page_content=source.get("text", ""),
                    metadata={
                        "doc_id":        source.get("doc_id"),
                        "document_name": source.get("document_name"),
                        "chunk_index":   source.get("chunk_index"),
                        "raw_text":      source.get("raw_text"),
                        "tables_html":   source.get("tables_html"),
                        "images_base64": source.get("images_base64"),
                        "bm25_score":    float(hit.get("_score", 0)),
                        "bm25_rank":     rank + 1,
                    },
                )
            )
        return docs


    # --------------------------------------------------------
    # Reciprocal Rank Fusion (RRF)
    # --------------------------------------------------------

    def _accumulate_rrf_scores(
        self,
        vector_docs: List[Document],
        bm25_docs: List[Document],
        combined: Dict[str, Dict],
    ) -> None:
        """
        Add RRF scores for one (vector_docs, bm25_docs) pair into `combined`.
        Mutates `combined` in place so scores accumulate across query variations.
        """

        def _add(docs: List[Document]) -> None:
            for rank, doc in enumerate(docs):
                doc_id = doc.metadata.get("doc_id") or doc.id or hash(doc.page_content)
                if not doc_id:
                    logger.warning("Document missing doc_id, falling back to content hash")
                score = 1.0 / (self.rrf_k + rank + 1)
                if doc_id not in combined:
                    combined[doc_id] = {"doc": doc, "score": 0.0}
                combined[doc_id]["score"] += score

        _add(vector_docs)
        _add(bm25_docs)

    # --------------------------------------------------------
    # *** UPDATED FUNCTION: Reranking ***
    # --------------------------------------------------------

    def _rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank `documents` with Cohere and return the result.

        The reranker is given only documents that passed `_filter_valid_chunks`,
        so LLM-generated chunks are already absent. If `compress_documents`
        returns an empty list (e.g. all scores below Cohere's internal threshold),
        the caller (`_get_relevant_documents`) will detect the shortfall via
        `min_docs_threshold` and pad with the top RRF-scored docs.
        """
        if not documents:
            return []

        logger.info("Starting reranking of %d documents", len(documents))
        try:
            reranked = self.reranker.compress_documents(documents, query)
        except Exception as e:
            logger.error("Reranking failed | model=%s | error=%s", self.reranker_model_name, e)
            raise RerankError(str(e)) from e

        logger.info("Reranking completed, returning %d documents", len(reranked))
        return reranked

    # --------------------------------------------------------
    # Query Reformulation
    # --------------------------------------------------------

    def _create_multiple_reformulated_queries(self, query: str) -> List[str]:
        """
        Use an LLM to generate semantically diverse reformulations of `query`.
        Returns a list that always includes the original query.
        """
        logger.info(
            "Generating %d reformulated queries for: %s",
            self.num_reformulated_queries,
            query,
        )

        llm = ChatOpenAI(model=self.llm_model_name, temperature=0)
        llm_with_structured_output = llm.with_structured_output(QueryVariations)

        prompt = (
            f"Generate {self.num_reformulated_queries} different variations of the "
            f"following query that would help retrieve relevant documents.\n"
            f"Original query: {query}\n"
            f"Return {self.num_reformulated_queries} alternative queries that rephrase "
            f"or approach the same question from different angles."
        )

        try:
            response: QueryVariations = retry_call(
                lambda: llm_with_structured_output.invoke(prompt)
            )
        except Exception as e:
            logger.error("Query reformulation failed | model=%s | error=%s", self.llm_model_name, e)
            raise LLMQueryGenerationError(str(e)) from e

        variations = response.queries
        logger.info("Generated %d query variations", len(variations))
        for i, v in enumerate(variations, 1):
            logger.info("  Variation %d: %s", i, v)

        # Always include the original so we never lose its signal
        return [query] + variations