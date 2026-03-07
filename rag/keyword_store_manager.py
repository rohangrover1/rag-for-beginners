import logging
from typing import Any, Dict, Iterable, List
from opensearchpy import OpenSearch, helpers, ConnectionError, TransportError
from langchain_core.documents import Document
from pydantic import BaseModel, Field, field_validator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

# ------------------------------------------------------------------ #
#  Module-level logger                                                #
# ------------------------------------------------------------------ #
logger = logging.getLogger("default_logger")  # Use the same logger configured in main.py


# ------------------------------------------------------------------ #
#  Index Mapping                                                   #
# ------------------------------------------------------------------ #
# OpenSearch index mappings — shared across create and ensure operations
INDEX_MAPPINGS = {
    "mappings": {
        "properties": {
            "doc_id":        {"type": "keyword"},
            "document_name": {"type": "keyword"},
            "chunk_index":   {"type": "integer"},
            "text":          {"type": "text"},       # BM25 field
            "raw_text":      {"type": "text"},
            "tables_html":   {"type": "text"},
            "images_base64": {
                "type":  "text",
                "index": False                      # never BM25-indexed
            },
            "ai_enhanced":   {"type": "boolean"},
        }
    }
}

# ------------------------------------------------------------------ #
#  Custom exceptions                                                  #
# ------------------------------------------------------------------ #

class KeywordStoreError(Exception):
    """Base exception for all KeywordStoreManager errors."""


class KeywordStoreConnectionError(KeywordStoreError):
    """Raised when a connection to OpenSearch cannot be established."""


class KeywordStoreIndexError(KeywordStoreError):
    """Raised when index creation or validation fails."""


class KeywordStoreIngestError(KeywordStoreError):
    """Raised when bulk upsert or single-document indexing fails."""


class KeywordStoreDeleteError(KeywordStoreError):
    """Raised when a delete-by-query operation fails."""


class KeywordStoreQueryError(KeywordStoreError):
    """Raised when a search or count query fails."""


class KeywordStoreReindexError(KeywordStoreError):
    """Raised when a full delete + re-index cycle fails."""


class DocumentIDError(KeywordStoreError):
    """Raised when a LangChain Document is missing a required ID."""


# ------------------------------------------------------------------ #
#  Main class                                                         #
# ------------------------------------------------------------------ #

class KeywordStoreManager(BaseModel):
    """
    Manages an OpenSearch BM25 keyword store backed by LangChain Documents
    produced by the PDFPartitioner class.

    Entry point: `update_keyword_store(document_name, chunks)`

    Example usage:
        manager = KeywordStoreManager(index_name="my_index")
        manager.update_keyword_store(document_name="annual_report", chunks=docs)
        count = manager.number_of_documents()

    Raises:
        ValueError:                   If constructor or method arguments are invalid.
        KeywordStoreConnectionError:  If OpenSearch is unreachable.
        KeywordStoreIndexError:       If index creation fails.
        KeywordStoreIngestError:      If bulk upsert fails.
        KeywordStoreDeleteError:      If delete-by-query fails.
        KeywordStoreQueryError:       If a search or count query fails.
        KeywordStoreReindexError:     If the reindex lifecycle fails.
        DocumentIDError:              If a document is missing a required ID.
    """

    # ------------------------------------------------------------------ #
    #  Pydantic fields                                                    #
    # ------------------------------------------------------------------ #

    index_name: str = Field(
        ...,
        min_length=1,
        description="OpenSearch index to read from and write to.",
    )
    host: str = Field(
        default="localhost",
        min_length=1,
        description="OpenSearch hostname.",
    )
    port: int = Field(
        default=9200,
        ge=1,
        le=65535,
        description="OpenSearch port (1–65535).",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        description="Retry attempts for transient OpenSearch errors.",
    )
    # Excluded from serialisation; populated by model_post_init.
    bm25client: Any = Field(default=None, exclude=True)

    # ------------------------------------------------------------------ #
    #  Field validators                                                   #
    # ------------------------------------------------------------------ #

    @field_validator("index_name")
    @classmethod
    def _validate_index_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("index_name must not be blank or whitespace.")
        return v

    @field_validator("host")
    @classmethod
    def _validate_host(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("host must not be blank or whitespace.")
        return v

    # ------------------------------------------------------------------ #
    #  Post-init: open the OpenSearch client                              #
    # ------------------------------------------------------------------ #

    def model_post_init(self, __context: Any) -> None:
        """Open and ping-verify the BM25 client after Pydantic validation.

        Raises:
            KeywordStoreConnectionError: If the client cannot connect.
        """
        # BaseModel fields are immutable by default; use object.__setattr__
        # to set the non-Field runtime attribute without triggering Pydantic.
        object.__setattr__(self, "bm25client", self._open_client())
        logger.debug(
            "KeywordStoreManager initialised | index=%s host=%s:%d max_retries=%d",
            self.index_name, self.host, self.port, self.max_retries,
        )

    # ------------------------------------------------------------------ #
    #  Client                                                             #
    # ------------------------------------------------------------------ #

    def _open_client(self) -> OpenSearch:
        """Create, ping-verify, and return a persistent OpenSearch client.

        Called once during __init__ to populate self.bm25client.

        Raises:
            KeywordStoreConnectionError: If the client cannot be created or
                                         does not respond to a ping.
        """
        try:
            client = OpenSearch(
                hosts=[{"host": self.host, "port": self.port}],
                http_compress=True,
            )
            if not client.ping():
                logger.error(
                    "OpenSearch at %s:%d did not respond to ping.",
                    self.host, self.port,
                )
                raise KeywordStoreConnectionError(
                    f"OpenSearch at {self.host}:{self.port} did not respond to ping."
                )
        except KeywordStoreConnectionError:
            raise
        except Exception as e:
            logger.error(
                "Failed to connect to OpenSearch at %s:%d: %s",
                self.host, self.port, e,
            )
            raise KeywordStoreConnectionError(
                f"Failed to connect to OpenSearch at {self.host}:{self.port}: {e}"
            ) from e

        logger.debug("OpenSearch client connected | %s:%d", self.host, self.port)
        return client

    # ------------------------------------------------------------------ #
    #  Entry point                                                         #
    # ------------------------------------------------------------------ #

    def update_keyword_store(
        self, document_name: str, chunks: List[Document]
    ) -> bool:
        """
        Ingest documents into OpenSearch, skipping if already indexed.

        Creates the index if it does not exist, checks for an existing
        ingest of the document, then runs a delete + bulk upsert lifecycle.

        Args:
            document_name: Logical document identifier.
            chunks:        LangChain Documents from PDFPartitioner.

        Returns:
            True on success.

        Raises:
            ValueError:                  If arguments are invalid.
            DocumentIDError:             If any chunk is missing an ID.
            KeywordStoreConnectionError: If OpenSearch is unreachable.
            KeywordStoreIndexError:      If index creation fails.
            KeywordStoreQueryError:      If the existence check query fails.
            KeywordStoreReindexError:    If the reindex lifecycle fails.
        """
        self._validate_document_name(document_name)
        self._validate_chunks(chunks)

        logger.info(
            "Starting keyword store update | document=%s chunks=%d",
            document_name, len(chunks),
        )

        self._ensure_index()

        if self._document_exists(document_name):
            logger.info(
                "Document '%s' already indexed — skipping ingest", document_name
            )
            return True

        result = self._reindex_document(document_name, chunks)

        logger.info(
            "Keyword store update complete | document=%s deleted=%d indexed=%d",
            document_name,
            result["deleted_chunks"],
            result["indexed_chunks"],
        )
        return True

    # ------------------------------------------------------------------ #
    #  Index management                                                   #
    # ------------------------------------------------------------------ #

    def _ensure_index(self) -> None:
        """Create the index with correct mappings if it does not exist.

        Raises:
            KeywordStoreIndexError: If index creation fails.
        """
        try:
            if self.bm25client.indices.exists(index=self.index_name):
                logger.debug("Index '%s' already exists", self.index_name)
                return

            self.bm25client.indices.create(
                index=self.index_name,
                body=self.INDEX_MAPPINGS,
            )
            logger.info("Created index '%s'", self.index_name)

        except Exception as e:
            logger.error("Failed to create index '%s': %s", self.index_name, e)
            raise KeywordStoreIndexError(
                f"Failed to create index '{self.index_name}': {e}"
            ) from e

    # ------------------------------------------------------------------ #
    #  Reindex lifecycle                                                  #
    # ------------------------------------------------------------------ #

    def _reindex_document(
        self,
        document_name: str,
        chunks: List[Document],
    ) -> Dict:
        """Delete all existing chunks for a document then bulk upsert new ones.

        Raises:
            KeywordStoreReindexError: If either the delete or upsert step fails.
        """
        logger.info("Starting reindex lifecycle for document '%s'", document_name)

        try:
            deleted = self._delete_document_chunks(
                document_name, refresh=True
            )
            logger.info(
                "Deleted %d existing chunks for document '%s'",
                deleted, document_name,
            )
        except KeywordStoreDeleteError as e:
            logger.error(
                "Reindex aborted during delete phase for '%s': %s",
                document_name, e,
            )
            raise KeywordStoreReindexError(
                f"Reindex aborted during delete phase for '{document_name}': {e}"
            ) from e

        try:
            indexed = self._bulk_upsert_documents(chunks, refresh=True)
            logger.info(
                "Upserted %d chunks for document '%s'", indexed, document_name
            )
        except KeywordStoreIngestError as e:
            logger.error(
                "Reindex failed during upsert phase for '%s': %s",
                document_name, e,
            )
            logger.error("Document was deleted but not re-indexed: %s", e)
            raise KeywordStoreReindexError(
                f"Reindex failed during upsert phase for '{document_name}'. "
                f"Document was deleted but not re-indexed: {e}"
            ) from e

        return {
            "document_name": document_name,
            "deleted_chunks": deleted,
            "indexed_chunks": indexed,
        }

    # ------------------------------------------------------------------ #
    #  Write operations (with retry)                                      #
    # ------------------------------------------------------------------ #

    def _bulk_upsert_documents(
        self,
        documents: Iterable[Document],
        refresh: bool = False,
    ) -> int:
        """Bulk upsert LangChain Documents using deterministic Document.id.

        Retries on transient OpenSearch errors.

        Returns:
            Number of successfully indexed documents.

        Raises:
            DocumentIDError:         If any document is missing an ID.
            KeywordStoreIngestError: If all retry attempts are exhausted.
        """
        # Materialise so we can validate IDs before hitting the network
        docs = list(documents)
        self._validate_document_ids(docs)

        @retry(
            retry=retry_if_exception_type((ConnectionError, TransportError)),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=False,
        )
        def _call() -> int:
            def actions():
                for doc in docs:
                    yield {
                        "_op_type": "update",
                        "_index": self.index_name,
                        "_id": doc.metadata["doc_id"],  # Use the deterministic doc_id from metadata
                        "doc": {
                            "doc_id":        doc.metadata["doc_id"],
                            "document_name": doc.metadata["document_name"],
                            "chunk_index":   doc.metadata["chunk_index"],
                            "text":          doc.page_content,
                            "raw_text":      doc.metadata["raw_text"],
                            "tables_html":   doc.metadata["tables_html"],
                            "images_base64": doc.metadata["images_base64"],
                            "ai_enhanced":   doc.metadata["ai_enhanced"],
                        },
                        "doc_as_upsert": True,
                    }

            success, failed = helpers.bulk(
                self.bm25client,
                actions(),
                refresh=refresh,
                raise_on_error=False,
            )

            if failed:
                logger.warning(
                    "%d document(s) failed during bulk upsert", len(failed)
                )

            return success

        try:
            return _call()
        except Exception as e:
            logger.error(
                "Bulk upsert into '%s' failed after %d attempts: %s",
                self.index_name, self.max_retries, e,
            )
            raise KeywordStoreIngestError(
                f"Bulk upsert into '{self.index_name}' failed after "
                f"{self.max_retries} attempts: {e}"
            ) from e

    def _delete_document_chunks(
        self,
        document_name: str,
        refresh: bool = True,
    ) -> int:
        """Delete all chunks belonging to a document using delete-by-query.

        Retries on transient OpenSearch errors.

        Returns:
            Number of deleted documents.

        Raises:
            KeywordStoreDeleteError: If all retry attempts are exhausted.
        """
        
        document_exists = self._document_exists(document_name)
        if not document_exists:
            logger.info(
                "No existing chunks found for document '%s' in index '%s'. "
                "Skipping delete-by-query.",
                document_name, self.index_name,
            )
            return 0
        
        num_docs = self.number_of_documents()
        logger.info(
            "Found existing chunks for document '%s' in index '%s with %d documents'. "
            "Proceeding with delete-by-query.",
            document_name, self.index_name, num_docs)

        @retry(
            retry=retry_if_exception_type((ConnectionError, TransportError)),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=False,
        )
        def _call() -> int:
            response = self.bm25client.delete_by_query(
                index=self.index_name,
                body={"query": {"term": {"document_name": document_name}}},
                conflicts="proceed",
                refresh=refresh,
            )
            num_docs = self.number_of_documents()
            logger.info("number of documents in index '%s' after delete_by_query: %d", self.index_name, num_docs)
            return response["deleted"]

        try:
            return _call()
        except Exception as e:
            logger.error(
                "delete_by_query for document '%s' in index '%s' failed after %d attempts: %s",
                document_name, self.index_name, self.max_retries, e,
            )
            raise KeywordStoreDeleteError(
                f"delete_by_query for document '{document_name}' in index "
                f"'{self.index_name}' failed after {self.max_retries} attempts: {e}"
            ) from e

    # ------------------------------------------------------------------ #
    #  Read operations                                                    #
    # ------------------------------------------------------------------ #

    def _document_exists(self, document_name: str) -> bool:
        """Return True if any chunk for the document is already indexed.

        Raises:
            KeywordStoreQueryError: If the search query fails.
        """
        try:
            response = self.bm25client.search(
                index=self.index_name,
                body={
                    "query": {"term": {"document_name": document_name}},
                    "size": 1,
                },
            )
            exists = response["hits"]["total"]["value"] > 0
        except Exception as e:
            logger.error(
                "Existence check for document '%s' failed: %s", document_name, e
            )
            raise KeywordStoreQueryError(
                f"Existence check for document '{document_name}' failed: {e}"
            ) from e

        logger.debug(
            "Document existence check | document_name=%s exists=%s",
            document_name, exists,
        )
        return exists

    def number_of_documents(self) -> int:
        """Return the total number of documents indexed in the store.

        Raises:
            KeywordStoreQueryError: If the count query fails.
        """
        try:
            if not self.bm25client.indices.exists(index=self.index_name):
                logger.warning("Index '%s' does not exist", self.index_name)
                return 0
            response = self.bm25client.count(index=self.index_name)
        except Exception as e:
            logger.error(
                "Count query on index '%s' failed: %s", self.index_name, e
            )
            raise KeywordStoreQueryError(
                f"Count query on index '{self.index_name}' failed: {e}"
            ) from e

        count = response["count"]
        logger.info(
            "Total documents in keyword store index '%s': %d",
            self.index_name, count,
        )
        return count

    def get_all_document_ids(self) -> List[str]:
        """Return all doc_ids stored in the index.

        Raises:
            KeywordStoreQueryError: If the scroll query fails.
        """
        try:
            response = self.bm25client.search(
                index=self.index_name,
                body={"query": {"match_all": {}}, "_source": ["doc_id"]},
                scroll="2m",
                size=1000,
            )
            ids = [hit["_source"]["doc_id"] for hit in response["hits"]["hits"]]
            scroll_id = response.get("_scroll_id")

            while scroll_id:
                page = self.bm25client.scroll(scroll_id=scroll_id, scroll="2m")
                hits = page["hits"]["hits"]
                if not hits:
                    break
                ids.extend(hit["_source"]["doc_id"] for hit in hits)
                scroll_id = page.get("_scroll_id")

        except Exception as e:
            logger.error(
                "Failed to retrieve all document IDs from index '%s': %s",
                self.index_name, e,
            )
            raise KeywordStoreQueryError(
                f"Failed to retrieve all document IDs from '{self.index_name}': {e}"
            ) from e

        logger.info(
            "Retrieved %d document IDs from index '%s'", len(ids), self.index_name
        )
        return ids

    # ------------------------------------------------------------------ #
    #  Helpers (private)                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_document_ids(documents: List[Document]) -> None:
        """Ensure every document has a non-null ID before hitting the network.

        Raises:
            DocumentIDError: If any document is missing an ID.
        """
        for i, doc in enumerate(documents):
            if not hasattr(doc, "id") or doc.id is None:
                logger.error(
                    "Document at index %d is missing 'id'. "
                    "doc.id is required for OpenSearch upserts.",
                    i,
                )
                raise DocumentIDError(
                    f"Document at index {i} is missing 'id'. "
                    "doc.id is required for OpenSearch upserts."
                )

    # ------------------------------------------------------------------ #
    #  Validation                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_document_name(document_name: str) -> None:
        """Validate document_name.

        Raises:
            ValueError: If document_name is not a non-empty string.
        """
        if not isinstance(document_name, str) or not document_name.strip():
            logger.error("document_name must be a non-empty string.")
            raise ValueError("document_name must be a non-empty string.")

    @staticmethod
    def _validate_chunks(chunks: List[Document]) -> None:
        """Validate the chunks list.

        Raises:
            ValueError: If chunks is None, not a list, or empty.
        """
        if chunks is None:
            logger.error("chunks must not be None.")
            raise ValueError("chunks must not be None.")

        if not isinstance(chunks, list):
            logger.error(
                "chunks must be a list, got %s.", type(chunks).__name__
            )
            raise ValueError(
                f"chunks must be a list, got {type(chunks).__name__}."
            )

        if len(chunks) == 0:
            logger.error("chunks must not be empty.")
            raise ValueError("chunks must not be empty.")