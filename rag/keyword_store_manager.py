import json
import logging
from typing import Dict, Iterable, List, Optional

from opensearchpy import OpenSearch, helpers, ConnectionError, TransportError
from langchain_core.documents import Document
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

class KeywordStoreManager:
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

    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = 9200
    DEFAULT_MAX_RETRIES = 3

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

    def __init__(
        self,
        index_name: str,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """
        Args:
            index_name:   OpenSearch index to read from and write to.
            host:         OpenSearch hostname.
            port:         OpenSearch port.
            max_retries:  Retry attempts for transient OpenSearch errors.

        Raises:
            ValueError: If any argument fails validation.
        """
        self._validate_init_args(index_name, host, port, max_retries)

        self.index_name = index_name
        self.host = host
        self.port = port
        self.max_retries = max_retries

        logger.debug(
            "KeywordStoreManager initialised | index=%s host=%s:%d max_retries=%d",
            index_name, host, port, max_retries,
        )

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

        client = self._get_client()
        self._ensure_index(client)

        if self._document_exists(client, document_name):
            logger.info(
                "Document '%s' already indexed — skipping ingest", document_name
            )
            return True

        result = self._reindex_document(client, document_name, chunks)

        logger.info(
            "Keyword store update complete | document=%s deleted=%d indexed=%d",
            document_name,
            result["deleted_chunks"],
            result["indexed_chunks"],
        )
        return True

    # ------------------------------------------------------------------ #
    #  Client                                                             #
    # ------------------------------------------------------------------ #

    def _get_client(self) -> OpenSearch:
        """Create and return an OpenSearch client.

        Raises:
            KeywordStoreConnectionError: If the client cannot be created.
        """
        try:
            client = OpenSearch(
                hosts=[{"host": self.host, "port": self.port}],
                http_compress=True,
            )
            # Ping to verify the connection is live
            if not client.ping():
                logger.error(f"OpenSearch at {self.host}:{self.port} did not respond to ping.")    
                raise KeywordStoreConnectionError(
                    f"OpenSearch at {self.host}:{self.port} did not respond to ping."
                )
        except KeywordStoreConnectionError:
            logger.error(f"OpenSearch at {self.host}:{self.port} is unreachable.")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to OpenSearch at {self.host}:{self.port}: {e}")   
            raise KeywordStoreConnectionError(
                f"Failed to connect to OpenSearch at {self.host}:{self.port}: {e}"
            ) from e

        logger.debug("OpenSearch client connected | %s:%d", self.host, self.port)
        return client

    # ------------------------------------------------------------------ #
    #  Index management                                                   #
    # ------------------------------------------------------------------ #

    def _ensure_index(self, client: OpenSearch) -> None:
        """Create the index with correct mappings if it does not exist.

        Raises:
            KeywordStoreIndexError: If index creation fails.
        """
        try:
            if client.indices.exists(index=self.index_name):
                logger.debug("Index '%s' already exists", self.index_name)
                return

            client.indices.create(
                index=self.index_name,
                body=self.INDEX_MAPPINGS,
            )
            logger.info("Created index '%s'", self.index_name)

        except Exception as e:
            logger.error(f"Failed to create index '{self.index_name}': {e}")
            raise KeywordStoreIndexError(
                f"Failed to create index '{self.index_name}': {e}"
            ) from e

    # ------------------------------------------------------------------ #
    #  Reindex lifecycle                                                  #
    # ------------------------------------------------------------------ #

    def _reindex_document(
        self,
        client: OpenSearch,
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
                client, document_name, refresh=True
            )
            logger.info(
                "Deleted %d existing chunks for document '%s'",
                deleted, document_name,
            )
        except KeywordStoreDeleteError as e:
            logger.error(f"Reindex aborted during delete phase for '{document_name}': {e}")
            raise KeywordStoreReindexError(
                f"Reindex aborted during delete phase for '{document_name}': {e}"
            ) from e

        try:
            indexed = self._bulk_upsert_documents(client, chunks, refresh=True)
            logger.info(
                "Upserted %d chunks for document '%s'", indexed, document_name
            )
        except KeywordStoreIngestError as e:
            logger.error(f"Reindex failed during upsert phase for '{document_name}': {e}")
            logger.error( f"Document was deleted but not re-indexed: {e}")
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
        client: OpenSearch,
        documents: Iterable[Document],
        refresh: bool = False,
    ) -> int:
        """Bulk upsert LangChain Documents using deterministic Document.id.

        Retries on transient OpenSearch errors.

        Returns:
            Number of successfully indexed documents.

        Raises:
            DocumentIDError:      If any document is missing an ID.
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
                            "ai_enhanced":  doc.metadata["ai_enhanced"],
                        },
                        "doc_as_upsert": True,
                    }

            success, failed = helpers.bulk(
                client,
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
            logger.error(f"Bulk upsert into '{self.index_name}' failed after {self.max_retries} attempts: {e}")
            raise KeywordStoreIngestError(
                f"Bulk upsert into '{self.index_name}' failed after "
                f"{self.max_retries} attempts: {e}"
            ) from e

    def _delete_document_chunks(
        self,
        client: OpenSearch,
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
        @retry(
            retry=retry_if_exception_type((ConnectionError, TransportError)),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=False,
        )
        def _call() -> int:
            response = client.delete_by_query(
                index=self.index_name,
                body={"query": {"term": {"document_name": document_name}}},
                conflicts="proceed",
                refresh=refresh,
            )
            return response["deleted"]

        try:
            return _call()
        except Exception as e:
            logger.error(f"delete_by_query for document '{document_name}' in index '{self.index_name}' failed after {self.max_retries} attempts: {e}")
            raise KeywordStoreDeleteError(
                f"delete_by_query for document '{document_name}' in index "
                f"'{self.index_name}' failed after {self.max_retries} attempts: {e}"
            ) from e

    # ------------------------------------------------------------------ #
    #  Read operations                                                    #
    # ------------------------------------------------------------------ #

    def _document_exists(self, client: OpenSearch, document_name: str) -> bool:
        """Return True if any chunk for the document is already indexed.

        Raises:
            KeywordStoreQueryError: If the search query fails.
        """
        try:
            response = client.search(
                index=self.index_name,
                body={
                    "query": {"term": {"document_name": document_name}},
                    "size": 1,
                },
            )
            exists = response["hits"]["total"]["value"] > 0
        except Exception as e:
            logger.error(f"Existence check for document '{document_name}' failed: {e}")
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
            KeywordStoreConnectionError: If OpenSearch is unreachable.
            KeywordStoreQueryError:      If the count query fails.
        """
        client = self._get_client()

        try:
            if not client.indices.exists(index=self.index_name):
                logger.warning("Index '%s' does not exist", self.index_name)
                return 0
            response = client.count(index=self.index_name)
        except Exception as e:
            logger.error(f"Count query on index '{self.index_name}' failed: {e}")
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
            KeywordStoreConnectionError: If OpenSearch is unreachable.
            KeywordStoreQueryError:      If the scroll query fails.
        """
        client = self._get_client()

        try:
            response = client.search(
                index=self.index_name,
                body={"query": {"match_all": {}}, "_source": ["doc_id"]},
                scroll="2m",
                size=1000,
            )
            ids = [hit["_source"]["doc_id"] for hit in response["hits"]["hits"]]
            scroll_id = response.get("_scroll_id")

            while scroll_id:
                page = client.scroll(scroll_id=scroll_id, scroll="2m")
                hits = page["hits"]["hits"]
                if not hits:
                    break
                ids.extend(hit["_source"]["doc_id"] for hit in hits)
                scroll_id = page.get("_scroll_id")

        except Exception as e:
            logger.error(f"Failed to retrieve all document IDs from index '{self.index_name}': {e}")
            raise KeywordStoreQueryError(
                f"Failed to retrieve all document IDs from '{self.index_name}': {e}"
            ) from e

        logger.info("Retrieved %d document IDs from index '%s'", len(ids), self.index_name)
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
                logger.error(f"Document at index {i} is missing 'id'. doc.id is required for OpenSearch upserts.")
                raise DocumentIDError(
                    f"Document at index {i} is missing 'id'. "
                    "doc.id is required for OpenSearch upserts."
                )

    # ------------------------------------------------------------------ #
    #  Validation                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_init_args(
        index_name: str,
        host: str,
        port: int,
        max_retries: int,
    ) -> None:
        """Validate constructor arguments.

        Raises:
            ValueError: On any invalid argument.
        """
        if not index_name or not index_name.strip():
            logger.error("index_name must be a non-empty string.")
            raise ValueError("index_name must be a non-empty string.")

        if not host or not host.strip():
            logger.error("host must be a non-empty string.")   
            raise ValueError("host must be a non-empty string.")

        if not isinstance(port, int) or not (1 <= port <= 65535):
            logger.error(f"port must be an integer between 1 and 65535, got {port}.")
            raise ValueError(
                f"port must be an integer between 1 and 65535, got {port}."
            )

        if max_retries < 1:
            logger.error(f"max_retries must be >= 1, got {max_retries}.")
            raise ValueError(
                f"max_retries must be >= 1, got {max_retries}."
            )

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
            logger.error(f"chunks must be a list, got {type(chunks).__name__}.")
            raise ValueError(
                f"chunks must be a list, got {type(chunks).__name__}."
            )

        if len(chunks) == 0:
            logger.error("chunks must not be empty.")  
            raise ValueError("chunks must not be empty.")