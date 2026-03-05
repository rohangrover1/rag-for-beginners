import logging
import os
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
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

logger = logging.getLogger("default_logger")


# ------------------------------------------------------------------ #
#  Custom exceptions                                                  #
# ------------------------------------------------------------------ #

class VectorStoreError(Exception):
    """Base exception for all VectorStore errors."""


class VectorStoreLoadError(VectorStoreError):
    """Raised when an existing vector store cannot be loaded."""


class VectorStoreCreateError(VectorStoreError):
    """Raised when a new vector store cannot be created."""


class VectorStoreUpdateError(VectorStoreError):
    """Raised when adding documents to an existing vector store fails."""


class VectorStoreDeleteError(VectorStoreError):
    """Raised when deleting vectors from the store fails."""


class VectorStoreQueryError(VectorStoreError):
    """Raised when a similarity search or metadata query fails."""


class DocumentIDError(VectorStoreError):
    """Raised when a LangChain Document is missing a required ID."""


class EmbeddingError(VectorStoreError):
    """Raised when the embedding API call fails after all retries."""


# ------------------------------------------------------------------ #
#  Main class                                                         #
# ------------------------------------------------------------------ #

class VectorStoreManager:
    """
    Manages a persistent ChromaDB vector store backed by OpenAI embeddings.

    Handles creation, update, deduplication, and deletion of vectors
    produced by the PDFPartitioner class.

    Entry point: `update_vector_store(documents, document_name)`

    Example usage:
        manager = VectorStoreManager(persist_directory="./chroma_db")
        manager.update_vector_store(documents, document_name="annual_report")
        count = manager.number_of_documents()

    Raises:
        ValueError:            If constructor or method arguments are invalid.
        VectorStoreLoadError:  If an existing store cannot be loaded.
        VectorStoreCreateError: If a new store cannot be created.
        VectorStoreUpdateError: If adding documents fails.
        VectorStoreDeleteError: If a bulk delete operation fails.
        VectorStoreQueryError:  If a similarity search fails.
        DocumentIDError:        If a document is missing a required ID.
        EmbeddingError:         If the embedding API fails after all retries.
    """

    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
    DEFAULT_SIMILARITY_THRESHOLD = 0.98
    DEFAULT_DUPLICATE_CHECK_K = 3
    DEFAULT_DELETE_BATCH_SIZE = 1000

    def __init__(
        self,
        persist_directory: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        duplicate_check_k: int = DEFAULT_DUPLICATE_CHECK_K,
        embedding_max_retries: int = 3,
    ):
        """
        Args:
            persist_directory:    Path where ChromaDB persists its data.
            embedding_model:      OpenAI embedding model name.
            similarity_threshold: Cosine similarity above which a chunk is
                                  considered a duplicate (0.0–1.0).
            duplicate_check_k:    Number of neighbours to inspect during
                                  semantic duplicate checking.
            embedding_max_retries: Retry attempts for embedding API calls.

        Raises:
            ValueError: If any argument fails validation.
        """
        self._validate_init_args(
            persist_directory,
            embedding_model,
            similarity_threshold,
            duplicate_check_k,
            embedding_max_retries,

        )

        self.persist_directory = persist_directory
        self.similarity_threshold = similarity_threshold
        self.duplicate_check_k = duplicate_check_k
        self.embedding_max_retries = embedding_max_retries
        self._embedding_model_name = embedding_model
        self._embedding_function = OpenAIEmbeddings(model=self._embedding_model_name)

        logger.debug(
            "VectorStoreManager initialised | persist_directory=%s "
            "embedding_model=%s similarity_threshold=%.2f",
            persist_directory,
            embedding_model,
            similarity_threshold,
        )

    # ------------------------------------------------------------------ #
    #  Entry point                                                         #
    # ------------------------------------------------------------------ #

    def update_vector_store(
        self, documents: List[Document], document_name: str) -> bool:
        """
        Create or update the ChromaDB vector store with new documents.

        If the store already exists and the document has been indexed before,
        the ingest is skipped. Otherwise documents are deduplicated semantically
        before being upserted.

        Args:
            documents:     LangChain Documents produced by PDFPartitioner.
            document_name: Logical name used to check for existing ingests.

        Returns:
            True on success.

        Raises:
            ValueError:             If arguments are invalid.
            DocumentIDError:        If any document is missing an ID.
            VectorStoreLoadError:   If the existing store cannot be loaded.
            VectorStoreCreateError: If a new store cannot be created.
            VectorStoreUpdateError: If upserting documents fails.
            EmbeddingError:         If the embedding API fails after all retries.
        """
        self._validate_documents(documents)
        self._validate_document_name(document_name)

        logger.info(
            "Starting vector store update | document=%s documents=%d",
            document_name,
            len(documents),
        )

        if os.path.exists(self.persist_directory):
            self._update_existing_store(documents, document_name)
        else:
            self._create_new_store(documents)

        logger.info(
            "Vector store update complete | document=%s", document_name
        )
        return True

    # ------------------------------------------------------------------ #
    #  Store operations (private)                                         #
    # ------------------------------------------------------------------ #

    def _create_new_store(self, documents: List[Document]) -> None:
        """Create a brand-new ChromaDB store from documents.

        Raises:
            VectorStoreCreateError: If Chroma raises during creation.
            EmbeddingError:         If the embedding API fails after all retries.
        """
        logger.info(
            "No existing store found — creating new store at: %s",
            self.persist_directory,
        )
        ids = self._extract_chunk_ids(documents)

        try:
            self._embed_and_create(documents, ids)
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"Failed to create vector store at '{self.persist_directory}': {e}"   )   
            raise VectorStoreCreateError(
                f"Failed to create vector store at '{self.persist_directory}': {e}"
            ) from e

        logger.info(
            "Vector store created and saved to: %s", self.persist_directory
        )

    def _update_existing_store(
        self, documents: List[Document], document_name: str
    ) -> None:
        """Load an existing store and upsert new, non-duplicate documents.

        Raises:
            VectorStoreLoadError:   If the store cannot be loaded.
            VectorStoreUpdateError: If the upsert fails.
            EmbeddingError:         If the embedding API fails after all retries.
        """
        logger.info(
            "Existing store found at %s — loading", self.persist_directory
        )
        vectorstore = self._load_store()

        if self._document_exists(vectorstore, document_name):
            logger.info(
                "Document '%s' already indexed — skipping ingest", document_name
            )
            return

        non_duplicate_docs = self._filter_semantic_duplicates(
            vectorstore, documents, document_name
        )

        if not non_duplicate_docs:
            logger.info(
                "All %d documents were semantic duplicates — nothing to upsert",
                len(documents),
            )
            return

        ids = self._extract_chunk_ids(non_duplicate_docs)

        try:
            self._embed_and_add(vectorstore, non_duplicate_docs, ids)
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"Failed to upsert documents into store at '{self.persist_directory}': {e}")
            raise VectorStoreUpdateError(
                f"Failed to upsert documents into store at "
                f"'{self.persist_directory}': {e}"
            ) from e

        logger.info(
            "Upserted %d documents into existing store", len(non_duplicate_docs)
        )

    def _load_store(self) -> Chroma:
        """Load the existing ChromaDB vector store from disk.

        Raises:
            VectorStoreLoadError: If Chroma raises during loading.
        """
        try:
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self._embedding_function,
                collection_metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            logger.error(f"Failed to load vector store from '{self.persist_directory}': {e}") 
            raise VectorStoreLoadError(
                f"Failed to load vector store from '{self.persist_directory}': {e}"
            ) from e

        count = vectorstore._collection.count()
        logger.info("Vector store loaded — %d documents in collection", count)
        return vectorstore

    # ------------------------------------------------------------------ #
    #  Embedding API calls with retry                                     #
    # ------------------------------------------------------------------ #

    def _embed_and_create(
        self, documents: List[Document], ids: List[str]
    ) -> Chroma:
        """Call Chroma.from_documents with retry on transient embedding errors.

        Raises:
            EmbeddingError: If all retry attempts are exhausted.
        """
        @retry(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(self.embedding_max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=False,
        )
        def _call() -> Chroma:
            return Chroma.from_documents(
                documents=documents,
                ids=ids,
                embedding=self._embedding_function,
                persist_directory=self.persist_directory,
                collection_metadata={"hnsw:space": "cosine"},
            )

        try:
            return _call()
        except Exception as e:
            logger.error(f"Embedding API failed after {self.embedding_max_retries} attempts during store creation: {e}")
            raise EmbeddingError(
                f"Embedding API failed after {self.embedding_max_retries} "
                f"attempts during store creation: {e}"
            ) from e

    def _embed_and_add(
        self, vectorstore: Chroma, documents: List[Document], ids: List[str]
    ) -> None:
        """Call vectorstore.add_documents with retry on transient embedding errors.

        Raises:
            EmbeddingError: If all retry attempts are exhausted.
        """
        @retry(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(self.embedding_max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=False,
        )
        def _call() -> None:
            vectorstore.add_documents(documents=documents, ids=ids)

        try:
            _call()
        except Exception as e:
            logger.error(f"Embedding API failed after {self.embedding_max_retries} attempts during store update: {e}")
            raise EmbeddingError(
                f"Embedding API failed after {self.embedding_max_retries} "
                f"attempts during store update: {e}"
            ) from e

    # ------------------------------------------------------------------ #
    #  Deduplication                                                      #
    # ------------------------------------------------------------------ #

    def _filter_semantic_duplicates(
        self,
        vectorstore: Chroma,
        documents: List[Document],
        document_name: str,
    ) -> List[Document]:
        """Return only documents that are not semantic duplicates of existing vectors."""
        non_duplicates = []

        for doc in documents:
            try:
                is_duplicate, similarity = self._is_semantic_duplicate(
                    vectorstore, document_name, doc.page_content
                )
            except VectorStoreQueryError as e:
                logger.warning(
                    "Duplicate check failed for chunk %s — including it anyway. Reason: %s",
                    doc.id,
                    e,
                )
                non_duplicates.append(doc)
                continue

            if is_duplicate:
                logger.debug(
                    "Skipping semantic duplicate chunk %s (similarity=%.4f)",
                    doc.id,
                    similarity,
                )
            else:
                logger.debug(
                    "Keeping chunk %s (similarity=%.4f)", doc.id, similarity
                )
                non_duplicates.append(doc)

        skipped = len(documents) - len(non_duplicates)
        if skipped:
            logger.info(
                "Semantic deduplication: skipped %d/%d duplicate chunks",
                skipped,
                len(documents),
            )

        return non_duplicates

    def _is_semantic_duplicate(
        self,
        vectorstore: Chroma,
        doc_name: str,
        text: str,
    ) -> Tuple[bool, float]:
        """Check whether a near-identical chunk already exists in the store.

        Returns:
            (is_duplicate, highest_similarity_score)

        Raises:
            VectorStoreQueryError: If the similarity search fails.
        """
        try:
            results = vectorstore.similarity_search_with_score(
                text,
                k=self.duplicate_check_k,
                filter={"document_name": doc_name},
            )
        except Exception as e:
            logger.error(f"Similarity search failed for document '{doc_name}': {e}")
            raise VectorStoreQueryError(
                f"Similarity search failed for document '{doc_name}': {e}"
            ) from e

        highest_similarity = 0.0
        for _, score in results:
            similarity = 1 - score  # Chroma returns cosine distance
            if similarity > highest_similarity:
                highest_similarity = similarity
            if similarity >= self.similarity_threshold:
                return True, similarity

        return False, highest_similarity

    # ------------------------------------------------------------------ #
    #  Public utilities                                                   #
    # ------------------------------------------------------------------ #

    def number_of_documents(self) -> int:
        """Return the total number of vectors in the store.

        Raises:
            VectorStoreLoadError: If the store cannot be loaded.
        """
        vectorstore = self._load_store()
        count = vectorstore._collection.count()
        logger.info("Vector store contains %d documents", count)
        return count

    def get_all_document_ids(self) -> List[str]:
        """Return all document IDs stored in the collection.

        Raises:
            VectorStoreLoadError: If the store cannot be loaded.
            VectorStoreQueryError: If the ID fetch fails.
        """
        vectorstore = self._load_store()
        try:
            result = vectorstore._collection.get(include=[])
        except Exception as e:
            logger.error(f"Failed to retrieve document IDs from store at '{self.persist_directory}': {e}")
            raise VectorStoreQueryError(
                f"Failed to retrieve document IDs: {e}"
            ) from e

        ids = result.get("ids", [])
        logger.info("Retrieved %d document IDs from store", len(ids))
        return ids

    def safe_bulk_delete_all_vectors(
        self, batch_size: int = DEFAULT_DELETE_BATCH_SIZE
    ) -> None:
        """Delete ALL vectors from the collection in batches.

        Preserves the collection itself. Safe for large collections.

        Args:
            batch_size: Number of vectors to delete per batch.

        Raises:
            ValueError:            If batch_size is invalid.
            VectorStoreLoadError:  If the store cannot be loaded.
            VectorStoreDeleteError: If deletion is incomplete.
        """
        if batch_size < 1:
            logger.error(f"batch_size must be >= 1, got {batch_size}.")
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

        vectorstore = self._load_store()
        collection = vectorstore._collection
        total = collection.count()

        logger.info("Starting bulk delete of %d vectors", total)

        if total == 0:
            logger.info("Collection already empty — nothing to delete")
            return

        deleted = 0
        while True:
            try:
                result = collection.get(limit=batch_size, include=[])
                ids = result.get("ids", [])
                if not ids:
                    break
                collection.delete(ids=ids)
            except Exception as e:
                logger.error(f"Batch delete failed after removing {deleted}/{total} vectors: {e}")
                raise VectorStoreDeleteError(
                    f"Batch delete failed after removing {deleted}/{total} "
                    f"vectors: {e}"
                ) from e

            deleted += len(ids)
            logger.info("Deleted %d/%d vectors", deleted, total)

        remaining = collection.count()
        if remaining != 0:
            logger.error(f"Bulk delete incomplete: {remaining} vectors still remain after attempting to delete {total}.")
            raise VectorStoreDeleteError(
                f"Bulk delete incomplete: {remaining} vectors still remain "
                f"after attempting to delete {total}."
            )

        logger.info("Bulk delete completed successfully — all %d vectors removed", total)

    # ------------------------------------------------------------------ #
    #  Helpers (private)                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _document_exists(vectorstore: Chroma, document_name: str) -> bool:
        """Return True if any vector exists for the given document_name."""
        collection = vectorstore._collection
        result = collection.get(
            where={"document_name": document_name},
            limit=1,
            include=[],
        )
        exists = len(result.get("ids", [])) > 0
        logger.debug(
            "Document existence check | document_name=%s exists=%s",
            document_name,
            exists,
        )
        return exists

    @staticmethod
    def _extract_chunk_ids(documents: List[Document]) -> List[str]:
        """Extract stable Chroma-compatible IDs from LangChain Documents.

        Raises:
            DocumentIDError: If any document is missing an ID.
        """
        ids = []
        for i, doc in enumerate(documents):
            if not hasattr(doc, "id") or doc.id is None:
                raise DocumentIDError(
                    f"Document at index {i} is missing 'id'. "
                    "doc.id is required for Chroma upserts."
                )
            ids.append(str(doc.id))
        return ids

    # ------------------------------------------------------------------ #
    #  Validation                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_init_args(
        persist_directory: str,
        embedding_model: str,
        similarity_threshold: float,
        duplicate_check_k: int,
        embedding_max_retries: int,
    ) -> None:
        """Validate constructor arguments.

        Raises:
            ValueError: On any invalid argument.
        """
        if not persist_directory or not persist_directory.strip():
            logger.error("persist_directory must be a non-empty string.")
            raise ValueError("persist_directory must be a non-empty string.")
        
        if not embedding_model or not embedding_model.strip():
            logger.error("embedding_model must be a non-empty string.")
            raise ValueError("embedding_model must be a non-empty string.")

        if not (0.0 < similarity_threshold <= 1.0):
            logger.error(f"similarity_threshold must be between 0.0 and 1.0, got {similarity_threshold}.")
            raise ValueError(
                f"similarity_threshold must be between 0.0 and 1.0, "
                f"got {similarity_threshold}."
            )

        if duplicate_check_k < 1:
            logger.error(f"duplicate_check_k must be >= 1, got {duplicate_check_k}.")
            raise ValueError(
                f"duplicate_check_k must be >= 1, got {duplicate_check_k}."
            )

        if embedding_max_retries < 1:
            logger.error(f"embedding_max_retries must be >= 1, got {embedding_max_retries}.")
            raise ValueError(
                f"embedding_max_retries must be >= 1, got {embedding_max_retries}."
            )

    @staticmethod
    def _validate_documents(documents: List[Document]) -> None:
        """Validate the documents list.

        Raises:
            ValueError: If documents is None, empty, or not a list.
        """
        if documents is None:
            logger.error("documents must not be None.")
            raise ValueError("documents must not be None.")

        if not isinstance(documents, list):
            logger.error(f"documents must be a list, got {type(documents).__name__}.")
            raise ValueError(
                f"documents must be a list, got {type(documents).__name__}."
            )

        if len(documents) == 0:
            logger.error("documents must not be empty.")
            raise ValueError("documents must not be empty.")

    @staticmethod
    def _validate_document_name(document_name: str) -> None:
        """Validate document_name.

        Raises:
            ValueError: If document_name is empty or not a string.
        """
        if not isinstance(document_name, str) or not document_name.strip():
            logger.error("document_name must be a non-empty string.")
            raise ValueError("document_name must be a non-empty string.")