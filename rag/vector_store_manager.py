import logging
from typing import Any, List, Tuple
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
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

logger = logging.getLogger("default_logger")


# ------------------------------------------------------------------ #
#  Custom exceptions                                                  #
# ------------------------------------------------------------------ #

class VectorStoreError(Exception):
    """Base exception for all VectorStore errors."""


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

class VectorStoreManager(BaseModel):
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
        ValueError:             If constructor or method arguments are invalid.
        VectorStoreUpdateError: If adding documents fails.
        VectorStoreDeleteError: If a bulk delete operation fails.
        VectorStoreQueryError:  If a similarity search fails.
        DocumentIDError:        If a document is missing a required ID.
        EmbeddingError:         If the embedding API fails after all retries.
    """

    # ------------------------------------------------------------------ #
    #  Pydantic fields                                                    #
    # ------------------------------------------------------------------ #

    persist_directory: str = Field(
        ...,
        min_length=1,
        description="Path where ChromaDB persists its data.",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        min_length=1,
        description="OpenAI embedding model name.",
    )
    similarity_threshold: float = Field(
        default=0.98,
        gt=0.0,
        le=1.0,
        description="Cosine similarity above which a chunk is considered a duplicate (0.0–1.0).",
    )
    duplicate_check_k: int = Field(
        default=3,
        ge=1,
        description="Number of neighbours to inspect during semantic duplicate checking.",
    )
    embedding_max_retries: int = Field(
        default=3,
        ge=1,
        description="Retry attempts for embedding API calls.",
    )
    delete_batch_size: int = Field(
        default=1000,
        ge=1,
        description="Number of vectors to delete per batch in bulk deletes.",
    )
    # Excluded from serialisation; populated by model_post_init.
    embedding_function: Any = Field(default=None, exclude=True)
    chroma_client: Any = Field(default=None, exclude=True)

    # ------------------------------------------------------------------ #
    #  Field validators                                                   #
    # ------------------------------------------------------------------ #

    @field_validator("persist_directory")
    @classmethod
    def _validate_persist_directory(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("persist_directory must not be blank or whitespace.")
        return v

    @field_validator("embedding_model")
    @classmethod
    def _validate_embedding_model(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("embedding_model must not be blank or whitespace.")
        return v

    # ------------------------------------------------------------------ #
    #  Post-init: build embedding function and Chroma client              #
    # ------------------------------------------------------------------ #

    def model_post_init(self, __context: Any) -> None:
        """Instantiate the embedding function and Chroma client after Pydantic validation.

        Uses object.__setattr__ because BaseModel fields are immutable by default.
        """
        embedding_fn = OpenAIEmbeddings(model=self.embedding_model)
        object.__setattr__(self, "embedding_function", embedding_fn)
        object.__setattr__(self, "chroma_client", self._open_client())

        logger.debug(
            "VectorStoreManager initialised | persist_directory=%s "
            "embedding_model=%s similarity_threshold=%.2f",
            self.persist_directory,
            self.embedding_model,
            self.similarity_threshold,
        )

    # ------------------------------------------------------------------ #
    #  Client                                                             #
    # ------------------------------------------------------------------ #

    def _open_client(self) -> Chroma:
        """Create and return a persistent Chroma client.

        Called once during model_post_init to populate self.chroma_client.
        If the store does not yet exist on disk, a client is still returned;
        the collection will be created on the first write.

        Raises:
            VectorStoreLoadError: If Chroma raises during client initialisation.
        """
        try:
            client = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function,
                collection_metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            logger.error(
                "Failed to initialise Chroma client at '%s': %s",
                self.persist_directory, e,
            )
            raise VectorStoreLoadError(
                f"Failed to initialise Chroma client at '{self.persist_directory}': {e}"
            ) from e

        logger.debug(
            "Chroma client initialised | persist_directory=%s",
            self.persist_directory,
        )
        return client

    # ------------------------------------------------------------------ #
    #  Entry point                                                        #
    # ------------------------------------------------------------------ #

    def update_vector_store(
        self, documents: List[Document], document_name: str
    ) -> bool:
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
            VectorStoreUpdateError: If upserting documents fails.
            EmbeddingError:         If the embedding API fails after all retries.
        """
        self._validate_documents(documents)
        self._validate_document_name(document_name)

        num_documents_in_store = self.number_of_documents()
        logger.info(
            "Preparing to update vector store at '%s' with %d documents "
            "for document_name='%s'. Current store size: %d documents.",
            self.persist_directory,
            len(documents),
            document_name,
            num_documents_in_store,
        )
        logger.info(
            "Starting vector store update | document=%s documents=%d",
            document_name,
            len(documents),
        )

        if self._document_exists(document_name):
            logger.info(
                "Document '%s' already indexed — skipping ingest", document_name
            )
            return True

        self._upsert_documents(documents, document_name)
        
        num_documents_in_store = self.number_of_documents()
        logger.info(
            "Vector store update complete and now contains %d documents after update.",
            num_documents_in_store
        )
        return True

    # ------------------------------------------------------------------ #
    #  Store operations (private)                                         #
    # ------------------------------------------------------------------ #

    def _upsert_documents(
        self, documents: List[Document], document_name: str
    ) -> None:
        """Deduplicate and upsert documents into the Chroma client.

        Works identically whether the collection is empty (new store) or
        already populated — Chroma handles both cases transparently via
        the shared chroma_client initialised in model_post_init.

        Raises:
            VectorStoreUpdateError: If the upsert fails.
            EmbeddingError:         If the embedding API fails after all retries.
        """
        non_duplicate_docs = self._filter_semantic_duplicates(
            documents, document_name
        )

        if not non_duplicate_docs:
            logger.info(
                "All %d documents were semantic duplicates — nothing to upsert",
                len(documents),
            )
            return

        ids = self._extract_chunk_ids(non_duplicate_docs)

        try:
            self._embed_and_upsert(non_duplicate_docs, ids)
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(
                "Failed to upsert documents into store at '%s': %s",
                self.persist_directory, e,
            )
            raise VectorStoreUpdateError(
                f"Failed to upsert documents into store at "
                f"'{self.persist_directory}': {e}"
            ) from e

        logger.info("Upserted %d documents into store", len(non_duplicate_docs))

    # ------------------------------------------------------------------ #
    #  Embedding API calls with retry                                     #
    # ------------------------------------------------------------------ #

    def _embed_and_upsert(
        self, documents: List[Document], ids: List[str]
    ) -> None:
        """Call chroma_client.add_documents with retry on transient embedding errors.

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
            self.chroma_client.add_documents(documents=documents, ids=ids)

        try:
            _call()
        except Exception as e:
            logger.error(
                "Embedding API failed after %d attempts during upsert: %s",
                self.embedding_max_retries, e,
            )
            raise EmbeddingError(
                f"Embedding API failed after {self.embedding_max_retries} "
                f"attempts during upsert: {e}"
            ) from e

    # ------------------------------------------------------------------ #
    #  Deduplication                                                      #
    # ------------------------------------------------------------------ #

    def _filter_semantic_duplicates(
        self,
        documents: List[Document],
        document_name: str,
    ) -> List[Document]:
        """Return only documents that are not semantic duplicates of existing vectors."""
        non_duplicates = []

        for doc in documents:
            try:
                is_duplicate, similarity = self._is_semantic_duplicate(
                    document_name, doc.page_content
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
            results = self.chroma_client.similarity_search_with_score(
                text,
                k=self.duplicate_check_k,
                filter={"document_name": doc_name},
            )
        except Exception as e:
            logger.error(
                "Similarity search failed for document '%s': %s", doc_name, e
            )
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
            VectorStoreQueryError: If the count query fails.
        """
        try:
            count = self.chroma_client._collection.count()
        except Exception as e:
            logger.error(
                "Count query on store at '%s' failed: %s",
                self.persist_directory, e,
            )
            raise VectorStoreQueryError(
                f"Count query on store at '{self.persist_directory}' failed: {e}"
            ) from e

        logger.info("Vector store contains %d documents", count)
        return count

    def get_all_document_ids(self) -> List[str]:
        """Return all document IDs stored in the collection.

        Raises:
            VectorStoreQueryError: If the ID fetch fails.
        """
        try:
            result = self.chroma_client._collection.get(include=[])
        except Exception as e:
            logger.error(
                "Failed to retrieve document IDs from store at '%s': %s",
                self.persist_directory, e,
            )
            raise VectorStoreQueryError(
                f"Failed to retrieve document IDs: {e}"
            ) from e

        ids = result.get("ids", [])
        logger.info("Retrieved %d document IDs from store", len(ids))
        return ids

    def safe_bulk_delete_all_vectors(self) -> None:
        """Delete ALL vectors from the collection in batches.

        Preserves the collection itself. Safe for large collections.

        Uses self.delete_batch_size to control batch sizing.

        Raises:
            VectorStoreDeleteError: If deletion is incomplete.
        """
        collection = self.chroma_client._collection
        total = collection.count()

        logger.info("Starting bulk delete of %d vectors", total)

        if total == 0:
            logger.info("Collection already empty — nothing to delete")
            return

        deleted = 0
        while True:
            try:
                result = collection.get(limit=self.delete_batch_size, include=[])
                ids = result.get("ids", [])
                if not ids:
                    break
                collection.delete(ids=ids)
            except Exception as e:
                logger.error(
                    "Batch delete failed after removing %d/%d vectors: %s",
                    deleted, total, e,
                )
                raise VectorStoreDeleteError(
                    f"Batch delete failed after removing {deleted}/{total} "
                    f"vectors: {e}"
                ) from e

            deleted += len(ids)
            logger.info("Deleted %d/%d vectors", deleted, total)

        remaining = collection.count()
        if remaining != 0:
            logger.error(
                "Bulk delete incomplete: %d vectors still remain after "
                "attempting to delete %d.",
                remaining, total,
            )
            raise VectorStoreDeleteError(
                f"Bulk delete incomplete: {remaining} vectors still remain "
                f"after attempting to delete {total}."
            )

        logger.info(
            "Bulk delete completed successfully — all %d vectors removed", total
        )

    # ------------------------------------------------------------------ #
    #  Helpers (private)                                                  #
    # ------------------------------------------------------------------ #

    def _document_exists(self, document_name: str) -> bool:
        """Return True if any vector exists for the given document_name."""
        collection = self.chroma_client._collection
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
    def _validate_documents(documents: List[Document]) -> None:
        """Validate the documents list.

        Raises:
            ValueError: If documents is None, empty, or not a list.
        """
        if documents is None:
            logger.error("documents must not be None.")
            raise ValueError("documents must not be None.")

        if not isinstance(documents, list):
            logger.error(
                "documents must be a list, got %s.", type(documents).__name__
            )
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