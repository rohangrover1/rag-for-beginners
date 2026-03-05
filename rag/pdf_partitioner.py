import json
import hashlib
import logging
import re
import unicodedata
from pathlib import Path
from typing import List

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv, find_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

_ = load_dotenv(find_dotenv())

# ------------------------------------------------------------------ #
#  Module-level logger                                                #
# ------------------------------------------------------------------ #

logger = logging.getLogger("default_logger")


# ------------------------------------------------------------------ #
#  Custom exceptions                                                  #
# ------------------------------------------------------------------ #

class PDFPartitionerError(Exception):
    """Base exception for all PDFPartitioner errors."""


class PDFFileNotFoundError(PDFPartitionerError):
    """Raised when the specified PDF file does not exist."""


class PDFExtractionError(PDFPartitionerError):
    """Raised when unstructured fails to extract elements from the PDF."""


class ChunkingError(PDFPartitionerError):
    """Raised when chunk creation fails."""


class AISummaryError(PDFPartitionerError):
    """Raised when all AI summary retry attempts are exhausted."""


class ExportError(PDFPartitionerError):
    """Raised when exporting chunks to JSON fails."""


# ------------------------------------------------------------------ #
#  Main class                                                         #
# ------------------------------------------------------------------ #

class PDFPartitioner:
    """
    Processes PDF documents into AI-enhanced, searchable chunks.

    Entry point: `partition_document(file_path)`

    Example usage:
        logging.basicConfig(level=logging.INFO)
        partitioner = PDFPartitioner(document_name="my_report")
        docs = partitioner.partition_document("path/to/file.pdf")
        PDFPartitioner.export_chunks_to_json(docs)

    Raises:
        ValueError:            If constructor arguments are invalid.
        PDFFileNotFoundError:  If the PDF path does not exist.
        PDFExtractionError:    If element extraction fails.
        ChunkingError:         If chunk creation fails.
        AISummaryError:        If the AI summary fails after all retries.
        ExportError:           If writing the JSON output fails.
    """

    SUPPORTED_EXTENSIONS = {".pdf"}

    def __init__(
        self,
        document_name: str = "document",
        max_characters: int = 3000,
        new_after_n_chars: int = 2400,
        combine_text_under_n_chars: int = 500,
        llm_model: str = "gpt-4o",
        ai_max_retries: int = 3,
    ):
        """
        Args:
            document_name:              Identifier embedded in chunk IDs and metadata.
            max_characters:             Hard character limit per chunk.
            new_after_n_chars:          Soft limit — start a new chunk after this many chars.
            combine_text_under_n_chars: Merge chunks smaller than this with their neighbours.
            llm_model:                  OpenAI model used for AI-enhanced summaries.
            ai_max_retries:             Number of retry attempts for AI summary calls.

        Raises:
            ValueError: If any argument fails validation.
        """
        self._validate_init_args(
            document_name,
            max_characters,
            new_after_n_chars,
            combine_text_under_n_chars,
            ai_max_retries,
        )

        self.document_name = document_name
        self.max_characters = max_characters
        self.new_after_n_chars = new_after_n_chars
        self.combine_text_under_n_chars = combine_text_under_n_chars
        self.llm_model = llm_model
        self.ai_max_retries = ai_max_retries

        logger.debug(
            "PDFPartitioner initialised | document_name=%s model=%s max_chars=%d",
            document_name,
            llm_model,
            max_characters,
        )

    # ------------------------------------------------------------------ #
    #  Entry point                                                         #
    # ------------------------------------------------------------------ #

    def partition_document(self, file_path: str, chunk_index: int = 0) -> List[Document]:
        """
        Full pipeline: validate → extract → chunk → summarise.

        Args:
            file_path:    Path to the PDF file.
            chunk_index:  Starting index offset written into chunk metadata.

        Returns:
            List of LangChain Documents ready for ingestion into a vector store.

        Raises:
            ValueError:           If file_path is empty or not a supported extension.
            PDFFileNotFoundError: If the file does not exist.
            PDFExtractionError:   If element extraction fails.
            ChunkingError:        If chunking fails.
        """
        self._validate_file_path(file_path)

        logger.info("Starting partition pipeline for: %s", file_path)

        elements = self._extract_elements(file_path)
        chunks = self._create_chunks(elements)
        documents = self._summarise_chunks(chunks, chunk_index=chunk_index)

        logger.info("Pipeline complete | file=%s chunks=%d", file_path, len(documents))
        return documents

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_init_args(
        document_name: str,
        max_characters: int,
        new_after_n_chars: int,
        combine_text_under_n_chars: int,
        ai_max_retries: int,
    ) -> None:
        """Validate constructor arguments and raise ValueError on any failure."""
        if not document_name or not document_name.strip():
            logger.error("document_name must be a non-empty string.")
            raise ValueError("document_name must be a non-empty string.")

        if max_characters <= 0:
            logger.error(f"max_characters must be > 0, got {max_characters}.")
            raise ValueError(f"max_characters must be > 0, got {max_characters}.")

        if new_after_n_chars <= 0:
            logger.error(f"new_after_n_chars must be > 0, got {new_after_n_chars}.")
            raise ValueError(f"new_after_n_chars must be > 0, got {new_after_n_chars}.")

        if new_after_n_chars >= max_characters:
            logger.error(
                f"new_after_n_chars ({new_after_n_chars}) must be less than "
                f"max_characters ({max_characters})."
            )
            raise ValueError(
                f"new_after_n_chars ({new_after_n_chars}) must be less than "
                f"max_characters ({max_characters})."
            )

        if combine_text_under_n_chars < 0:
            logger.error(
                f"combine_text_under_n_chars must be >= 0, got {combine_text_under_n_chars}."
            )
            raise ValueError(
                f"combine_text_under_n_chars must be >= 0, got {combine_text_under_n_chars}."
            )

        if combine_text_under_n_chars >= new_after_n_chars:
            logger.error(
                f"combine_text_under_n_chars ({combine_text_under_n_chars}) must be less than "
                f"new_after_n_chars ({new_after_n_chars})."
            )
            raise ValueError(
                f"combine_text_under_n_chars ({combine_text_under_n_chars}) must be less than "
                f"new_after_n_chars ({new_after_n_chars})."
            )

        if ai_max_retries < 1:
            logger.error(f"ai_max_retries must be >= 1, got {ai_max_retries}.")
            raise ValueError(f"ai_max_retries must be >= 1, got {ai_max_retries}.")

    def _validate_file_path(self, file_path: str) -> None:
        """Validate the file path before processing."""
        if not file_path or not file_path.strip():
            logger.error("file_path must be a non-empty string.")
            raise ValueError("file_path must be a non-empty string.")

        path = Path(file_path)

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            logger.error(
                f"Unsupported file type '{path.suffix}' for file: {file_path}. "
                f"Supported types: {self.SUPPORTED_EXTENSIONS}"
            )
            raise ValueError(
                f"Unsupported file type '{path.suffix}'. "
                f"Supported types: {self.SUPPORTED_EXTENSIONS}"
            )

        if not path.exists() or not path.is_file():
            logger.error(f"PDF file not found: {file_path}")
            raise PDFFileNotFoundError(f"PDF file not found: {file_path}")

    # ------------------------------------------------------------------ #
    #  Pipeline steps (private)                                           #
    # ------------------------------------------------------------------ #

    def _extract_elements(self, file_path: str):
        """Extract raw elements from PDF using unstructured.

        Raises:
            PDFExtractionError: If unstructured raises any exception, or returns nothing.
        """
        logger.info("Extracting elements from: %s", file_path)
        try:
            elements = partition_pdf(
                filename=file_path,
                strategy="hi_res",
                infer_table_structure=True,
                extract_image_block_types=["Image"],
                extract_image_block_to_payload=True,
                languages=["English"],
            )
        except Exception as e:
            logger.error(f"Failed to extract elements from '{file_path}': {e}")
            raise PDFExtractionError(
                f"Failed to extract elements from '{file_path}': {e}"
            ) from e

        if not elements:
            logger.error(f"No elements extracted from '{file_path}'. The file may be empty, image-only, or corrupt.")
            raise PDFExtractionError(
                f"No elements were extracted from '{file_path}'. "
                "The file may be empty, image-only, or corrupt."
            )

        logger.info("Extracted %d elements", len(elements))
        return elements

    def _create_chunks(self, elements):
        """Create intelligent, title-based chunks from elements.

        Raises:
            ChunkingError: If chunk_by_title raises, or produces no output.
        """
        logger.info("Creating chunks from %d elements", len(elements))
        try:
            chunks = chunk_by_title(
                elements,
                max_characters=self.max_characters,
                new_after_n_chars=self.new_after_n_chars,
                combine_text_under_n_chars=self.combine_text_under_n_chars,
            )
        except Exception as e:
            logger.error(f"Failed to create chunks: {e}")
            raise ChunkingError(f"Failed to create chunks: {e}") from e

        if not chunks:
            logger.error("Chunking produced no output from the extracted elements.")
            raise ChunkingError("Chunking produced no output from the extracted elements.")

        logger.info("Created %d chunks", len(chunks))
        return chunks

    def _summarise_chunks(self, chunks, chunk_index: int = 0) -> List[Document]:
        """Process all chunks, applying AI summaries where content warrants it."""
        logger.info("Summarising %d chunks (starting at index %d)", len(chunks), chunk_index)

        documents = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            logger.debug("Processing chunk %d/%d", i + 1, total)

            content_data = self._separate_content_types(chunk)

            if content_data["tables"] or content_data["images"]:
                logger.debug(
                    "Chunk %d has mixed content — requesting AI summary "
                    "(tables=%d, images=%d)",
                    i + 1,
                    len(content_data["tables"]),
                    len(content_data["images"]),
                )
                try:
                    page_content = self._create_ai_enhanced_summary(
                        content_data["text"],
                        content_data["tables"],
                        content_data["images"],
                    )
                    ai_enhanced = True
                except AISummaryError as e:
                    # All retries exhausted — fall back gracefully and log a warning
                    logger.warning(
                        "AI summary failed for chunk %d/%d after all retries; "
                        "falling back to raw text. Reason: %s",
                        i + 1,
                        total,
                        e,
                    )
                    page_content = content_data["text"]
                    ai_enhanced = False
            else:
                logger.debug("Chunk %d — using raw text (no tables/images)", i + 1)
                page_content = content_data["text"]
                ai_enhanced = False

            page_content = self._canonicalize_text(page_content)

            doc_id = self._chunk_id(page_content)
            doc = Document(
                page_content=page_content,
                metadata={
                    "doc_id":        doc_id,          
                    "document_name": self.document_name,
                    "chunk_index":   chunk_index + i,
                    "ai_enhanced":   ai_enhanced,                           
                    "raw_text":      content_data["text"],
                    "tables_html":   json.dumps(content_data["tables"]),   # scalar-safe
                    "images_base64": json.dumps(content_data["images"]),   # scalar-safe
                },
                id=doc_id
            )
            documents.append(doc)

        logger.info("Finished summarising — produced %d documents", len(documents))
        return documents

    # ------------------------------------------------------------------ #
    #  Helpers (private)                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _canonicalize_text(text: str) -> str:
        """Normalize text for consistent chunk IDs and comparisons."""
        text = unicodedata.normalize("NFKD", text)
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip()

    @staticmethod
    def _separate_content_types(chunk) -> dict:
        """Analyse what types of content are present in a chunk."""
        content_data: dict = {
            "text": chunk.text,
            "tables": [],
            "images": [],
            "types": ["text"],
        }

        if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
            for element in chunk.metadata.orig_elements:
                element_type = type(element).__name__

                if element_type == "Table":
                    content_data["types"].append("table")
                    table_html = getattr(element.metadata, "text_as_html", element.text)
                    content_data["tables"].append(table_html)

                elif element_type == "Image":
                    if hasattr(element, "metadata") and hasattr(
                        element.metadata, "image_base64"
                    ):
                        content_data["types"].append("image")
                        content_data["images"].append(element.metadata.image_base64)

        content_data["types"] = list(set(content_data["types"]))
        return content_data

    def _create_ai_enhanced_summary(
        self, text: str, tables: List[str], images: List[str]
    ) -> str:
        """Create an AI-enhanced, searchable summary for mixed content.

        Retries automatically on transient OpenAI errors (rate limits, timeouts,
        server errors). Raises AISummaryError if all attempts are exhausted.

        Raises:
            AISummaryError: When all retry attempts fail.
        """
        @retry(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(self.ai_max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=False,
        )
        def _call_llm() -> str:
            llm = ChatOpenAI(model=self.llm_model, temperature=0)

            prompt_text = (
                "You are creating a searchable description for document content retrieval.\n\n"
                "CONTENT TO ANALYZE:\n"
                f"TEXT CONTENT:\n{text}\n"
            )

            if tables:
                prompt_text += "TABLES:\n"
                for idx, table in enumerate(tables):
                    prompt_text += f"Table {idx + 1}:\n{table}\n\n"

            prompt_text += (
                "\nYOUR TASK:\n"
                "Generate a comprehensive, searchable description that covers:\n\n"
                "1. Key facts, numbers, and data points from text and tables\n"
                "2. Main topics and concepts discussed\n"
                "3. Questions this content could answer\n"
                "4. Visual content analysis (charts, diagrams, patterns in images)\n"
                "5. Alternative search terms users might use\n\n"
                "Make it detailed and searchable - prioritize findability over brevity.\n\n"
                "SEARCHABLE DESCRIPTION:"
            )

            message_content: list = [{"type": "text", "text": prompt_text}]
            for image_base64 in images:
                message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    }
                )

            response = llm.invoke([HumanMessage(content=message_content)])
            return response.content

        try:
            return _call_llm()
        except Exception as e:
            logger.error(f"AI summary failed after {self.ai_max_retries} attempts: {e}")
            raise AISummaryError(
                f"AI summary failed after {self.ai_max_retries} attempts: {e}"
            ) from e

    def _chunk_id(self, chunk_text: str) -> str:
        """Generate a unique chunk ID using document name and a SHA-256 hash."""
        h = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        return f"{self.document_name}:{h}"

    # ------------------------------------------------------------------ #
    #  Export                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def export_chunks_to_json(
        chunks: List[Document],
        output_path: str = "processed_chunks.json",
    ) -> None:
        """Export processed chunks to a clean JSON file for inspection.

        Args:
            chunks:      List of LangChain Documents to export.
            output_path: Destination file path.

        Raises:
            ValueError:  If chunks is empty.
            ExportError: If the file cannot be written.
        """
        if not chunks:
            logger.error("chunks must not be empty.")
            raise ValueError("chunks must not be empty.")

        export_data = [
            {
                "doc_id": doc.metadata.get("doc_id"),
                "enhanced_content": doc.page_content,
                "chunk_index":      doc.metadata.get("chunk_index"),
                "document_name":    doc.metadata.get("document_name"),
                "ai_enhanced":      doc.metadata.get("ai_enhanced"),                          
                "raw_text":         doc.metadata.get("raw_text"),                           
                "tables_html":      doc.metadata.get("tables_html"),                           
                "images_base64":    doc.metadata.get("images_base64"),                           
                
            }
            for doc in chunks
        ]
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logger.error(f"Failed to write chunks to '{output_path}': {e}")
            raise ExportError(f"Failed to write chunks to '{output_path}': {e}") from e

        logger.info("Exported %d chunks to %s", len(export_data), output_path)