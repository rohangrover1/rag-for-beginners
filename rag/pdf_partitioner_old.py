from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv, find_dotenv
import json
import hashlib
import re
import unicodedata


_ = load_dotenv(find_dotenv())


class PDFPartitioner:
    """
    Processes PDF documents into AI-enhanced, searchable chunks.

    Entry point: `partition_document(file_path)`

    Example usage:
        partitioner = PDFPartitioner(document_name="my_report")
        docs = partitioner.partition_document("path/to/file.pdf")
        partitioner.export_chunks_to_json(docs)
    """

    def __init__(
        self,
        document_name: str = "document",
        max_characters: int = 3000,
        new_after_n_chars: int = 2400,
        combine_text_under_n_chars: int = 500,
        llm_model: str = "gpt-4o",
    ):
        self.document_name = document_name
        self.max_characters = max_characters
        self.new_after_n_chars = new_after_n_chars
        self.combine_text_under_n_chars = combine_text_under_n_chars
        self.llm_model = llm_model

    # ------------------------------------------------------------------ #
    #  Entry point                                                         #
    # ------------------------------------------------------------------ #

    def partition_pdf_document(self, file_path: str, chunk_index: int = 0) -> List[Document]:
        """
        Full pipeline: extract → chunk → summarise.

        Args:
            file_path:    Path to the PDF file.
            chunk_index:  Starting index offset for chunk metadata.

        Returns:
            List of LangChain Documents ready for a vector store.
        """
        elements = self._extract_elements(file_path)
        if not elements:
            return []

        chunks = self._create_chunks(elements)
        documents = self._summarise_chunks(chunks, chunk_index=chunk_index)
        return documents

    # ------------------------------------------------------------------ #
    #  Pipeline steps (private)                                           #
    # ------------------------------------------------------------------ #

    def _extract_elements(self, file_path: str):
        """Extract raw elements from PDF using unstructured."""
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            print(f"❌ File not found: {file_path}")
            return []

        try:
            print(f"📄 Partitioning document: {file_path}")
            elements = partition_pdf(
                filename=file_path,
                strategy="hi_res",
                infer_table_structure=True,
                extract_image_block_types=["Image"],
                extract_image_block_to_payload=True,
                languages=["English"],
            )
            print(f"✅ Extracted {len(elements)} elements")
            return elements

        except Exception as e:
            print(f"❌ Error processing PDF: {e}")
            return []

    def _create_chunks(self, elements):
        """Create intelligent, title-based chunks from elements."""
        print("🔨 Creating smart chunks...")
        chunks = chunk_by_title(
            elements,
            max_characters=self.max_characters,
            new_after_n_chars=self.new_after_n_chars,
            combine_text_under_n_chars=self.combine_text_under_n_chars,
        )
        print(f"✅ Created {len(chunks)} chunks")
        return chunks

    def _summarise_chunks(self, chunks, chunk_index: int = 0) -> List[Document]:
        """Process all chunks, applying AI summaries where needed."""
        print("🧠 Processing chunks with AI Summaries...")

        documents = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            print(f"   Processing chunk {i + 1}/{total}")

            content_data = self._separate_content_types(chunk)

            if content_data["tables"] or content_data["images"]:
                print("     → Creating AI summary for mixed content...")
                try:
                    page_content = self._create_ai_enhanced_summary(
                        content_data["text"],
                        content_data["tables"],
                        content_data["images"],
                    )
                    ai_enhanced = True
                except Exception as e:
                    print(f"     ❌ AI summary failed: {e}")
                    page_content = content_data["text"]
                    ai_enhanced = False
            else:
                print("     → Using raw text (no tables/images)")
                page_content = content_data["text"]
                ai_enhanced = False

            page_content = self._canonicalize_text(page_content)

            doc = Document(
                page_content=page_content,
                metadata={
                    "original_content": json.dumps(
                        {
                            "raw_text": content_data["text"],
                            "tables_html": content_data["tables"],
                            "images_base64": content_data["images"],
                        }
                    ),
                    "chunk_index": chunk_index + i,
                    "document_name": self.document_name,
                    "ai_enhanced": ai_enhanced,
                },
                id=self._chunk_id(page_content),
            )
            documents.append(doc)

        print(f"✅ Processed {len(documents)} chunks")
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
        content_data = {
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
        """Create an AI-enhanced, searchable summary for mixed content."""
        try:
            llm = ChatOpenAI(model=self.llm_model, temperature=0)

            prompt_text = f"""You are creating a searchable description for document content retrieval.
                CONTENT TO ANALYZE:
                TEXT CONTENT:
                {text}
                """
            if tables:
                prompt_text += "TABLES:\n"
                for i, table in enumerate(tables):
                    prompt_text += f"Table {i + 1}:\n{table}\n\n"

            prompt_text += """
                YOUR TASK:
                Generate a comprehensive, searchable description that covers:

                1. Key facts, numbers, and data points from text and tables
                2. Main topics and concepts discussed
                3. Questions this content could answer
                4. Visual content analysis (charts, diagrams, patterns in images)
                5. Alternative search terms users might use

                Make it detailed and searchable - prioritize findability over brevity.

                SEARCHABLE DESCRIPTION:"""

            message_content = [{"type": "text", "text": prompt_text}]

            for image_base64 in images:
                message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    }
                )

            response = llm.invoke([HumanMessage(content=message_content)])
            return response.content

        except Exception as e:
            print(f"     ❌ AI summary failed: {e}")
            summary = f"{text[:300]}..."
            if tables:
                summary += f" [Contains {len(tables)} table(s)]"
            if images:
                summary += f" [Contains {len(images)} image(s)]"
            return summary

    def _chunk_id(self, chunk_text: str) -> str:
        """Generate a unique chunk ID using document name and a hash of the text."""
        h = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        return f"{self.document_name}:{h}"

    # ------------------------------------------------------------------ #
    #  Export                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def export_chunks_to_json(chunks: List[Document], output_path: str = "processed_chunks.json"):
        """Export processed chunks to a clean JSON file for inspection."""
        export_data = []

        for doc in chunks:
            chunk_data = {
                "chunk_id": doc.id,
                "enhanced_content": doc.page_content,
                "metadata": {
                    "original_content": json.loads(
                        doc.metadata.get("original_content", "{}")
                    ),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "document_name": doc.metadata.get("document_name"),
                },
            }
            export_data.append(chunk_data)

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"✅ Exported {len(export_data)} chunks to {output_path}")