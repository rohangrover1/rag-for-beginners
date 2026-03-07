import os
import json
from store_documents import update_databases
from retrieve_documents import HybridRetriever
from pdf_partitioner import PDFPartitioner
import logging
from vector_store_manager import VectorStoreManager
from keyword_store_manager import KeywordStoreManager
from langchain_core.documents import Document

def load_docs_from_json(json_path: str) -> list:
    """Load documents from a JSON file and convert them to LangChain Document objects."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            doc = Document(
                page_content=item.get("enhanced_content", ""),
                metadata={
                    "doc_id": item.get("doc_id"),
                    "chunk_index": item.get("chunk_index"),
                    "document_name": item.get("document_name"),
                    "ai_enhanced": item.get("ai_enhanced"),
                    "raw_text": item.get("raw_text"),
                    "tables_html": item.get("tables_html"),
                    "images_base64": item.get("images_base64"),
                },
                id=item.get("doc_id")
            )
            documents.append(doc)
        
        return documents
    except Exception as e:
        logger.error(f"❌ Failed to load documents from JSON: {e}")
        return []


def setup_logger(name: str, log_file: str = "log.txt", level: int = logging.DEBUG) -> logging.Logger:
    """Configure a logger that writes to both the console and a log file."""
    
    logger = logging.getLogger(name)
    logger.setLevel(level)  # Master level — handlers can narrow this further

    # Shared formatter for both handlers
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(filename)30s:%(lineno)5d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Console handler --- prints to screen
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)   # Only INFO and above on screen
    console_handler.setFormatter(formatter)

    # --- File handler --- writes to log.txt
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)     # Everything including DEBUG to file
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def read_config(config_path: str) -> dict:
    """Load configuration from a JSON file."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"❌ Failed to read config file: {e}")
        return {}

if __name__ == "__main__":
    try:

        logger = setup_logger("default_logger")
        
        # load the config file
        config = read_config("rag/rag_config.json")
        delete_documents_before_ingest = config.get("delete_documents_before_ingest", False)  # default to False if not specified
        ai_summarization_model = config.get("ai_summarization_model", "gpt-4o")  # default to gpt-4o if not specified
        persist_directory = config.get("chomradb", "db/chroma_db")  # default to db/chroma_db if not specified
        bm25_index_name = config.get("bm25_index_name", "langchain_bm25")  # default to langchain_bm25 if not specified
        embedding_model_name = config.get("embedding_model", "text-embedding-3-small")  # default to text-embedding-3-small if not specified
        num_retrieved_docs = config.get("num_retrieved_docs", 3)  # default to 3 if not specified
        num_reformulated_queries = config.get("num_reformulated_queries", 2)  # default to 2 if not specified
        multi_query_reformulation = config.get("multi_query_reformulation", True)  # default to True if not specified

        # Test with your PDF file
        #file_path = "./docs/attention-is-all-you-need.pdf"  # Change this to your PDF path
        file_path = "./docs/Federated Wireless - Private Wireless.pdf"  # Change this to your PDF path 
        #file_path = "./docs/Berkeley ExecEd AI for Executives Program Guide.pdf"  # Change this to your PDF path 
        document_name = file_path.split("/")[-1]   
        
        # # paritions the document and creates langchain documents with enhanced summaries and rich metadata
        # try:
        #     partitioner = PDFPartitioner(document_name=document_name, llm_model=ai_summarization_model)
        #     docs = partitioner.partition_document(file_path)            # get the langchain documents with enhanced summaries and rich metadata
        # except Exception as e:
        #     logger.error(f"Error during document partitioning: {e}")
        #     docs = None
        
        # # dump the processed documents to a json file for inspection
        # if docs:
        #     PDFPartitioner.export_chunks_to_json(docs, "processed_chunks.json") # dump the processed documents to a json file for inspection

        docs = load_docs_from_json("processed_chunks.json")  # Load processed documents from JSON file
        logger.info(f"Loaded {len(docs)} documents from JSON file.")

        # # store the processed documents in the vector store and keyword storea
        if docs:
        #     # update the vector store
        #     # vector_store_manager = VectorStoreManager(persist_directory=persist_directory, embedding_model=embedding_model_name)
        #     # if delete_documents_before_ingest:
        #     #     vector_store_manager.safe_bulk_delete_all_vectors()  # Clear existing collection before ingesting new documents
        #     # vector_store_manager.update_vector_store(documents=docs, document_name=document_name)

            # update the keyword store
            keyword_store_manager = KeywordStoreManager(index_name=bm25_index_name)
            if delete_documents_before_ingest:
                keyword_store_manager._delete_document_chunks(document_name=document_name)  # Clear existing index before ingesting new documents
            keyword_store_manager.update_keyword_store(document_name=document_name, chunks=docs)



        # # retrieve documents for given query and print results
        # query = "What is the Certificate of Business Excellence"

        # results = HybridRetriever(query, persist_directory, embedding_model_name, bm25_index_name, 
        #                           num_retrieved_docs, num_reformulated_queries, multi_query_reformulation)
        # # bm25_retriever = BM25Retriever(host="localhost", port=9200, index_name=bm25_index_name, k=num_retrieved_docs)
        # # sparse_results = bm25_retriever.invoke(query)
        # print(f"Retrieved {len(results)} documents for query: '{query}'")
        # for i, doc in enumerate(results):
        #     print(f"Document {i+1}:")
        #     print(f"Text: {doc.page_content}")
            
            
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
