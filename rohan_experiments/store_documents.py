from file_parsing import partition_document, create_chunks_by_title, summarise_chunks, export_chunks_to_json
from vector_store import update_vector_store, number_of_documents_in_vector_store, get_all_document_ids, safe_bulk_delete_all_vectors
from keyword_store import update_keyword_store, number_of_documents_in_keyword_store
import os
from langchain_core.documents import Document
from typing import List

def chunk_file(file_path: str, persist_directory: str) -> List[Document]:
    """Process a document into chunks ready for vector or keyword store ingestion"""
    try:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None

        document_name = file_path.split("/")[-1]           
        elements = partition_document(file_path)
        print(f"number of elements {len(elements)} in document {document_name}")
        if not elements:
            print("No elements extracted, exiting.")
            return None
        
        # Create chunks
        chunks = create_chunks_by_title(elements)
        print(f"number of chunks {len(chunks)}")

        # Summarize chunks with AI and create LangChain Documents with rich metadata
        processed_chunks = summarise_chunks(chunks, document_name)   
        return processed_chunks
    except Exception as e:
        print(f"❌ An error occurred: {e}")

def update_databases(file_path: str, persist_directory: str, bm25_index_name: str):
    """Process document and update vector store and opensearch keyword store"""
    try:
        document_name = file_path.split("/")[-1]   

        print(f"=== Processing document: {document_name} ===")
        chunks = chunk_file(file_path, persist_directory)
        if chunks is None:
            print("Failed to process document, skipping database update.")
            return None
        
        # Export processed chunks to JSON for inspection
        # export_chunks_to_json(chunks)
    
        # doc_ids = get_all_document_ids(persist_directory)
        # print(f"Existing document IDs in vector store: {doc_ids}")
        # bulk delete existing vectors for this document
        # print(f"--- Deleting existing vectors for document: {document_name} ---")
        # safe_bulk_delete_all_vectors(persist_directory)
        
        # Update the vector store
        print(f"--- Updating vector store for document: {document_name} ---")
        update_vector_store(chunks, persist_directory, document_name)
        number_of_documents_in_vector_store(persist_directory)
        
        # update the keyword store
        print(f"--- Updating keyword store for document: {document_name} ---")
        status = update_keyword_store(document_name, chunks, bm25_index_name) 
        if status:
            print(f"✅ Successfully updated keyword store for document: {document_name}")
        number_of_documents_in_keyword_store(bm25_index_name)
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return None
