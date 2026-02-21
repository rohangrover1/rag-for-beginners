from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

def document_exists(vectorstore, document_name: str) -> bool:
    """
    Returns True if any vector exists for the given document_name.
    Fast and safe for large collections.
    """
    collection = vectorstore._collection

    result = collection.get(
        where={"document_name": document_name},
        limit=1,
        include=[]  # IDs only, no embeddings/docs
    )

    return len(result.get("ids", [])) > 0

def is_semantic_duplicate(
    vectorstore,
    doc_name: str,
    text: str,
    threshold: float = 0.98,
    k: int = 3,
):
    """
    Returns True if a near-identical semantic duplicate exists.
    """
    results = vectorstore.similarity_search_with_score(text, 
                                                       k=k,
                                                       filter={"document_name": doc_name})
    similarity = 0.0
    for _, score in results:
        # Chroma returns distance; cosine distance → similarity = 1 - distance
        similarity = 1 - score
        if similarity >= threshold:
            return True, similarity

    return False, similarity

def get_all_document_ids(persist_directory: str):
    """
    Return all document IDs stored in the Chroma collection.
    """
    vectorstore = load_vector_store(persist_directory)
    result = vectorstore._collection.get(
        include=[]  # IDs are always returned; this avoids loading embeddings/docs
    )
    return result["ids"]

def extract_chunk_ids(documents):
    """
    Extract stable Chroma document IDs from LangChain Documents.

    Requirements:
    - doc.id MUST exist
    - IDs must be deterministic across re-indexes
    - Returned IDs must be strings
    """
    ids = []

    for i, doc in enumerate(documents):
        if not hasattr(doc, "id") or doc.id is None:
            raise ValueError(
                f"Document at index {i} is missing 'id'. "
                "doc.id is required for Chroma upserts."
            )

        ids.append(str(doc.id))

    return ids


def update_vector_store(documents, persist_directory: str, document_name: str):
    """
    Create or update a ChromaDB vector store.
    Documents with the same chunk_id (doc.id) will overwrite existing vectors.
    """
    try:
        print("🔮 Creating embeddings and storing in ChromaDB...")

        if os.path.exists(persist_directory):
            # vector store exists, load and update
            print(f"--- Loading existing vector store from {persist_directory} ---")
            vectorstore = load_vector_store(persist_directory)
            
            # check if document already exists in vector store
            if document_exists(vectorstore, document_name):
                print("🚫 Document already indexed in vector store — skipping ingest")
                return True
        

            print(f"--- Adding {len(documents)} new documents to vector store ---")
            print(f"--- Upserting {len(documents)} documents (overwrite by ID) ---")
            
            # add only those documents which are not semantic duplicates
            non_duplicate_docs = []
            for doc in documents:
                status, simularity = is_semantic_duplicate(vectorstore, document_name, doc.page_content)
                if status:
                    print(f"⚠️ Skipping semantic duplicate chunk: {doc.id}. Found existing chunk with similarity {simularity:.2f}")
                else:
                    print(f"✅ Adding non-duplicate chunk: {doc.id} with similarity {simularity:.2f}")
                    non_duplicate_docs.append(doc)
            
            ids = extract_chunk_ids(non_duplicate_docs)
            vectorstore.add_documents(
                documents=non_duplicate_docs,
                ids=ids,               # ✅ THIS ENABLES OVERWRITE
            )
            print(f"--- Finished updating vector store ---")
        else:
            # Create ChromaDB vector store
            print("--- Creating vector store ---")
            ids = extract_chunk_ids(documents)
            vectorstore = Chroma.from_documents(
                documents=documents,
                ids=ids,               # ✅ IDs on initial create
                embedding=embedding_model,
                persist_directory=persist_directory, 
                collection_metadata={"hnsw:space": "cosine"}
            )
            print("--- Finished creating vector store ---")
            print(f"✅ Vector store created and saved to {persist_directory}")
        return True
    except Exception as e:
        print(f"❌ An error occurred while updating vector store: {e}")
        return None

def load_vector_store(persist_directory: str):
    """Load existing ChromaDB vector store"""
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}  
    )
    print(f"✅ Vector store loaded successfully with {vectorstore._collection.count()} documents")    
    return vectorstore

def number_of_documents_in_vector_store(persist_directory: str):
    """Get the number of documents in the ChromaDB vector store"""
    try:
        vectorstore = load_vector_store(persist_directory)
        count = vectorstore._collection.count()
        print(f"📊 Number of documents in vector store: {count}")
        return count
    except Exception as e:
        print(f"❌ An error occurred while counting documents in vector store: {e}")
        return None

def safe_bulk_delete_all_vectors(persist_directory: str, batch_size: int = 1000):
    """
    Safely delete ALL vectors from a Chroma collection in batches.

    - Preserves the collection
    - Avoids loading embeddings/documents
    - Safe for large collections
    """

    vectorstore = load_vector_store(persist_directory)
    collection = vectorstore._collection

    total = collection.count()
    print(f"🧹 Starting bulk delete of {total} vectors")

    if total == 0:
        print("✅ Collection already empty")
        return

    deleted = 0
    while True:
        result = collection.get(
            limit=batch_size,
            include=[]  # IDs only (fast + safe)
        )
        ids = result.get("ids", [])
        if not ids:
            break

        collection.delete(ids=ids)
        deleted += len(ids)

        print(f"🗑️ Deleted {deleted}/{total}")

    remaining = collection.count()
    if remaining != 0:
        raise RuntimeError(
            f"❌ Bulk delete incomplete: {remaining} vectors still remain"
        )
    print("✅ Bulk delete completed successfully")