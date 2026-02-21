import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "default"


def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
   
    # for i, doc in enumerate(documents[:2]):  # Show first 2 documents
    #     print(f"\nDocument {i+1}:")
    #     print(f"  Source: {doc.metadata['source']}")
    #     print(f"  Content length: {len(doc.page_content)} characters")
    #     print(f"  Content preview: {doc.page_content[:100]}...")
    #     print(f"  metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,  
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
        
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():
    """Main ingestion pipeline"""
    print("=== RAG Document Ingestion Pipeline ===\n")
    
    # Define paths
    docs_path = "docs"
    persistent_directory = "dbsmall1/chroma_db"
    
    # Check if vector store already exists
    if os.path.exists(persistent_directory):
        print("✅ Vector store already exists. No need to re-process documents.")
        
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model, 
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Loaded existing vector store with {vectorstore._collection.count()} documents")
        return vectorstore
    
    print("Persistent directory does not exist. Initializing vector store...\n")
    
    # Step 1: Load documents
    documents = load_documents(docs_path)[0:1]    # just load one document for now
    #documents = load_documents(docs_path) 
    print(len(documents))
    print(type(documents[0]))
    
    # Step 2: Split into chunks
    chunks = split_documents(documents)
    print(len(chunks))
    print(type(chunks[0]))
    
    # Step 3: Create vector store
    vectorstore = create_vector_store(chunks, persistent_directory)
    
    print("\n✅ Ingestion complete! Your documents are now ready for RAG queries.")
    return vectorstore


if __name__ == "__main__":
    main()
