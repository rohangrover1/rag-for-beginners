import os
import json
from store_documents import update_databases
from retrieve_documents import HybridRetriever


if __name__ == "__main__":
    try:
        
        # load the config file
        with open("rohan_experiments/rag_config.json", "r") as f:
            config = json.load(f)
        persist_directory = config.get("chomradb", "db/chroma_db")  # default to db/chroma_db if not specified
        bm25_index_name = config.get("bm25_index_name", "langchain_bm25")  # default to langchain_bm25 if not specified
        embedding_model_name = config.get("embedding_model", "text-embedding-3-small")  # default to text-embedding-3-small if not specified
        num_retrieved_docs = config.get("num_retrieved_docs", 3)  # default to 3 if not specified
        num_reformulated_queries = config.get("num_reformulated_queries", 2)  # default to 2 if not specified
        multi_query_reformulation = config.get("multi_query_reformulation", True)  # default to True if not specified

        # Test with your PDF file
        #file_path = "./docs/attention-is-all-you-need.pdf"  # Change this to your PDF path
        #file_path = "./docs/Federated Wireless - Private Wireless.pdf"  # Change this to your PDF path 
        file_path = "./docs/Berkeley ExecEd AI for Executives Program Guide.pdf"  # Change this to your PDF path 
        
        # add document to vector store and keyword store
        # update_databases(file_path, persist_directory, bm25_index_name)

        # retrieve documents for given query and print results
        query = "What is the Certificate of Business Excellence"
        results = HybridRetriever(query, persist_directory, embedding_model_name, bm25_index_name, 
                                  num_retrieved_docs, num_reformulated_queries, multi_query_reformulation)
        # bm25_retriever = BM25Retriever(host="localhost", port=9200, index_name=bm25_index_name, k=num_retrieved_docs)
        # sparse_results = bm25_retriever.invoke(query)
        print(f"Retrieved {len(results)} documents for query: '{query}'")
        for i, doc in enumerate(results):
            print(f"Document {i+1}:")
            print(f"Text: {doc.page_content}")
            
            
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
