import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


_ = load_dotenv(find_dotenv())

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

def calculate_similarity_scores(query: str, retrieved_docs):
    """Calculate similarity scores between query and retrieved documents"""
    try:
        print("📊 Calculating similarity scores...")
        
        # Get embedding for the query
        query_embedding = embedding_model.embed_query(query)
        query_vector = np.array([query_embedding])
        
        results = []
        
        # Calculate similarity for each retrieved document
        for i, doc in enumerate(retrieved_docs):
            # Get embedding for the document
            doc_embedding = embedding_model.embed_query(doc.page_content)
            doc_vector = np.array([doc_embedding])
            
            # Calculate cosine similarity (0 to 1 scale)
            similarity_score = cosine_similarity(query_vector, doc_vector)[0][0]
            
            results.append({
                "doc_index": i,
                "similarity_score": float(similarity_score),
                "content_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                "metadata": doc.metadata
            })
        
        # Sort by similarity score descending
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        print(f"✅ Calculated similarity scores for {len(results)} documents")
        return results
        
    except Exception as e:
        print(f"❌ Error calculating similarity: {e}")
        return []

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
#query = "How much did Microsoft pay to acquire GitHub?"
query = "When did Larry Elison start Oracle"

#retriever = db.as_retriever(search_kwargs={"k": 5})
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )
# relevant_docs = retriever.invoke(query)
# print(len(relevant_docs))
# print(type(relevant_docs[0]))

# similarity_results = calculate_similarity_scores(query, relevant_docs)
# for result in similarity_results:
#     print(f"Doc Index: {result['doc_index']}")
#     print(f"Similarity Score: {result['similarity_score']}")
#     print(f"Content Preview: {result['content_preview']}")
#     print(f"Metadata: {result['metadata']}\n")

# get the scores
docs_and_scores = db.similarity_search_with_score(query, k=5)
doc_num = 1
relevant_docs = []
for doc, score in docs_and_scores:
    relevant_docs.append(doc)   
    print(f"Document {doc_num}")
    print(f"Score: {score}")
    print(doc.metadata)
    print(f"{doc.page_content[:120]}\n")
    doc_num += 1

similarity_results = calculate_similarity_scores(query, relevant_docs)
for result in similarity_results:
    print(f"Doc Index: {result['doc_index']}")
    print(f"Similarity Score: {result['similarity_score']}")
    print(f"Content Preview: {result['content_preview']}")
    print(f"Metadata: {result['metadata']}\n")


# print(f"User Query: {query}")
# # Display results
# print("--- Context ---")
# for i, doc in enumerate(relevant_docs, 1):    
#     print(f"Document {i}: Metadata {doc.metadata}\n")
#     print(f"{doc.page_content}\n")



# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"


# code to combine the docs
# '''
# Using the str.join() method: This is a string method used to concatenate a sequence of strings 
# (like a list of strings) into a single string, using a specified separator.
# '''
# docsx = {chr(10).join([f"- {doc.page_content[:50]}" for doc in relevant_docs])} # chr(10)= "\n"
# print(docsx)
# for i, doc in enumerate(relevant_docs):
#     print(f"doc{i}:{doc.page_content[:50]}")
#     print(chr(10))



# call the LLM
# Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question: {query}
Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}
Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)