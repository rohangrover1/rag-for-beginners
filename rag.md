
- [RAG pipleline](#rag-pipleline)
- [Different embedding models](#different-embedding-models)
  - [OpenAI](#openai)
  - [Others](#others)
- [Document loaders](#document-loaders)
- [Vector DB](#vector-db)
- [Langchain Document classes](#langchain-document-classes)
  - [Document Loader](#document-loader)
- [Chunking](#chunking)
  - [Text splitter](#text-splitter)
  - [Chunking strategies](#chunking-strategies)
    - [Character Text Splitter](#character-text-splitter)
    - [Recursive Character Text Splitter](#recursive-character-text-splitter)
    - [Document-Specific Splitting](#document-specific-splitting)
    - [Semantic Splitting](#semantic-splitting)
      - [Here's a summary of semantic chunking:](#heres-a-summary-of-semantic-chunking)
    - [Agentic Splitting (AI-Powered Chunking)](#agentic-splitting-ai-powered-chunking)
- [Vector stores](#vector-stores)
  - [Chroma vector store](#chroma-vector-store)
    - [creating the vector store](#creating-the-vector-store)
    - [preventing overwriting of the same chunk](#preventing-overwriting-of-the-same-chunk)
    - [Collections in vector store](#collections-in-vector-store)
    - [to get the collection names](#to-get-the-collection-names)
    - [Collection document count](#collection-document-count)
    - [Deleting the documents in the vector store](#deleting-the-documents-in-the-vector-store)
      - [Soft reset](#soft-reset)
      - [Deleting ny document name or multiple conditions](#deleting-ny-document-name-or-multiple-conditions)
      - [Hard reset](#hard-reset)
    - [retreiving the vector store](#retreiving-the-vector-store)
- [Retriever](#retriever)
  - [Chroma DB retriever](#chroma-db-retriever)
  - [Retreiving directly with similarity search](#retreiving-directly-with-similarity-search)
  - [cosine similarity](#cosine-similarity)
  - [Two stage retreival](#two-stage-retreival)
  - [different retrieval methods](#different-retrieval-methods)
    - [Basic retreival](#basic-retreival)
    - [Retreival with threshold](#retreival-with-threshold)
    - [Maximum Marginal Relevance](#maximum-marginal-relevance)
    - [Summary](#summary)
- [Tips](#tips)
  - [Don't use similarity\_search\_with\_score](#dont-use-similarity_search_with_score)
  - [Joining the chunks for the LLM](#joining-the-chunks-for-the-llm)
  - [Multi query retreival](#multi-query-retreival)
  - [Reciprocal Rank Fusion](#reciprocal-rank-fusion)
    - [Summary](#summary-1)
- [unstructured library for multi-modal document extraction](#unstructured-library-for-multi-modal-document-extraction)
  - [Extraction](#extraction)
    - [attributes of the element dict](#attributes-of-the-element-dict)
      - [general attributes](#general-attributes)
      - [Images](#images)
      - [Tables](#tables)
  - [Chunking](#chunking-1)
    - [attributes of the chunks dict](#attributes-of-the-chunks-dict)
- [Hybrid search](#hybrid-search)
  - [Summary](#summary-2)
    - [Vector Search:](#vector-search)
    - [Keyword Search:](#keyword-search)
    - [Why Hybrid Search?](#why-hybrid-search)
  - [Hybrid search commands](#hybrid-search-commands)
  - [RAG at scale](#rag-at-scale)
    - [Challenges with Scaling RAG and Large Documents:](#challenges-with-scaling-rag-and-large-documents)
    - [Tips for Working with Large Numbers of Documents (Solutions for Scaling):](#tips-for-working-with-large-numbers-of-documents-solutions-for-scaling)
    - [Steps for advanced RAG with large number of documents](#steps-for-advanced-rag-with-large-number-of-documents)
- [Rerankers](#rerankers)
  - [Summary](#summary-3)
  - [Algorithm](#algorithm)
- [OpenSearch](#opensearch)
  - [Installing and running OpenSearch on MAC](#installing-and-running-opensearch-on-mac)
    - [Option 1: Install OpenSearch with Homebrew](#option-1-install-opensearch-with-homebrew)
      - [Install](#install)
  - [Indexing with Opensearch](#indexing-with-opensearch)
  - [Retreiving with OpenSearch](#retreiving-with-opensearch)

# RAG pipleline
![RAG piplepile](images/rag_pipeline.png)


# Different embedding models
## OpenAI
- Most popular choice
-  openAI provides uh two different models which are very popular. We've got the 
   -  text-embedding-3-small: Default is 1536 dimensions. It is great for most use cases but we can actually reduce it to 512 dimensions, 1024 dimensions or anything<=1536.
   -  text-embedding-3-large: Default is 3072 dimensions. Greate performance but costs more. Can be reduced to smaller dimensions, anything<=3072
   -  Both models allow reduction in dimensions without losing much quality

## Others
- Cohere: Strong multilingual support
- Voyage AI:
- Mistral

# Document loaders
- Langchain has document loaders that loads documents from folder
```python
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )    
    documents = loader.load()
```
- `documents` is a list where each member is an object of class `langchain_core.documents.base.Document'`
- `TextLoader` is the class. Other documents can also be loaded.

# Vector DB
Different vector DBs
- Specialized vector DBs - Pincecone, Weaviate, ChromaDB, FAISS
- Regular SQL DBs: offers structured storage and retreival

*Consitency in Ingention and Retreival*
- You MUST use the same model for documents and user queries
- Even with the same model you must use exact dimensions in all systems

# Langchain Document classes

## Document Loader
- Used to load files
```python
from langchain_community.document_loaders import TextLoader, DirectoryLoader
# Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
```
- `DirectoryLoader` inputs are
  - path: pth of the files
  - glob: type of files to load
  - loader_cls: loader class and many supported such as text, pdf, csv [https://docs.langchain.com/oss/python/integrations/document_loaders] 

- Once the files are loaded into documents they have the following attributes
```python
for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

```
- Attributes
  - doc.metadata: file metadata
  - doc.page_content: large vector with the contents
  
# Chunking
- Breaks large file into characters (not tokens)
 
## Text splitter
- Most basic text splitting class in langchain
```python
from langchain_text_splitters import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,  
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
```
- Attributes:
  - Inputs:
    - chunk_size: size in chars
    - chunk_overlap: chunk overlap
  - Outputs:
    - Outputs the chunks 
    - Each chunk is an object of class `<class 'langchain_core.documents.base.Document'>`
- Once the data are loaded into chunks they have the following attributes
```python
for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
```
- Attributes
  - chunk.metadata: file metadata
  - chunk.page_content: vector with the contents

## Chunking strategies

### Character Text Splitter
- This is the most basic method, cutting text at fixed character counts.
- You can use custom separators.
- It's useful for simple, uniform documents or when speed is a priority.
- Default split, called piece, is at `\n\n` which can be changed. 
- Each piece is then joined until chunk size
- Can take only one splitter 

### Recursive Character Text Splitter
- An upgrade from the basic character splitter.
- It tries to split at natural boundaries like paragraphs, sentences, or words.
- It falls back gracefully if chunks become too large.
- It preserves more context than basic splitting.
- Can take a list of splitters to try them until in order 
  - `separators=["\n\n", "\n", ". ", " ", ""],  # Multiple separators` 

### Document-Specific Splitting
- This method respects the document's inherent structure.
- It has awareness of pages, sections, and headers for PDFs.
- Each document type (like Markdown, CSV, DOCX, PPT) gets appropriate treatment.

### Semantic Splitting
- Uses embeddings to detect topic shifts.
- It keeps related concepts together.
- Splits when the meaning changes, not just based on size.
- More intelligent, but computationally very expensive.

#### Here's a summary of semantic chunking:
- Purpose: Semantic chunking breaks long documents into meaningful pieces by identifying where topics naturally change, rather than using fixed word counts (2:31-2:36).
- Method: It uses AI embeddings to understand the semantic meaning of sentences. If the topic shifts significantly between sentences, a split is made (2:39-2:52).
- Process:
  - Each sentence is converted into numerical vectors or embeddings (2:57-3:02).
  - Similarity scores are calculated between nearby sentences using these embeddings (3:04-3:10).
  - Boundaries are created where the similarity score drops significantly (3:13-3:17, 3:31-3:38).
  - Breakpoint Criteria: The most common criterion for splitting is using percentiles. This helps adapt to different types of documents with varying similarity patterns (5:01-5:07, 11:11-11:14). For example, a 70th percentile threshold means that a split occurs when the similarity score is in the lowest 30% of all similarity scores for that document (11:18-11:35).
  - Practicality: The video creator notes that this method is often not economical for large-scale production use due to the cost of using embedding models (2:04-2:09, 12:15-12:28).


### Agentic Splitting (AI-Powered Chunking)
- An LLM (Large Language Model) analyzes content and decides optimal splits.
- It can understand complex relationships.
- Adapts to content type automatically.
- The most sophisticated but also the slowest and most expensive because an LLM is involved.

# Vector stores

## Chroma vector store

### creating the vector store
- Create a vector store using chroma DB
- Can be created locally
```python
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vectorstore

```
- Need to provide the embedding models and using the openAI small model which has 1536 tokens
- Attributes
  - documents: the chunks we created with 1000 odd chars. Objects of class `<class 'langchain_core.documents.base.Document'>`
  - embedding: embedding model
  - persist_directory: local directory where vector data is stored
  - collection_metadata: the algorithm, which is cosine similarity in this case
  
*NOTE:* If you keep writing the same document to the vector store it keeps making ocopies of the same document so be careful to write only once to the vector store

### preventing overwriting of the same chunk
- add a `id` attribute to the chunk
- Same chunk → same ID → overwritten, not duplicated
```python
import hashlib
from langchain.schema import Document

def chunk_id(document_name: str, chunk_text: str) -> str:
    h = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
    return f"{document_name}:{h}"

docs = []
for chunk in chunks:
    docs.append(
        Document(
            page_content=chunk,
            metadata={"document_name": "Design_Spec_v1"},
            id=chunk_id("Design_Spec_v1", chunk)
        )
    )

vectorstore.add_documents(docs)
```

### Collections in vector store 
- Each vector store can have `collections` like MongoDB
- The db objects for each collection are different but they have the same directory
```python
rom langchain_chroma import Chroma

# Collection 1
db1 = Chroma(collection_name="collection_one", persist_directory="./chroma_db", embedding_function=embedding)

# Collection 2
db2 = Chroma(collection_name="collection_two", persist_directory="./chroma_db", embedding_function=embedding)
```

### to get the collection names
```python
collection_name = db._collection.name
print(f"Collection name: {collection_name}")
```

### Collection document count
```python
count = vector_store._collection.count()
print(f"Number of vectors: {count}")
```


### Deleting the documents in the vector store
#### Soft reset
```python
vectorstore.reset_collection()   
```

- Collection, embeddings config, and persistence remain intact
- This is the safest “truncate” operation

#### Deleting ny document name or multiple conditions
```python
vectorstore.delete(
    where={"document_name": "Design_Spec_v1"}
)
```
- `document_name` should be part of meta data

```python
vectorstore.delete(
    where={
        "$and": [
            {"document_name": "Design_Spec_v1"},
            {"chunk_index": {"$gte": 5}}
        ]
    }
)
```
- `document_name` and `chunk_index` should be part of meta data

#### Hard reset
- Need to find collection names and then delete the full collection
```python
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
import shutil
import os

# Configuration
PERSIST_DIR = "./chroma_store"
COLLECTION_NAME = "documents"

# Embeddings (must match what you'll re-ingest with)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# 1. Initialize Chroma client
client = chromadb.Client(
    settings=chromadb.Settings(
        persist_directory=PERSIST_DIR
    )
)

# 2. Delete collection if it exists
existing_collections = [c.name for c in client.list_collections()]
if COLLECTION_NAME in existing_collections:
    client.delete_collection(name=COLLECTION_NAME)

# 3. (Optional but recommended) Remove persisted files
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)

# 4. Recreate LangChain Chroma vector store
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR
)

print("Chroma hard reset complete.")
```
- where={} matches all documents
- Collection, embeddings config, and persistence remain intact
- This is the safest “truncate” operation


### retreiving the vector store
- Create a vector store using chroma DB
```python
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model, 
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Loaded existing vector store with {vectorstore._collection.count()} documents")
        return vectorstore
```
- Almost the same formar as the creating the vector store
- No `chunks` input is needed
- To see the length of the vector store use
'''python
vectorstore._collection.count()
'''

# Retriever
## Chroma DB retriever
- Creating the retriever
```python
# simple retriever that gets first 5 documents
retriever = db.as_retriever(search_kwargs={"k": 5})

# More complex retriever that gets 5 documents but has a score threshold
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
    }
)
```
- Making the query 
```python
query = "How much did Microsoft pay to acquire GitHub?"
relevant_docs = retriever.invoke(query)
```
- The `relevant_docs` is a list of objects of document class `langchain_core.documents.base.Document`



## Retreiving directly with similarity search
- Get the chunks directly instead of using the `similarity_search_with_score` retriever class
```python
docs_and_scores = db.similarity_search_with_score(query, k=5)
doc_num = 1
for doc, score in docs_and_scores:
    print(f"Document {doc_num}")
    print(f"Score: {score}")
    print(doc.metadata)
    print(f"{doc.page_content}\n")
    doc_num += 1
```
- Provides the score

## cosine similarity
- Cos of the angle between two vector a,b 
- Using the dot product formula for vector
- Formula
  - $a.b = ||a||.||b|| cos(\theta)$
  - $cos(\theta) = a.b / (||a||.||b||)$
- The vectors in these DB are normalized to 1 so magnitudes are all 1
  - $cos(\theta) = a.b$
  
## Two stage retreival
- Search only specific documents
- Two level search
```python
# force the retriver to look into a specific document using metadata filter
retriever1 = db.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"document_name": "Nvidia.txt"}
    }
)

relevant_docs1 = retriever1.invoke(query)
for i, chunk in enumerate(relevant_docs1):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Metadata: {chunk.metadata}")
    print(f"Content: {chunk.page_content[0:100]}")

```
- Specify the document name in the `as_retriever` which acts as filter
- The `document_name` should be meta_data attribute of all the chunks

## different retrieval methods
### Basic retreival
- Similarity search
```python
print("=== METHOD 1: Similarity Search (k=3) ===")
retriever = db.as_retriever(search_kwargs={"k": 3})

docs = retriever.invoke(query)
print(f"Retrieved {len(docs)} documents:\n")

for i, doc in enumerate(docs, 1):
    print(f"Document {i}:")
    print(f"{doc.page_content}\n")

print("-" * 60)
```

### Retreival with threshold
- Similarity with Score Threshold
```python
print("\n=== METHOD 2: Similarity with Score Threshold ===")
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.3  # Only return docs with similarity >= 0.3
    }
)

docs = retriever.invoke(query)
print(f"Retrieved {len(docs)} documents (threshold: 0.3):\n")
```
- Threshold of 0.3 deemed optimal by the user

### Maximum Marginal Relevance
- Balances relevance and diversity - avoids redundant results
- Allows deverse results
- Two part search
  - First finn all vectors with similarity search
  - Find the most diverse vectors score from the subset
  
```python
print("\n=== METHOD 3: Maximum Marginal Relevance (MMR) ===")
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,           # Final number of docs
        "fetch_k": 10,    # Initial pool to select from
        "lambda_mult": 0.5  # 0=max diversity, 1=max relevance
    }
)

docs = retriever.invoke(query)
print(f"Retrieved {len(docs)} documents (λ=0.5):\n")
```
- lambda_mult: Threshold to adjust diversity and relevance 
  - 0=max diversity, 1=max relevance



### Summary
- Similarity Search (Method 1): This is a basic method that returns the top 'k' most similar chunks to your query (0:39). Use it when you need a simple, direct search for the most relevant results. However, a disadvantage is that it will always return 'k' chunks, even if they are irrelevant to the query (1:35).
- Similarity with Score Threshold (Method 2): This method allows you to set a minimum similarity score (e.g., 0.3) (1:05). Use it when you want to ensure that only chunks above a certain relevance are retrieved, preventing the return of irrelevant information when the query is completely unrelated to the documents (2:47).
- Maximum Marginal Relevance (MMR) (Method 3): Use MMR when you want to balance relevance to the query with diversity among the retrieved chunks (3:56). It avoids redundant information by selecting chunks that are not only relevant but also offer different perspectives. This is useful when documents might have overlapping content, you want a well-rounded answer, or you're doing research that requires diverse perspectives (8:35). Do not use MMR when you need only the absolute most relevant results or when speed is critical, as it can be slower (8:54).

# Tips

## Don't use similarity_search_with_score 
```python
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
```
- Gives garbage scores
- Use own function to compute the scores as shown in `retrieval_pipeline.py`

## Joining the chunks for the LLM
- Use the str.join() method method: This is a string method used to concatenate a sequence of strings (like a list of strings) into a single string, using a specified separator.
```python
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}
```
- here the str is `chr(10)` which is newline or `\n`

## Multi query retreival
- Restructure the query into new queries using a LLM and then retrieve the documents for each of the new queries
```python


llm_with_tools = llm.with_structured_output(QueryVariations)
prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:
Original query: {original_query}
Return 3 alternative queries that rephrase or approach the same question from different angles."""
```

## Reciprocal Rank Fusion
- Chunks in Multi query retrieval can repeat and hence we need a way to combine them
- Reciprocal Rank Fusion is way to combine the chunks
- Formula $RRF_{score} = \sum\frac{1}{K + rank-position}$
  - K is a constant typically set to 60
  - `rank-position` is the position in the list where the chunk occurs
  - Sum a chunk across all positions 

### Summary

- Boosts Chunks in Multiple Queries: RRF enhances the scores of chunks that appear in results from multiple queries, demonstrating a "consensus effect" (9:09).
- K Value Significance:
K=0 is too harsh: It overemphasizes top positions and undervalues lower positions significantly (9:13-9:20).
- K=60 provides balance: While top positions still matter, the score differences are more reasonable, preventing small similarity differences from being over-penalized (9:23-9:29). - This value of K is widely used in popular products and projects (9:30-9:32).
- Preserves Diversity: RRF allows unique chunks from each query to still contribute to the final ranking, maintaining diversity in the results (9:35-9:40).
- Simple yet Effective: Despite its simplicity, RRF often outperforms more complex fusion methods (9:43-9:49).
  


# unstructured library for multi-modal document extraction
- To install add `unstructured[all-docs]` in the requirements.txt or `pip install unstructured[all-docs]`
- Need system level install of the following packages
  - Poppler: This library handles PDF processing. It extracts text, images, and metadata from PDFs, which Unstructured uses to process documents (19:19).
  - Tesseract: This is an Optical Character Recognition (OCR) engine. It reads text from scanned documents, images with text, or PDFs that are essentially pictures, converting them into machine-readable text (19:36).
  - Libmagic: This library is used for file type detection. It identifies the file type (like PDF, Word document, or image) by analyzing its content, which helps Unstructured choose the correct processing method (19:55).
- To install on mac `brew install poppler tesseract libmagic`
- To install on linux `apt-get install poppler-utils tesseract-ocr libmagic-dev`

## Extraction
- Use the class `from unstructured.partition.pdf import partition_pdf` 
- Can import other doc types such as `from unstructured.partition.csv, unstructured.partition.ppt`
- Use the `partition_pdf` as follows with the following attributes
```python
def partition_document(file_path: str):
    """Extract elements from PDF using unstructured"""
    print(f"📄 Partitioning document: {file_path}")
    
    elements = partition_pdf(
        filename=file_path,  # Path to your PDF file
        strategy="hi_res", # Use the most accurate (but slower) processing method of extraction
        infer_table_structure=True, # Keep tables as structured HTML, not jumbled text
        extract_image_block_types=["Image"], # Grab images found in the PDF
        extract_image_block_to_payload=True # Store images as base64 data you can actually use
    )
    
    print(f"✅ Extracted {len(elements)} elements")
    return elements

# Test with your PDF file
file_path = "./docs/attention-is-all-you-need.pdf"  # Change this to your PDF path
elements = partition_document(file_path)
```
- elements is list of unstructured class objects of different sections of the pdf as shown below (extracted into a set)
```python 
{"<class 'unstructured.documents.elements.FigureCaption'>",
 "<class 'unstructured.documents.elements.Footer'>",
 "<class 'unstructured.documents.elements.Formula'>",
 "<class 'unstructured.documents.elements.Header'>",
 "<class 'unstructured.documents.elements.Image'>",
 "<class 'unstructured.documents.elements.ListItem'>",
 "<class 'unstructured.documents.elements.NarrativeText'>",
 "<class 'unstructured.documents.elements.Table'>",
 "<class 'unstructured.documents.elements.Text'>",
 "<class 'unstructured.documents.elements.Title'>"}
```
- to see the contents of the element use `to_dict()` function
```python
elements[36].to_dict()
```

### attributes of the element dict

#### general attributes
```python
{'type': 'FigureCaption',         # type of element
 'element_id': '16127143ecf02d9e57b39a1d31c11552',   # element id
 'text': 'Figure 1: The Transformer - model architecture.',  # text if text is present
 'metadata': {'detection_class_prob': 0.8285209536552429,   # meta data 
  'is_extracted': 'true',
  'coordinates': {'points': ((np.float64(582.2593994140625),  # coordinates on the page 
     np.float64(1123.5842266666666)),
    (np.float64(582.2593994140625), np.float64(1151.2581155555554)),
    (np.float64(1118.408203125), np.float64(1151.2581155555554)),
    (np.float64(1118.408203125), np.float64(1123.5842266666666))),
   'system': 'PixelSpace',
   'layout_width': 1700,
   'layout_height': 2200},
  'last_modified': '2025-11-27T22:04:14',
  'filetype': 'application/pdf',            # file type
  'languages': ['eng'],                     # language 
  'page_number': 3,                         # page num
  'file_directory': './docs',               # path
  'filename': 'attention-is-all-you-need.pdf'}}   # filename
```
#### Images
- For images there is a extra keys called
```python
'image_base64': 'AP....//Z',      # base64 conversion of the image
'image_mime_type': 'image/jpeg',  # image type
```
- to plot the image from its base64 code use [https://codebeautify.org/base64-to-image-converter#google_vignette]

#### Tables
- For images there is a extra keys called
```python
'text_as_html': '',      # Table in html format
```
- to view the the table use [https://codebeautify.org/base64-to-image-converter#google_vignette]

## Chunking
- Use the `create_chunks_by_title` function
- Creates chunks by title as the name suggests
- Attributes of the function are shown below
```python
def create_chunks_by_title(elements):
    """Create intelligent chunks using title-based strategy"""
    print("🔨 Creating smart chunks...")
    
    chunks = chunk_by_title(
        elements, # The parsed PDF elements from previous step
        max_characters=3000, # Hard limit - never exceed 3000 characters per chunk
        new_after_n_chars=2400, # Try to start a new chunk after 2400 characters
        combine_text_under_n_chars=500 # Merge tiny chunks under 500 chars with neighbors
    )
    
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# Create chunks
chunks = create_chunks_by_title(elements)
```
- to see the contents of the element use `to_dict()` function
```python
chunks[3].to_dict()
```

### attributes of the chunks dict
- Consists of list of `composite_elements`
```python 
{'type': 'CompositeElement',
 'element_id': '2e9c90aa-0d04-4dd1-8795-b7154f577fbe',
 'text': '1 Introduction\n\nR.... P100 GPUs.',
 'metadata': {'file_directory': './docs',
  'filename': 'attention-is-all-you-need.pdf',
  'filetype': 'application/pdf',
  'languages': ['eng'],
  'last_modified': '2025-11-27T22:04:14',
  'page_number': 2,
  'orig_elements: "ccccc"
 }
}
```
-  To see the original elements use `chunks[11].metadata.orig_elements` which is a list of all the original elements that were chunked together
-  to see the value of the original elements use `chunks[11].metadata.orig_elements[-1].to_dict()` shows the value of the last oriignal element

# Hybrid search

## Summary
Hybrid search is a retrieval strategy that combines vector search and keyword search to enhance the accuracy of Retrieval Augmented Generation (RAG) applications (0:04-0:17, 2:47-2:50). It has become a standard approach for building RAG systems, used by major players like Microsoft Copilot and smaller products alike (0:20-0:33).

Here's a breakdown of its components and benefits:

### Vector Search:

- Excels at understanding semantic meaning and context (1:12-1:14, 2:25-2:27).
- It uses embeddings to find chunks that are semantically similar to a query, even if the exact words aren't present (1:35-1:52).
- Blind Spot: Struggles with exact matches and can miss important keywords if the user knows precisely what they're looking for (1:21-2:17).

### Keyword Search:

- Looks for exact word matches in chunks (2:27-2:30, 3:27-3:30).
- It's a classical approach, historically used by search engines like Google in the early 2000s and currently by Amazon's search bar (2:26-2:39).
- Super reliable for finding specific terms, names, model numbers, and technical jargon (3:37-3:41).
- The primary algorithm used for keyword search is BM25 (Best Matching 25) (4:02-4:04, 4:13-4:17). BM25 scores chunks based on:
  - Term Frequency: How often a search term appears in a specific chunk (4:25-4:29, 5:31-5:35).
  - Inverse Document Frequency: How rare the term is across the entire collection of chunks (4:31-4:36, 6:11-6:17).
- Drawback: It's "dumb" in that it doesn't understand synonyms (e.g., "car" and "automobile" mean the same thing) (3:43-3:53).

### Why Hybrid Search?
- Hybrid search combines the strengths of both, allowing them to cover each other's weaknesses (3:57-3:59, 7:32-7:36). 
- This leads to a significant increase in the accuracy of RAG applications, especially when dealing with specialized or technical documents where exact keyword matching is crucial (2:50-3:23).

## Hybrid search commands
```python
#  3. Hybrid Retriever (Combination)
print("Setting up Hybrid Retriever...")
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # Equal weight to vector and keyword search
)
```
- In hybrid search, weights are used to control the relative importance of different retrievers, such as vector search and keyword search (16:20-16:37).
- When combining results from multiple retrievers using techniques like Reciprocal Rank Fusion (RRF), weights are applied to the individual scores or ranks of each retriever. For instance, a weight of 0.7 for vector search and 0.3 for keyword search means that vector search contributes more to the final ranking of retrieved chunks (19:07-20:07). 
- This allows you to fine-tune the hybrid search system based on the specific needs of your application.

## RAG at scale

- When working with a large number of documents (e.g., 10,000 to 20,000 documents) in RAG applications, the video highlights several tips and challenges related to scaling:

### Challenges with Scaling RAG and Large Documents:
- Increased Chunk Retrieval: Instead of retrieving a small number of chunks (e.g., 3), a large corpus might require retrieving a much larger number of chunks, such as 30, 50, or even 60, from both vector and keyword search components (20:49-21:10).
- Context Window Limitations: Sending a large number of retrieved chunks (e.g., 30 or more) directly to the LLM can exceed its context window, making it unable to process all the information (22:58-23:08).
- Cost Implications: Processing a vast number of tokens from many chunks can become expensive for the LLM (23:08-23:10).
- Complexity with Multi-Query Hybrid Search: For highly complex scenarios, a single user query might be expanded into multiple variations, each performing its own hybrid search. This leads to an even larger number of chunks from various sources that need to be merged and re-ranked (21:25-22:42).

### Tips for Working with Large Numbers of Documents (Solutions for Scaling):
- Hybrid Search as a De Facto Standard: Hybrid search, combining vector and keyword search, is crucial for accurate retrieval, especially with technical jargon or specific terms that pure vector search might miss (0:08-0:17, 2:56-3:23).
- Reciprocal Rank Fusion (RRF): After initial retrieval by individual search methods, RRF is used to combine and re-rank the results from multiple retrievers, handling duplicates and prioritizing more relevant chunks (24:50-25:33).
- Reranking Models: This is the "final piece of the puzzle" for managing scale. A reranking model, which is an intelligent model (not an LLM and less costly), takes all the chunks produced by RRF (e.g., 30 chunks) and re-ranks them based on their relevance to the user's query. This allows you to confidently select only the top K chunks (e.g., top 5 or 10) to send to the LLM for final answer generation (25:57-27:10).
- Specialized Rerankers: There are different types of rerankers, including those trained for specific domains like legal or medical documents, which can further enhance retrieval accuracy for specialized corpora (27:28-27:35).

### Steps for advanced RAG with large number of documents
- Take original query and create multiple queries, e.g. 4-5 varirations 
- For each query use Hybrid retrieval to get 30 to 50 documents
  - NOTEL Dcuments are ranked for each query by relevance 
- Combine all the queries using Reciprocal Rank Fusion but still end up with 10s of chunks
- Use a final ranking algorithm to re-rank to 4-5 chunks
  - Much cheaper than sending all chunks to LLM
- Query the LLM

![RAG Algorithm Part 1](images/full_pag_part1.png)
![RAG Algorithm Part 2](images/full_pag_part2.png)


# Rerankers

## Summary 

- Stage 1: Fast and Broad (Embeddings) (11:38-11:41)
  - Purpose: To quickly cast a wide net and find a large pool of potential candidate chunks (10:07-10:18).
  - Mechanism: Uses embedding models (also called "bi-encoders") that process the query and chunks separately (14:00-14:10). Cosine similarity is then used to compare the embedded query with millions of pre-embedded chunks (10:07-10:08, 14:10-14:14).
  - Characteristics: It is fast (10:40), broad (10:42-10:46), cheap (10:46-10:49) with low computational cost, and reliable (10:52-10:54) in providing chunks with a good probability of relevance. However, it's an approximation (10:57-10:58) and not highly accurate for exact ranking.

- Stage 2: Precise and Focused (Reranker Model) (11:41-11:47)
  - Purpose: To refine the initial broad search results by applying a more sophisticated and precise analysis to finalize the rankings of a smaller set of candidate chunks (13:09-13:13).
  - Mechanism: Uses a "reranker" or "cross-encoder" model (11:47-12:07, 15:17-15:22). This model takes both the user's query and each candidate chunk together as a single input (15:29-15:39). It analyzes their relationship to provide a more accurate relevance score for each chunk (16:09-16:14).
  - Characteristics: It is precise (12:36), context-aware (12:41-12:44) as it reads the query and chunk together, and expensive (12:45-12:47) compared to embeddings, but only applied to a smaller number of chunks (typically 10-100) (12:47-12:57). This stage significantly increases the probability that the top retrieved chunks are the absolute best (12:30-12:33).

## Algorithm

- Vector embedding are bi-encoders
  - The query is converted into its own vector representation (14:12-14:21).
  - Each chunk is also converted into its own separate vector representation (14:23-14:29).
  - The model never processes the query and chunk together during the encoding phase; it processes them independently (15:05-15:07).
  - Similarity is then determined by performing a mathematical comparison, such as cosine similarity, between these two separate vectors (14:32-14:34, 15:10-15:14).

- Re-Rankers are cross encoders
  - The reranker model processes both the user's query and a candidate chunk together as a single input (15:29-15:43). 
  - The model combines the query and chunk into a single string, often using a separator keyword (15:51-15:54). 
  - The reranker is specifically trained to analyze the relationship between the query and the chunk simultaneously, allowing it to understand how similar they are and whether the chunk directly answers the question (15:57-16:06, 16:09-16:12, 16:24-16:28). 
  - This joint processing enables the reranker to provide a much more accurate relevance score compared to embedding models (16:14-16:15, 16:38-16:40).

# OpenSearch
- Database for BM25 search
- Can be run locally on MAC

## Installing and running OpenSearch on MAC

This guide shows **two common ways** to install OpenSearch on a Mac and how to **start / stop** the server.  
For most development workflows, **Docker is recommended**. Homebrew is included for completeness.

### Option 1: Install OpenSearch with Homebrew

#### Install
```bash
brew install opensearch
brew services start opensearch    # start the server
brew services stop opensearch     # stop the server
brew services restart opensearch  # restart the server
brew services list                # Check status
```
- To check opensource is running
```bash
curl http://localhost:9200
```

## Indexing with Opensearch
- Open the client 
- Add an index
```python
client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])

        if not client.indices.exists(index=index_name):
            client.indices.create(
                index=index_name,
                body={
                    "mappings": {
                        "properties": {
                            "doc_id": {"type": "keyword"},
                            "text": {"type": "text"},
                            "source": {"type": "keyword"},
                            "tags": {"type": "keyword"},
                        }
                    }
                },
            )

        # Index chunks
        for chunk in sample_chunks:
            client.index(index=index_name, id=chunk["doc_id"], body=chunk)
```

## Retreiving with OpenSearch
- Open the client
- Query the document
```python
body = {"size": self.k, "query": {"match": {"text": {"query": query}}}}
        resp = self.client.search(index=self.index_name, body=body)
        docs = []
        for hit in resp["hits"]["hits"]:
            source = hit["_source"]
            docs.append(
                Document(
                    page_content="",  # can fetch from vector DB
                    metadata={
                        "doc_id": source.get("doc_id"),
                        "bm25_score": float(hit["_score"]),
                        "source": source.get("source"),
                        "tags": source.get("tags"),
                    },
                )
            )
        return docs
```