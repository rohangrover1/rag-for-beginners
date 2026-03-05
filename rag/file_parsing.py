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


_ = load_dotenv(find_dotenv())  # read .env file


def canonicalize_text(text: str) -> str:
    """Normalize text for consistent chunk IDs and comparisons"""
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def partition_document(file_path: str):
    """Extract elements from PDF using unstructured"""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        print(f"❌ File not found: {file_path}")
        return []
    try:
        print(f"📄 Partitioning document: {file_path}")
        elements = partition_pdf(
            filename=file_path,  # Path to your PDF file
            strategy="hi_res", # Use the most accurate (but slower) processing method of extraction
            infer_table_structure=True, # Keep tables as structured HTML, not jumbled text
            extract_image_block_types=["Image"], # Grab images found in the PDF
            extract_image_block_to_payload=True, # Store images as base64 data you can actually use
            languages=["English"] # specify language to be english
        )
        print(f"✅ Extracted {len(elements)} elements")
        return elements
    
    except Exception as e:
        # Catch-all for other unexpected errors
        print(f"❌ Error processing PDF: {e}")
        return []

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

def separate_content_types(chunk):
    """Analyze what types of content are in a chunk"""
    content_data = {
        'text': chunk.text,     # text is all texts from diffeent elements joined together and is a single string
        'tables': [],
        'images': [],
        'types': ['text']
    }
    
    # Check for tables and images in original elements
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:        #list of all the elements in the chunk
            element_type = type(element).__name__
            
            # Handle tables
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
            
            # Handle images
            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
    
    content_data['types'] = list(set(content_data['types']))
    return content_data

def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Create AI-enhanced summary for mixed content"""
    try:
        # Initialize LLM (needs vision model for images)
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Build the text prompt
        prompt_text = f"""You are creating a searchable description for document content retrieval.

        CONTENT TO ANALYZE:
        TEXT CONTENT:
        {text}
        """
        
        # Add tables if present
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"
        
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

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add images to the message
        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        print(f"     ❌ AI summary failed: {e}")
        # Fallback to simple summary
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary

def chunk_id(document_name: str, chunk_text: str) -> str:
    """Generate a unique chunk ID using document name and hash of chunk text"""
    h = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
    return f"{document_name}:{h}"

def summarise_chunks(chunks, document_name: str, chunk_index: int = 0):
    """Process all chunks with AI Summaries"""
    print("🧠 Processing chunks with AI Summaries...")
    
    langchain_documents = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        print(f"   Processing chunk {current_chunk}/{total_chunks}")
        
        # Analyze chunk content
        content_data = separate_content_types(chunk)
        
        # Debug prints
        #print(f"     Types found: {content_data['types']}")
        #print(f"     Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")
        
        # Create AI-enhanced summary if chunk has tables/images
        ai_enhanced = False
        if content_data['tables'] or content_data['images']:
            print(f"     → Creating AI summary for mixed content...")
            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data['text'],
                    content_data['tables'], 
                    content_data['images']
                )
                ai_enhanced = True
                #print(f"     → AI summary created successfully")
                #print(f"     → Enhanced content preview: {enhanced_content[:200]}...")
            except Exception as e:
                print(f"     ❌ AI summary failed: {e}")
                enhanced_content = content_data['text']
        else:
            print(f"     → Using raw text (no tables/images)")
            enhanced_content = content_data['text']
        
        # canicalize the enhanced content for consistent chunk IDs
        enhanced_content = canonicalize_text(enhanced_content)

        # create the chunk_id to be used as unique id for the chunk
        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data['text'],
                    "tables_html": content_data['tables'],
                    "images_base64": content_data['images']
                }),
                "chunk_index": chunk_index + i,
                "document_name": document_name,
                "ai_enhanced": ai_enhanced
            },
            id=chunk_id(document_name, enhanced_content)
        )
        langchain_documents.append(doc)
    
    print(f"✅ Processed {len(langchain_documents)} chunks")
    return langchain_documents    

def export_chunks_to_json(chunks):
    """Export processed chunks to clean JSON format"""
    export_data = []
    
    for i, doc in enumerate(chunks):
        chunk_data = {
            "chunk_id": doc.id,
            "enhanced_content": doc.page_content,
            "metadata": {
                "original_content": json.loads(doc.metadata.get("original_content", "{}")),
                "chunk_index": doc.metadata.get("chunk_index"),
                "document_name": doc.metadata.get("document_name")
            }
        }
        export_data.append(chunk_data)
    
    # dump to a text file for inspection
    with open("processed_chunks.json", "w") as f:
        json.dump(export_data, f, indent=2)
    
        


