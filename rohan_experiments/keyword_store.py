import json
from typing import Iterable, List
from opensearchpy import OpenSearch, helpers
from langchain_core.documents import Document


def index_documents(sample_chunks, index_name: str) -> str:
    ''' code to index langchain documents into OpenSearch BM25 one document at a time'''
    try:
        client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])

        ''' # Document format
        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data['text'],
                    "tables_html": content_data['tables'],
                    "images_base64": content_data['images']
                }),
                "chunk_index": chunk_index + i,
                "document_name": document_name
            },
            id=chunk_id(document_name, enhanced_content)
        )
        '''

        if not client.indices.exists(index=index_name):
            client.indices.create(
                index=index_name,
                body={
                    "mappings": {
                        "properties": {
                            "doc_id": {"type": "keyword"},
                            "document_name": {"type": "keyword"},
                            "chunk_index": {"type": "integer"},
                            "text": {"type": "text"},        # BM25
                            "raw_text": {"type": "text"},
                            "tables_html": {"type": "text"},
                            "images_base64": {
                                "type": "text",
                                "index": False               # critical
                            }
                        }
                    }
                },
            )

        # Index chunks
        for chunk in sample_chunks:
            # create the dict body for indexing
            original_content = json.loads(
                chunk.metadata.get("original_content", "{}")
            )

            chunk_body = {
                "doc_id": chunk.id,  # ✅ use deterministic Document.id
                "document_name": chunk.metadata["document_name"],
                "chunk_index": chunk.metadata["chunk_index"],
                "text": chunk.page_content,
                "raw_text": original_content.get("raw_text"),
                "tables_html": original_content.get("tables_html"),
                "images_base64": original_content.get("images_base64"),
            }
            client.index(index=index_name, id=chunk.id, body=chunk_body)
    except Exception as e:
        print(f"Error indexing documents: {e}")
        return None
    
''' Recommended code in production
1. Generate new chunks
2. delete_by_query(document_name)
3. bulk upsert new chunks
4. refresh index
'''

def get_opensearch_client() -> OpenSearch:
    '''Return an OpenSearch client connected to localhost:9200'''
    return OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        http_compress=True,
    )

def ensure_index(client: OpenSearch, index_name: str) -> None:
    '''Ensure the OpenSearch index exists with the correct mappings'''
    if client.indices.exists(index=index_name):
        return

    client.indices.create(
        index=index_name,
        body={
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "document_name": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "text": {"type": "text"},          # BM25 field
                    "raw_text": {"type": "text"},
                    "tables_html": {"type": "text"},
                    "images_base64": {
                        "type": "text",
                        "index": False
                    },
                }
            }
        },
    )

def bulk_upsert_documents(
    client: OpenSearch,
    index_name: str,
    documents: Iterable[Document],
    refresh: bool = False,
) -> int:
    """
    Bulk upsert LangChain Documents using deterministic Document.id.
    """

    def actions():
        for doc in documents:
            original_content = json.loads(
                doc.metadata.get("original_content", "{}")
            )

            yield {
                "_op_type": "update",
                "_index": index_name,
                "_id": doc.id,
                "doc": {
                    "doc_id": doc.id,
                    "document_name": doc.metadata["document_name"],
                    "chunk_index": doc.metadata["chunk_index"],
                    "text": doc.page_content,
                    "raw_text": original_content.get("raw_text"),
                    "tables_html": original_content.get("tables_html"),
                    "images_base64": original_content.get("images_base64"),
                },
                "doc_as_upsert": True,
            }

    success, _ = helpers.bulk(
        client,
        actions(),
        refresh=refresh,
        raise_on_error=False,
    )

    return success

def delete_document_chunks(
    client: OpenSearch,
    index_name: str,
    document_name: str,
    refresh: bool = True,
) -> int:
    """
    Delete all chunks belonging to a document.
    """

    response = client.delete_by_query(
        index=index_name,
        body={
            "query": {
                "term": {
                    "document_name": document_name
                }
            }
        },
        conflicts="proceed",
        refresh=refresh,
    )

    return response["deleted"]

def reindex_document(
    client: OpenSearch,
    index_name: str,
    document_name: str,
    new_chunks: List[Document],
) -> dict:
    """
    Delete all existing chunks for a document and re-index new ones.
    """

    deleted = delete_document_chunks(
        client,
        index_name,
        document_name,
        refresh=True,
    )

    indexed = bulk_upsert_documents(
        client,
        index_name,
        new_chunks,
        refresh=True,
    )

    return {
        "document_name": document_name,
        "deleted_chunks": deleted,
        "indexed_chunks": indexed,
    }


def update_keyword_store(document_name: str, chunks: List[Document], index_name: str) -> None:
    '''Example function to update the OpenSearch keyword store with new chunks'''

    # 2. OpenSearch setup
    client = get_opensearch_client()
    ensure_index(client, index_name)

    # check if document already exists in index
    # No need to do this check if we are doing a full re-index with delete_by_query, 
    # but it's a nice optimization to avoid unnecessary deletes + upserts if the document is unchanged
    existing_docs = client.search(
        index=index_name,
        body={
            "query": {
                "term": {
                    "document_name": document_name
                }            }
        })
    if existing_docs["hits"]["total"]["value"] > 0:
        print("🚫 Document already indexed in keyword store — skipping ingest")
        return None
        
    # 3. Re-index lifecycle
    result = reindex_document(
        client=client,
        index_name=index_name,
        document_name=document_name,
        new_chunks=chunks,
    )
    print("Re-index result:", result)
    return True

def number_of_documents_in_keyword_store(index_name: str):
    '''Return the total number of documents indexed in the OpenSearch keyword store'''
    client = get_opensearch_client()
    if not client.indices.exists(index=index_name):
        print(f"Index '{index_name}' does not exist.")
    response = client.count(index=index_name)
    print(f"📊 Total documents in keyword store index '{index_name}': {response['count']}")