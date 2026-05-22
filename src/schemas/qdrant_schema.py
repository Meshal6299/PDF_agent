# src/schemas/qdrant_schema.py

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance,
    PayloadSchemaType, PointStruct
)

COLLECTION_NAME = "chunks"
VECTOR_SIZE     = 384          # bge-small-en output dimension

def get_qdrant_client(host="localhost", port=6333):
    return QdrantClient(host=host, port=port)

def setup_qdrant(host="localhost", port=6333):
    """
    Call this ONCE at the start of D1 notebook.
    Creates collection and payload indexes.
    """
    client = get_qdrant_client(host, port)
    
    # create collection if it doesn't exist
    existing = [c.name for c in client.get_collections().collections]
    
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
    
    # payload indexes for fast filtering
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="paper_id",
        field_schema=PayloadSchemaType.KEYWORD
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="year",
        field_schema=PayloadSchemaType.INTEGER
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="topics",
        field_schema=PayloadSchemaType.KEYWORD
    )
    
    print("Qdrant collection and indexes ready.")
    return client

# ── qdrant point builder ───────────────────────────────
def build_qdrant_point(chunk_id, embedding, chunk_record):
    """
    chunk_record is the same dict you stored in MongoDB.
    We pull only the lightweight fields into the payload.
    """
    return PointStruct(
        id      = chunk_id,
        vector  = embedding.tolist(),
        payload = {
            "paper_id":    chunk_record["paper_id"],
            "chunk_index": chunk_record["chunk_index"],
            "page_start":  chunk_record["page_start"],
            "page_end":    chunk_record["page_end"],
            "year":        chunk_record["provenance"]["year"],
            "topics":      chunk_record["provenance"].get("topics", []),
            "text":        chunk_record["text"]
        }
    )