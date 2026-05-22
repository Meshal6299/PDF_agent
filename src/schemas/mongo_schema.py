# src/schemas/mongo_schema.py

from pymongo import MongoClient, ASCENDING, TEXT
from datetime import datetime

def get_database(uri="mongodb://localhost:27017", db_name="csai415"):
    client = MongoClient(uri)
    return client[db_name]

def setup_mongo(uri="mongodb://localhost:27017"):
    """
    Call this ONCE at the start of D1 notebook.
    Creates collections and indexes.
    """
    db = get_database(uri)
    
    # ── documents collection indexes ──────────────────
    db.documents.create_index(
        [("arxiv_id", ASCENDING)], unique=True
    )
    db.documents.create_index([("topics", ASCENDING)])
    db.documents.create_index([("year",   ASCENDING)])
    
    # ── chunks collection indexes ──────────────────────
    db.chunks.create_index([("paper_id",    ASCENDING)])
    db.chunks.create_index([("page_start",  ASCENDING)])
    db.chunks.create_index([("text",        TEXT)])      # text search
    
    print("MongoDB collections and indexes ready.")
    return db

# ── document record builder ────────────────────────────
def build_document_record(
    paper_id, title, authors, venue,
    year, doi, pdf_path, topics,
    page_count, chunk_count=0
):
    return {
        "_id":         paper_id,
        "title":       title,
        "authors":     authors,
        "venue":       venue,
        "year":        year,
        "doi":         doi,
        "pdf_path":    pdf_path,
        "topics":      topics,
        "page_count":  page_count,
        "chunk_count": chunk_count,
        "ingested_at": datetime.utcnow().isoformat(),
        "status":      "ingested"
    }

# ── chunk record builder ───────────────────────────────
def build_chunk_record(
    chunk_id, paper_id, chunk_index,
    text, page_start, page_end,
    token_count, provenance
):
    return {
        "_id":          chunk_id,
        "paper_id":     paper_id,
        "chunk_index":  chunk_index,
        "text":         text,
        "page_start":   page_start,
        "page_end":     page_end,
        "token_count":  token_count,
        "embedding_id": chunk_id,      # mirrors Qdrant point ID
        "provenance":   provenance,    # dict with title/authors/year/venue/doi
        "created_at":   datetime.utcnow().isoformat()
    }