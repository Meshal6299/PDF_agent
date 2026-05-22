"""
ingest.py
---------
D1 Ingestion Pipeline — Extract, Clean, Chunk, Store, Embed

Steps:
    1. Extract text from PDFs (page by page)
    2. Clean the text
    3. Chunk the text (300 tokens, 50 overlap)
    4. Store chunks in MongoDB
    5. Embed chunks with bge-small-en
    6. Store embeddings in Qdrant
    7. Sanity check both databases

Usage:
    python ingest.py --csv data/papers.csv --pdf_dir data/pdfs
    python ingest.py --csv data/papers.csv --pdf_dir data/pdfs --limit 10
"""

import os
import re
import csv
import sys
import time
import uuid
import argparse
from datetime import datetime
from typing import Optional

# ── Windows terminal fix ───────────────────────────────────────
# The default cp1252 console can't encode Unicode symbols like ✓/✗/—/↔.
# Reconfigure stdout/stderr to UTF-8 so sanity-check output renders correctly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import fitz
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from pymongo import MongoClient, ASCENDING, TEXT
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance,
    PayloadSchemaType, PointStruct
)

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────

MONGO_URI       = os.getenv("MONGO_URI",   "mongodb://localhost:27017")
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", 6333))
DB_NAME         = "csai415"
COLLECTION_NAME = "chunks"
EMBEDDING_MODEL = "BAAI/bge-small-en"
VECTOR_SIZE     = 384
CHUNK_SIZE      = 300   # tokens (words used as proxy)
CHUNK_OVERLAP   = 50    # tokens
BATCH_SIZE      = 64    # embeddings per batch

# ── Qdrant ID helper ──────────────────────────────────────────
# Qdrant only accepts unsigned-integer or UUID point IDs.
# Use uuid5 (deterministic / reversible) so the same chunk_id string
# always maps to the same UUID across runs.
_UUID_NS = uuid.NAMESPACE_DNS

def chunk_id_to_uuid(chunk_id: str) -> str:
    """Convert a chunk_id string to a deterministic UUID string for Qdrant."""
    return str(uuid.uuid5(_UUID_NS, chunk_id))


# ──────────────────────────────────────────────────────────────
# STEP 1 — SETUP DATABASES
# ──────────────────────────────────────────────────────────────

def setup_mongo(uri: str = MONGO_URI):
    """Connect to MongoDB and create collections + indexes."""
    client = MongoClient(uri)
    db = client[DB_NAME]

    # documents collection indexes
    db.documents.create_index([("arxiv_id",  ASCENDING)], unique=True, sparse=True)
    db.documents.create_index([("topics",    ASCENDING)])
    db.documents.create_index([("year",      ASCENDING)])

    # chunks collection indexes
    db.chunks.create_index([("paper_id",   ASCENDING)])
    db.chunks.create_index([("page_start", ASCENDING)])
    db.chunks.create_index([("text",       TEXT)])

    print("[MongoDB] Connected — indexes ready.")
    return db


def setup_qdrant(host: str = QDRANT_HOST, port: int = QDRANT_PORT):
    """Connect to Qdrant and create collection + payload indexes."""
    client = QdrantClient(host=host, port=port)

    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        print(f"[Qdrant] Collection '{COLLECTION_NAME}' created.")
    else:
        print(f"[Qdrant] Collection '{COLLECTION_NAME}' already exists — skipping creation.")

    for field, schema in [
        ("paper_id", PayloadSchemaType.KEYWORD),
        ("year",     PayloadSchemaType.INTEGER),
        ("topics",   PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=schema
        )

    print("[Qdrant] Payload indexes ready.")
    return client


# ──────────────────────────────────────────────────────────────
# STEP 2 — LOAD PAPER METADATA FROM CSV
# ──────────────────────────────────────────────────────────────

def load_papers_csv(csv_path: str) -> list:
    """
    Load paper metadata from CSV.

    Expected columns:
        paper_id, title, authors, venue, year,
        pdf_path, doi (optional), topics (optional)

    authors and topics are semicolon-separated inside the cell.
    Example: "Vaswani; Shazeer; Parmar"
    """
    papers = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            papers.append({
                "paper_id": row["paper_id"].strip(),
                "title":    row.get("title",    "Unknown").strip(),
                "authors":  [a.strip() for a in row.get("authors", "").split(";") if a.strip()],
                "venue":    row.get("venue",    "Unknown").strip(),
                "year":     int(row.get("year", 0) or 0),
                "doi":      row.get("doi",      "").strip(),
                "pdf_path": row.get("pdf_path", "").strip(),
                "topics":   [t.strip() for t in row.get("topics", "").split(";") if t.strip()],
            })
    print(f"[CSV] Loaded {len(papers)} papers.")
    return papers


# ──────────────────────────────────────────────────────────────
# STEP 3 — PDF EXTRACTION
# ──────────────────────────────────────────────────────────────

def extract_pages(pdf_path: str) -> list:
    """
    Extract text from each page of a PDF using PyMuPDF.

    Returns:
        List of {"page_num": int, "text": str}
        Empty list if extraction fails.
    """
    pages = []
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append({"page_num": page_num, "text": text})
        doc.close()
    except Exception as e:
        print(f"  [PDF error] '{pdf_path}': {e}")
    return pages


# ──────────────────────────────────────────────────────────────
# STEP 4 — TEXT CLEANING
# ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean raw PDF text.

    Fixes:
        - Broken hyphenated words across lines  e.g. "trans-\\nformer" → "transformer"
        - Non-breaking spaces and zero-width chars
        - Multiple spaces / tabs collapsed to one
        - Excessive newlines collapsed to two
        - Lone page number lines removed
    """
    # fix hyphenated line breaks
    text = re.sub(r"-\n", "", text)

    # unicode whitespace cleanup
    text = text.replace("\xa0", " ").replace("​", "")

    # collapse tabs and multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # remove standalone page number lines
    text = re.sub(
        r"^\s*[\-–—]?\s*\d+\s*[\-–—]?\s*$",
        "",
        text,
        flags=re.MULTILINE
    )

    return text.strip()


# ──────────────────────────────────────────────────────────────
# STEP 5 — CHUNKING
# ──────────────────────────────────────────────────────────────

def chunk_pages(
    pages:         list,
    paper_id:      str,
    chunk_size:    int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> list:
    """
    Split extracted pages into overlapping token-window chunks.

    Strategy:
        1. Flatten all pages into a single token list
        2. Keep a parallel list mapping each token → source page number
        3. Slide a window of `chunk_size` tokens, stepping by (chunk_size - chunk_overlap) each time
        4. Each chunk records page_start and page_end from the page map

    Returns:
        List of chunk dicts with chunk_id, text, page_start, page_end, etc.
    """
    all_tokens    = []
    token_to_page = []

    for page in pages:
        tokens = clean_text(page["text"]).split()
        all_tokens.extend(tokens)
        token_to_page.extend([page["page_num"]] * len(tokens))

    chunks      = []
    chunk_index = 0
    step        = chunk_size - chunk_overlap
    total       = len(all_tokens)

    for start in range(0, total, step):
        end    = min(start + chunk_size, total)
        tokens = all_tokens[start:end]

        if len(tokens) < 10:   # skip tiny trailing fragments
            break

        chunks.append({
            "chunk_id":    f"{paper_id}_chunk_{chunk_index:03d}",
            "paper_id":    paper_id,
            "chunk_index": chunk_index,
            "text":        " ".join(tokens),
            "page_start":  token_to_page[start],
            "page_end":    token_to_page[end - 1],
            "token_count": len(tokens),
        })

        chunk_index += 1
        if end == total:
            break

    return chunks


# ──────────────────────────────────────────────────────────────
# STEP 6 — BUILD STORAGE RECORDS
# ──────────────────────────────────────────────────────────────

def build_document_record(paper: dict, page_count: int, chunk_count: int) -> dict:
    """Build the MongoDB document record for a paper."""
    return {
        "_id":         paper["paper_id"],
        "title":       paper["title"],
        "authors":     paper["authors"],
        "venue":       paper["venue"],
        "year":        paper["year"],
        "doi":         paper["doi"],
        "pdf_path":    paper["pdf_path"],
        "topics":      paper["topics"],
        "page_count":  page_count,
        "chunk_count": chunk_count,
        "ingested_at": datetime.utcnow().isoformat(),
        "status":      "ingested"
    }


def build_chunk_record(chunk: dict, provenance: dict) -> dict:
    """Build the MongoDB chunk record.

    embedding_id stores the UUID used as the Qdrant point ID so callers
    can cross-reference the two stores without recomputing the UUID.
    """
    return {
        "_id":          chunk["chunk_id"],
        "paper_id":     chunk["paper_id"],
        "chunk_index":  chunk["chunk_index"],
        "text":         chunk["text"],
        "page_start":   chunk["page_start"],
        "page_end":     chunk["page_end"],
        "token_count":  chunk["token_count"],
        "embedding_id": chunk_id_to_uuid(chunk["chunk_id"]),  # UUID == Qdrant point ID
        "provenance":   provenance,
        "created_at":   datetime.utcnow().isoformat()
    }


def build_qdrant_point(chunk: dict, embedding: np.ndarray, paper: dict) -> PointStruct:
    """Build a Qdrant PointStruct from a chunk and its embedding.

    Qdrant only accepts unsigned integers or UUID strings as point IDs.
    A deterministic UUID is derived from the chunk_id via uuid5 so the
    mapping is stable across re-runs.  The original human-readable chunk_id
    is preserved inside the payload for debugging / cross-reference.
    """
    return PointStruct(
        id     = chunk_id_to_uuid(chunk["chunk_id"]),  # valid UUID for Qdrant
        vector = embedding.tolist(),
        payload = {
            "chunk_id":    chunk["chunk_id"],           # human-readable, kept for reference
            "paper_id":    chunk["paper_id"],
            "chunk_index": chunk["chunk_index"],
            "page_start":  chunk["page_start"],
            "page_end":    chunk["page_end"],
            "year":        paper["year"],
            "topics":      paper["topics"],
            "text":        chunk["text"],
        }
    )


# ──────────────────────────────────────────────────────────────
# STEP 7 — EMBEDDING
# ──────────────────────────────────────────────────────────────

def embed_chunks(texts: list, model: SentenceTransformer) -> np.ndarray:
    """
    Embed a list of chunk texts in batches.

    normalize_embeddings=True ensures vectors are unit-length,
    which is required for correct cosine similarity in Qdrant.

    Returns:
        np.ndarray of shape (len(texts), VECTOR_SIZE)
    """
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        emb   = model.encode(
            batch,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        all_embeddings.append(emb)
    return np.vstack(all_embeddings)


# ──────────────────────────────────────────────────────────────
# STEP 8 — MAIN INGESTION LOOP
# ──────────────────────────────────────────────────────────────

def ingest_papers(
    papers:  list,
    db,
    qdrant:  QdrantClient,
    model:   SentenceTransformer,
    pdf_dir: Optional[str] = None
):
    """
    Full ingestion loop.

    For each paper:
        1. Skip if already ingested (idempotent)
        2. Extract text pages from PDF
        3. Chunk pages into overlapping windows
        4. Build and store MongoDB document + chunk records
        5. Embed all chunks for this paper
        6. Upload Qdrant points in batches
    """
    total_chunks_stored = 0

    for paper in tqdm(papers, desc="Ingesting papers"):
        pid = paper["paper_id"]

        # idempotency — skip already ingested papers
        if db.documents.find_one({"_id": pid}):
            tqdm.write(f"  [skip] '{pid}' already ingested.")
            continue

        # resolve PDF path
        pdf_path = paper["pdf_path"]
        if pdf_dir and not os.path.isabs(pdf_path):
            pdf_path = os.path.join(pdf_dir, pdf_path)

        if not os.path.exists(pdf_path):
            tqdm.write(f"  [warn] PDF not found: '{pdf_path}' — skipping.")
            continue

        # ── extract ───────────────────────────────────────────
        pages = extract_pages(pdf_path)
        if not pages:
            tqdm.write(f"  [warn] No text in '{pdf_path}' — skipping.")
            continue

        # ── chunk ─────────────────────────────────────────────
        chunks = chunk_pages(pages, pid)
        if not chunks:
            tqdm.write(f"  [warn] No chunks for '{pid}' — skipping.")
            continue

        # ── provenance (shared across all chunks of this paper) ─
        provenance = {
            "title":   paper["title"],
            "authors": paper["authors"],
            "year":    paper["year"],
            "venue":   paper["venue"],
            "doi":     paper["doi"],
            "topics":  paper["topics"],
        }

        # ── MongoDB ───────────────────────────────────────────
        doc_record    = build_document_record(paper, len(pages), len(chunks))
        chunk_records = [build_chunk_record(c, provenance) for c in chunks]

        db.documents.insert_one(doc_record)
        db.chunks.insert_many(chunk_records)

        # ── Qdrant ────────────────────────────────────────────
        texts      = [c["text"] for c in chunks]
        embeddings = embed_chunks(texts, model)

        points = [
            build_qdrant_point(chunks[i], embeddings[i], paper)
            for i in range(len(chunks))
        ]

        for i in range(0, len(points), BATCH_SIZE):
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i : i + BATCH_SIZE]
            )

        total_chunks_stored += len(chunks)
        tqdm.write(f"  [ok] '{pid}' — {len(chunks)} chunks stored.")

    print(f"\n[Ingest] Finished. Total new chunks stored: {total_chunks_stored:,}")


# ──────────────────────────────────────────────────────────────
# STEP 9 — SANITY CHECKS
# ──────────────────────────────────────────────────────────────

def sanity_check_mongo(db):
    """
    MongoDB sanity check.

    Verifies:
        - document and chunk counts
        - sample document structure
        - sample chunk structure
        - chunk_count field matches actual chunks for one paper
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK — MongoDB")
    print("=" * 60)

    doc_count   = db.documents.count_documents({})
    chunk_count = db.chunks.count_documents({})

    print(f"  documents : {doc_count:,} records")
    print(f"  chunks    : {chunk_count:,} records")

    # sample document
    doc = db.documents.find_one()
    if doc:
        print("\n  Sample document:")
        for k, v in doc.items():
            print(f"    {k:<15}: {v}")

    # sample chunk
    chunk = db.chunks.find_one()
    if chunk:
        print("\n  Sample chunk:")
        for k, v in chunk.items():
            if k == "text":
                print(f"    {'text':<15}: {str(v)[:80]}...")
            elif k == "provenance":
                print(f"    {'provenance':<15}: {list(v.keys())}")
            else:
                print(f"    {k:<15}: {v}")

    # consistency check
    doc = db.documents.find_one({"status": "ingested"})
    if doc:
        actual = db.chunks.count_documents({"paper_id": doc["_id"]})
        stored = doc["chunk_count"]
        status = "✓" if actual == stored else "✗ MISMATCH"
        print(f"\n  Chunk count check for '{doc['_id']}': "
              f"stored={stored}, actual={actual} {status}")

    print("=" * 60)


def sanity_check_qdrant(qdrant: QdrantClient):
    """
    Qdrant sanity check.

    Verifies:
        - point count and vector dimension
        - sample point payload
        - a live similarity search with a random vector

    Note: uses query_points() — the search() method was removed in
    qdrant-client >= 1.7.
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK — Qdrant")
    print("=" * 60)

    info = qdrant.get_collection(COLLECTION_NAME)
    print(f"  collection : {COLLECTION_NAME}")
    print(f"  points     : {info.points_count:,}")
    print(f"  vector dim : {info.config.params.vectors.size}")

    # sample point
    scroll_result = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=1,
        with_vectors=True,
        with_payload=True
    )
    points = scroll_result[0]

    if points:
        p = points[0]
        print(f"\n  Sample point:")
        print(f"    id           : {p.id}")
        print(f"    vector length: {len(p.vector)}")
        print(f"    vector[:5]   : {[round(v, 4) for v in p.vector[:5]]}")
        print(f"\n  Payload:")
        for k, v in p.payload.items():
            if k == "text":
                print(f"    {'text':<15}: {str(v)[:80]}...")
            else:
                print(f"    {k:<15}: {v}")

    # live search with a random query vector
    # query_points() replaces the removed search() method (qdrant-client >= 1.7)
    dummy = np.random.rand(VECTOR_SIZE).astype(np.float32)
    dummy /= np.linalg.norm(dummy)

    result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=dummy.tolist(),
        limit=3
    )

    print(f"\n  Test search (random vector) — top 3 results:")
    for rank, h in enumerate(result.points, start=1):
        print(f"    #{rank}  id={h.id}  score={h.score:.4f}  "
              f"paper={h.payload.get('paper_id', '?')}  "
              f"pages={h.payload.get('page_start', '?')}-{h.payload.get('page_end', '?')}")

    print("=" * 60)


def sanity_check_crossref(db, qdrant: QdrantClient):
    """
    Cross-reference check — MongoDB chunk count must equal Qdrant point count,
    and a spot-check of 5 IDs must resolve in both stores.
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK — Cross-reference MongoDB ↔ Qdrant")
    print("=" * 60)

    mongo_n  = db.chunks.count_documents({})
    qdrant_n = qdrant.get_collection(COLLECTION_NAME).points_count

    status = "✓ match" if mongo_n == qdrant_n else "✗ MISMATCH"
    print(f"  MongoDB chunks : {mongo_n:,}")
    print(f"  Qdrant points  : {qdrant_n:,}")
    print(f"  Status         : {status}")

    # spot-check 5 IDs
    print("\n  Spot-check 5 chunk IDs:")
    samples = list(db.chunks.find({}, {"_id": 1}).limit(5))

    for s in samples:
        cid       = s["_id"]
        qdrant_id = chunk_id_to_uuid(cid)   # MongoDB _id → deterministic UUID
        results   = qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[qdrant_id],
            with_payload=False,
            with_vectors=False
        )
        found = "✓ in Qdrant" if results else "✗ MISSING from Qdrant"
        print(f"    {cid:<40} → {found}")

    print("=" * 60)


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="D1 Ingestion Pipeline")
    parser.add_argument("--csv",     required=True,
                        help="Path to papers metadata CSV")
    parser.add_argument("--pdf_dir", default=None,
                        help="Base directory containing PDF files")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Only ingest the first N papers (for testing)")
    args = parser.parse_args()

    # ── setup ─────────────────────────────────────────────────
    print("\n[Setup] Connecting to databases...")
    db     = setup_mongo()
    qdrant = setup_qdrant()

    print("[Setup] Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # ── load metadata ─────────────────────────────────────────
    papers = load_papers_csv(args.csv)
    if args.limit:
        papers = papers[: args.limit]
        print(f"[Setup] Limiting ingestion to {args.limit} papers.")

    # ── ingest ────────────────────────────────────────────────
    print(f"\n[Ingest] Processing {len(papers)} papers...\n")
    t0 = time.time()
    ingest_papers(papers, db, qdrant, model, pdf_dir=args.pdf_dir)
    print(f"[Ingest] Wall time: {time.time() - t0:.1f}s")

    # ── sanity checks ─────────────────────────────────────────
    sanity_check_mongo(db)
    sanity_check_qdrant(qdrant)
    sanity_check_crossref(db, qdrant)

    print("\n[Done] Ingestion pipeline complete.\n")


if __name__ == "__main__":
    main()
