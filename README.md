# PDF-Papers AI Agent
### CSAI415 — Hybrid Retrieval + GraphRAG with Online Learning and AutoML

An end-to-end AI agent that answers questions over a corpus of scientific PDFs with grounded citations and page ranges. The system combines hybrid retrieval (BM25 + dense embeddings), a knowledge graph (GraphRAG via Neo4j), lightweight online learning with River, and a PEFT/QLoRA-tuned small language model.

## Architecture Overview

```
                          ┌─────────────────────────────────────────┐
                          │              User / API Client           │
                          └──────────────┬──────────────────────────┘
                                         │ /ask  /ingest  /feedback  /stats
                          ┌──────────────▼──────────────────────────┐
                          │           FastAPI Gateway                │
                          └──────┬─────────────┬────────────────────┘
                                 │             │
               ┌─────────────────▼──┐   ┌──────▼──────────────────┐
               │   ReAct/LangGraph  │   │   Ingest Pipeline        │
               │   Agent Planner    │   │  PDF→Text→Chunks+Meta    │
               └──┬──────┬──────┬───┘   └──┬──────┬───────────────┘
                  │      │      │           │      │
         ┌────────▼─┐ ┌──▼───┐ ┌▼──────┐  ┌▼──────▼──┐
         │ Vector   │ │Cypher│ │Mongo  │  │  MongoDB  │
         │ Search   │ │Query │ │Lookup │  │  Qdrant   │
         │(Qdrant)  │ │(Neo4j│ │       │  │  Neo4j    │
         └────────┬─┘ └──┬───┘ └┬──────┘  └──────────┘
                  │      │      │
         ┌────────▼──────▼──────▼────────────┐
         │       GraphRAG Executor            │
         │  1. Cypher subgraph selection      │
         │  2. Chunk expansion                │
         │  3. Hybrid blend + rerank          │
         │  4. Answer with citations+pages    │
         └──────────────┬────────────────────┘
                        │
         ┌──────────────▼────────────────────┐
         │     SLM (PEFT/QLoRA tuned)         │
         │  + River Online Learner (ADWIN)    │
         └───────────────────────────────────┘
```

## Project Structure

```
PDF_agent/
│
├── arxiv_cs_ai_pdfs/          # 250 downloaded arXiv cs.AI PDFs (corpus)
├── notebooks/                 # Jupyter notebooks for experiments & ablations
│
├── src/
│   ├── ingest/
│   │   ├── pdf_parser.py      # PDF → text + page map (PyMuPDF)
│   │   ├── chunker.py         # Text → overlapping chunks + metadata
│   │   └── embedder.py        # Chunk → dense vectors (bge-small-en)
│   │
│   ├── stores/
│   │   ├── mongo_client.py    # MongoDB: chunk/doc metadata, provenance, evals
│   │   ├── qdrant_client.py   # Qdrant: vector index for chunk embeddings
│   │   └── neo4j_client.py    # Neo4j: Paper–Topic–Author–Venue graph
│   │
│   ├── retrieval/
│   │   ├── bm25_retriever.py  # Lexical retrieval (BM25 / TF-IDF)
│   │   ├── dense_retriever.py # Dense retrieval via Qdrant
│   │   ├── hybrid_retriever.py# Hybrid fusion (RRF / weighted)
│   │   └── reranker.py        # Cross-encoder reranking (optional)
│   │
│   ├── graphrag/
│   │   ├── cypher_queries.py  # Subgraph selection Cypher templates
│   │   ├── graph_expander.py  # Cypher path → supporting chunks
│   │   └── graphrag_executor.py # Full 4-step GraphRAG pipeline
│   │
│   ├── agent/
│   │   ├── planner.py         # ReAct/LangGraph tool-using planner
│   │   └── tools.py           # vector_search, cypher_query, mongo_lookup,
│   │                          #   read_pdf_page_range
│   │
│   ├── online_learning/
│   │   ├── river_learner.py   # River: query→topic classifier / hybrid weight
│   │   └── drift_handler.py   # ADWIN drift detection + prequential metrics
│   │
│   ├── automl/
│   │   ├── automl_retriever.py# Optuna/FLAML: kNN k, metric, SVD, hybrid wt
│   │   └── cluster_gating.py  # Unsupervised KMeans cluster gating (Track B)
│   │
│   ├── tuning/
│   │   ├── prepare_qa.py      # Curate Q/A pairs from corpus for fine-tuning
│   │   ├── finetune_qlora.py  # PEFT/QLoRA training script (1–3B model)
│   │   └── tuning_card.yaml   # Dataset size, epochs, lr, LoRA ranks, hardware
│   │
│   ├── evaluation/
│   │   ├── evaluator.py       # Faithfulness, answer-relevance (RAGAS-style)
│   │   ├── metrics.py         # Recall@k, NDCG@k, MRR, latency p95
│   │   └── safety_check.py    # Source pinning, provenance filtering
│   │
│   └── api/
│       ├── main.py            # FastAPI app entry point
│       ├── routes.py          # /ask, /ingest, /feedback, /stats endpoints
│       └── schemas.py         # Pydantic request/response models
│
├── tests/
│   ├── test_ingest.py
│   ├── test_retrieval.py
│   ├── test_graphrag.py
│   └── test_api.py            # pytest smoke tests
│
├── data/
│   └── query_set.csv          # 150 evaluation queries (50 papers × 3 Qs each)
│
├── download_pdfs.py           # arXiv corpus downloader script
├── docker-compose.yml         # MongoDB + Qdrant + Neo4j services
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
└── README.md
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| API | FastAPI |
| PDF Parsing | PyMuPDF |
| Embeddings | sentence-transformers (`bge-small-en-v1.5`) |
| Vector DB | Qdrant |
| Document Store | MongoDB |
| Knowledge Graph | Neo4j |
| Lexical Search | BM25 / TF-IDF |
| Online Learning | River (+ ADWIN drift) |
| AutoML | Optuna / FLAML |
| SLM Fine-tuning | PEFT / QLoRA (1–3B model) |
| Experiment Tracking | MLflow |
| Evaluation | RAGAS-style (faithfulness + relevance) |
| Containerisation | Docker Compose |
| Testing | pytest |

## System Components

### 1. Ingest Pipeline
- PDF → raw text extraction with page-level mapping (PyMuPDF)
- Sliding-window chunking with configurable size and overlap
- Metadata extraction: title, authors, venue, year, DOI, local path, page map
- Storage: chunks + metadata → **MongoDB**; embeddings → **Qdrant**

### 2. Knowledge Graph (Neo4j)
Nodes and relationships:
```
(Author)-[:WROTE]->(Paper)-[:ABOUT]->(Topic)
(Paper)-[:PUBLISHED_IN]->(Venue)
(Paper)-[:CITES]->(Paper)          ← if time permits
```
Loaded via seed scripts in `docker compose` setup.

### 3. Hybrid Retrieval
- **Lexical:** BM25 / TF-IDF over chunk text
- **Dense:** `bge-small-en-v1.5` embeddings via Qdrant ANN
- **Fusion:** Reciprocal Rank Fusion (RRF) or learned weighted blend
- **Reranking:** Optional cross-encoder reranking

### 4. GraphRAG Executor
Four-step pipeline:
1. **Subgraph selection** — Cypher query to find relevant Paper/Topic nodes
2. **Chunk expansion** — retrieve supporting chunks from matched papers
3. **Hybrid blend** — merge graph-guided + vector results; optional rerank
4. **Answer generation** — SLM generates answer with inline citations + page ranges

### 5. Agent Planner (ReAct/LangGraph)
Tool-using planner with four tools:
- `vector_search(query, k)` — dense/hybrid search in Qdrant
- `cypher_query(statement)` — execute Cypher on Neo4j
- `mongo_lookup(filter)` — fetch doc metadata from MongoDB
- `read_pdf_page_range(paper_id, start, end)` — retrieve raw text from pages

### 6. Online Learning (River)
- **Component:** query→topic classifier **or** adaptive hybrid fusion weight
- **Drift handling:** ADWIN detector with lightweight retraining triggers
- **Feedback loop:** user clicks "helpful / not helpful" on `/feedback`
- **Monitoring:** prequential accuracy plot per deliverable

### 7. AutoML (Optuna / FLAML)
**Track A (Supervised):** auto-tune kNN retriever over:
`k`, distance metric, SVD dimension, normalization, hybrid weight

**Track B (Unsupervised):** cluster-gating with KMeans/spherical-KMeans over:
`k`, SVD, hybrid features; optimise internal metrics + labeled proxy

### 8. SLM Fine-Tuning (PEFT/QLoRA)
- Base model: 1–3B parameter open-source LLM
- Training data: curated Q/A pairs from corpus (see `src/tuning/prepare_qa.py`)
- Quantize (4-bit) + cache for low-latency inference
- See `src/tuning/tuning_card.yaml` for full reproducibility card

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest` | Ingest a PDF or directory of PDFs |
| `POST` | `/ask` | Ask a question; returns answer + citations |
| `POST` | `/feedback` | Submit helpful/not-helpful signal |
| `GET` | `/stats` | Retrieval and system performance metrics |

### Example: `/ask`

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How does GraphRAG improve retrieval over vector-only search?"}'
```

```json
{
  "answer": "GraphRAG improves retrieval by ...",
  "citations": [
    {"paper": "2605.20815v1_GraphRAG on Consumer Hardware...", "pages": "3-5"},
    {"paper": "2605.20084v1_BalanceRAG...", "pages": "7"}
  ],
  "latency_ms": 840
}
```

## Evaluation & Baseline Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Recall@5 | ≥ 0.60 | On `query_set.csv` gold set |
| MRR | ↑ vs baseline | Mean Reciprocal Rank |
| p95 Latency | ≤ 2s | CPU-only on small corpora |
| Faithfulness | ≥ 0.80 | RAGAS-style |
| Answer-relevance | ≥ 0.80 | RAGAS-style |
| Online learning gain | > +5% rel. | vs. static model on temporal slice |
| QLoRA gain | +3–5 pts | Relevance/faithfulness, or equal quality at lower latency |

Run evaluation:
```bash
python -m src.evaluation.evaluator --query-set data/query_set.csv --mode graphrag
```

## Deliverables & Timeline

| # | Due | Weight | Description |
|---|-----|--------|-------------|
| **D1** | Week 5 | 15% | Streaming Learner & AutoML Notebook |
| **D2** | Week 7 | 15% | Retrieval Stack & Graph Build |
| **D3** | Week 9 | 15% | GraphRAG Executor, Evaluation & Safety |
| **D4** | Week 10/11 | 15% | SLM Tuning & Final Demo Package |

### D1 — Week 5: Streaming Learner & AutoML
- [ ] AutoML search (Track A or B) with Optuna/FLAML run card (YAML/JSON)
- [ ] River online learner with ADWIN drift handling
- [ ] Prequential metrics plot
- [ ] 2-page report: baseline vs AutoML (NDCG@5/Recall@5), p95 latency

### D2 — Week 7: Retrieval Stack & Graph Build
- [ ] Full ingest pipeline (PDF→chunks→MongoDB+Qdrant)
- [ ] Hybrid `/search` endpoint (BM25 + Dense)
- [ ] Neo4j graph loaded (Authors, Papers, Topics, 3–5 Cypher examples)
- [ ] Dataflow diagram + Recall@k/latency metrics table
- [ ] Docker Compose + seed scripts

### D3 — Week 9: GraphRAG Executor, Evaluation & Safety
- [ ] GraphRAG 4-step executor with page-level citations
- [ ] Gold Q/A evaluation (faithfulness, answer-relevance, latency p95)
- [ ] At least one safety mitigation (source pinning / provenance filtering) with before/after evidence
- [ ] Ablation: vector-only vs graph-guided vs hybrid

### D4 — Week 10/11: SLM Tuning & Final Demo
- [ ] PEFT/QLoRA fine-tuning with complete `tuning_card.yaml`
- [ ] Zero-shot vs tuned comparison inside GraphRAG
- [ ] 8-minute live demo
- [ ] 8–10 page report (architecture, experiments, ablations, failure cases, ethics, future work)
- [ ] pytest smoke tests pass; `.env.example` present
