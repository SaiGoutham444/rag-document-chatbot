# 📚 RAG Document Chatbot

[![CI Pipeline](https://github.com/SaiGoutham444/rag-document-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/SaiGoutham444/rag-document-chatbot/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> **Production-grade RAG chatbot** — Upload any document, ask questions, get cited answers grounded in your content. Every claim is traceable to its source.

---

## 🎯 What Makes This Advanced

Most RAG implementations use basic vector search + an LLM. This project goes further:

| Feature | Basic RAG | This Project |
|---------|-----------|--------------|
| Retrieval | Vector search only | **Hybrid BM25 + Vector** |
| Ranking | Cosine similarity | **Cross-encoder reranking** |
| Fusion | Single retriever | **Reciprocal Rank Fusion** |
| Citations | Optional | **Enforced + validated** |
| Hallucination detection | None | **Citation coverage scoring** |
| Quality assurance | Manual | **Automated eval gate in CI** |
| Testing | None | **45 unit + integration tests** |

---

## 🏗️ Architecture

```
USER UPLOADS DOCUMENT
       │
       ▼
┌─────────────────┐
│ Document Loader │  ← PDF, DOCX, TXT, CSV, HTML
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Chunker   │  ← RecursiveCharacterTextSplitter
│  chunk=500c     │    chunk_size=500, overlap=50
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
[BM25 Index] [ChromaDB]   ← HYBRID RETRIEVAL
(sparse)     (dense)
    │         │
    └────┬────┘
         ▼
┌──────────────────────┐
│   Hybrid Retriever   │  ← Reciprocal Rank Fusion (RRF)
│   BM25 + Vector      │    top-20 candidates
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Cross-Encoder       │  ← ms-marco-MiniLM-L-6-v2
│  Reranker            │    top-5 finalists
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Citation Enforcer   │  ← Prompt engineering +
│                      │    output validation
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  LLM                 │  ← Groq llama-3.3-70b (free)
│  (Groq / OpenAI)     │    or GPT-4o-mini
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Streamlit UI        │  ← Dark theme, citation panel,
│                      │    retrieval stats
└──────────────────────┘
         │
┌──────────────────────┐
│  CI/CD Pipeline      │  ← GitHub Actions
│  (pytest + eval)     │    Quality gate blocks bad PRs
└──────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Framework | LangChain 0.2 | RAG orchestration |
| Vector DB | ChromaDB | Local, persistent, free |
| Embeddings | all-MiniLM-L6-v2 | Free, 384-dim, fast on CPU |
| BM25 | rank-bm25 | Keyword retrieval |
| Reranker | ms-marco-MiniLM-L-6-v2 | Precision cross-encoder |
| LLM | Groq llama-3.3-70b | Free, ultra-fast |
| UI | Streamlit | Pure Python web app |
| Testing | pytest + coverage | 45 tests, >80% coverage |
| Linting | ruff + black | Code quality |
| CI/CD | GitHub Actions | Automated quality gate |

---

## ⚡ Quick Start

### Prerequisites
- Python 3.11+
- Free [Groq API key](https://console.groq.com) (takes 30 seconds)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/SaiGoutham444/rag-document-chatbot.git
cd rag-document-chatbot

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Groq API key
# Get it free at: https://console.groq.com
```

Your `.env` file:
```env
LLM_PROVIDER=groq
EMBEDDING_PROVIDER=local
GROQ_API_KEY=your_groq_key_here
CHROMA_DB_PATH=./chroma_db
BM25_INDEX_PATH=./bm25_index
RETRIEVAL_TOP_K=20
RERANK_TOP_K=5
MIN_CITATION_SCORE=0.7
```

### Run

```bash
streamlit run app.py
```

Browser opens at `http://localhost:8501` automatically.

---

## 💬 Usage

1. **Upload** — Click "Browse files" in the sidebar, upload a PDF/DOCX/TXT/CSV/HTML
2. **Wait** — Document is chunked, embedded, and indexed (~5-30 seconds)
3. **Ask** — Type any question about your document in the chat
4. **Verify** — Click "View Sources" to see the exact chunks used for each answer

### Example Questions
```
"What is the main topic of this document?"
"What are the key findings on page 3?"
"Summarize the technical requirements."
"What does the author recommend?"
```

### Understanding the Response

```
Answer with citations:
  "The revenue was $4.2M [SOURCE: report.pdf | PAGE: 3 | CHUNK: abc123]"

Stats bar:
  Retrieved: 20   ← chunks from hybrid search
  Reranked to: 3  ← chunks after cross-encoder
  BM25/Vec: 1/4   ← keyword vs semantic hits
  Time: 1847ms    ← total query time

Sources panel:
  [HIGH 0.891] Source 1 · report.pdf · Page 3
  "The quarterly revenue reached $4.2 million..."
```

---

## 📁 Project Structure

```
rag-document-chatbot/
│
├── app.py                    ← Streamlit UI entry point
│
├── src/
│   ├── config.py             ← All settings and constants
│   ├── document_loader.py    ← PDF, DOCX, TXT, CSV, HTML loading
│   ├── chunker.py            ← Text splitting with overlap
│   ├── embeddings.py         ← Vector embedding models
│   ├── vector_store.py       ← ChromaDB operations
│   ├── bm25_retriever.py     ← BM25 keyword search
│   ├── hybrid_retriever.py   ← RRF fusion of BM25 + vector
│   ├── reranker.py           ← Cross-encoder reranking
│   ├── citation_enforcer.py  ← Citation building + validation
│   ├── rag_pipeline.py       ← Full pipeline orchestration
│   └── llm.py                ← LLM factory (Groq/OpenAI)
│
├── tests/
│   ├── conftest.py           ← Shared pytest fixtures
│   ├── test_document_loader.py
│   ├── test_chunker.py
│   ├── test_citation_enforcer.py
│   ├── test_rag_pipeline.py
│   └── eval/
│       ├── eval_dataset.json ← 10 ground truth Q&A pairs
│       ├── metrics.py        ← 5 RAG evaluation metrics
│       └── run_evals.py      ← Quality gate runner
│
├── .github/workflows/
│   ├── ci.yml                ← Lint + test pipeline
│   └── eval_gate.yml         ← RAG quality gate
│
├── .env.example              ← Environment template
├── requirements.txt          ← Production dependencies
├── requirements-dev.txt      ← Development dependencies
├── Makefile                  ← Developer shortcuts
└── ruff.toml                 ← Linter configuration
```

---

## 🧪 Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_citation_enforcer.py -v

# Run with coverage HTML report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

Expected output:
```
45 passed in 1.50s
Coverage: 84% citation_enforcer, 78% chunker
```

---

## 📊 Running Evaluations

```bash
# Fast eval (3 questions, ~30 seconds)
python tests/eval/run_evals.py --fast --doc data/sample_docs/resume.pdf

# Full eval (10 questions, ~3 minutes)
python tests/eval/run_evals.py --doc data/sample_docs/resume.pdf
```

### Quality Gate Thresholds

| Metric | Threshold | What It Measures |
|--------|-----------|-----------------|
| Faithfulness | ≥ 0.40 | Claims grounded in context |
| Answer Relevancy | ≥ 0.55 | Answer addresses the question |
| Context Recall | ≥ 0.65 | Retriever finds relevant chunks |
| Context Precision | ≥ 0.65 | Retrieved chunks are relevant |
| Citation Coverage | ≥ 0.40 | % of claims with citations |

---

## 🔄 CI/CD Pipeline

Every push triggers:

```
Push to any branch
      │
      ▼
Code Quality (ruff + black)
      │ passes
      ▼
Unit Tests (pytest, 45 tests)
      │ passes
      ▼
RAG Quality Gate (PRs to main only)
  runs 3 eval questions
  checks all 5 metrics
  posts score table as PR comment
      │
  ┌───┴───┐
  ✅       ❌
merge    blocked
```

### Setting Up CI Secrets

Go to **Settings → Secrets and variables → Actions** and add:

```
GROQ_API_KEY = your_groq_api_key
```

---

## 🚀 Makefile Shortcuts

```bash
make install   # Install all dependencies
make run       # Start the Streamlit app
make test      # Run tests with coverage
make eval      # Run RAG evaluation
make lint      # Check code style
make format    # Auto-fix code style
make clean     # Remove generated files
```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | `groq` or `openai` |
| `EMBEDDING_PROVIDER` | `local` | `local` or `openai` |
| `GROQ_API_KEY` | — | From console.groq.com |
| `OPENAI_API_KEY` | — | From platform.openai.com |
| `RETRIEVAL_TOP_K` | `20` | Chunks from hybrid search |
| `RERANK_TOP_K` | `5` | Chunks after reranking |
| `MIN_CITATION_SCORE` | `0.7` | Citation quality threshold |
| `CHROMA_DB_PATH` | `./chroma_db` | Vector store location |
| `BM25_INDEX_PATH` | `./bm25_index` | BM25 index location |

---

## 📈 Performance

Tested on a standard laptop (no GPU):

| Operation | Time |
|-----------|------|
| First startup (model download) | ~25s |
| Subsequent startup | ~3s |
| Process 1-page PDF | ~2s |
| Process 10-page PDF | ~15s |
| Query (retrieve + rerank + generate) | ~1-3s |
| BM25 index build (200 chunks) | ~0.5s |

---

## ⚠️ Known Limitations

- **Scanned PDFs** — Image-only PDFs return no text. Use Google Drive OCR first.
- **Small documents** — BM25 performs poorly on < 10 chunks (too few for IDF scoring).
- **Groq context window** — llama-3.3-70b has 128K token limit; very long documents may need trimming.
- **Local embeddings** — 384 dimensions vs OpenAI's 1536; slightly lower semantic precision.
- **No memory** — Each query is independent; no conversation history across questions.

---

## 🗺️ Roadmap

- [ ] Multi-document support (query across multiple uploaded files)
- [ ] Conversation memory (follow-up questions referencing previous answers)
- [ ] GPU acceleration for embeddings and reranker
- [ ] Streaming responses in UI
- [ ] PDF OCR support via Tesseract
- [ ] API endpoint (FastAPI wrapper)
- [ ] Docker containerization
- [ ] Pinecone / Weaviate vector store options
- [ ] OpenAI embeddings option in UI

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make changes with tests
4. Ensure CI passes: `make lint && make test`
5. Submit a Pull Request — the eval gate will run automatically

**Commit message convention:**
```
feat:  new feature
fix:   bug fix
docs:  documentation
test:  adding tests
chore: maintenance
ci:    CI/CD changes
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) — RAG framework
- [ChromaDB](https://github.com/chroma-core/chroma) — Vector database
- [Groq](https://groq.com) — Free LLM inference
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) — Embeddings + reranker
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25 implementation
- [Streamlit](https://streamlit.io) — UI framework

---

<div align="center">
Built with ❤️ by <a href="https://github.com/SaiGoutham444">SaiGoutham444</a>
</div>