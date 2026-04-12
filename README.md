# 🛢️ OpsRAG — Oil & Gas Regulatory Intelligence System

> An AI-powered Retrieval-Augmented Generation (RAG) system for querying Nigerian upstream oil & gas regulations, API 570 pipeline inspection standards, and IOGP Safety Guidelines 2024 — in plain English.

[![Live Demo](https://huggingface.co/spaces/Rasheed91/field-compliance-ai)

---

## 🔍 What Problem Does This Solve?

Integrity engineers, HSE professionals, and petroleum engineers regularly need to cross-reference hundreds of pages of regulatory documents to make compliance and safety decisions. This process is slow, error-prone, and expensive.

**OpsRAG reduces hours of regulatory lookup to seconds** — letting engineers ask questions in plain English and get accurate, cited answers grounded in the actual source documents.

---

## 📚 Knowledge Base

The system is grounded in three authoritative industry document categories:

| Folder | Documents | Coverage |
|---|---|---|
| `regulatory` | Nigerian Upstream Petroleum Regulations (NUPRC) | Upstream compliance, licensing, environmental and operational standards |
| `engineering` | API 570 | Pipeline inspection, repair, alteration and rerating procedures for in-service piping |
| `safety` | IOGP Safety Guidelines 2024 | International upstream safety standards, incident prevention and risk management |

Users can query a specific document category or query across **all documents at once**.

---

## 🧠 How It Works

```
User Question
     │
     ▼
 Folder Filter (engineering / regulatory / safety / all)
     │
     ▼
Query Embedding  ──►  Vector Store Search  ──►  Top-K Relevant Chunks
                                                        │
                                                        ▼
                                            Groq LLM + Retrieved Context
                                                        │
                                                        ▼
                                     Structured Answer with Sources + Domain Tag
```

1. **Ingestion** — Source PDFs are chunked and embedded into a persistent vector store
2. **Filtering** — Queries can be scoped to a specific document folder or run across all
3. **Retrieval** — Semantically similar chunks are retrieved using vector similarity search
4. **Generation** — Groq LLM synthesises a grounded answer using only the retrieved context
5. **Citation** — Responses return source references, domain classification, and reasoning

---

## 🚀 Live Demo

👉 **[Try it on Render](https://huggingface.co/spaces/Rasheed91/field-compliance-ai)**

> ⚠️ Hosted on Render free tier — the app may take ~30 seconds to wake up from cold start. The `/health` endpoint will return `warming_up` status while the vectorstore loads in the background.

### Example Queries

- *"What are the minimum inspection intervals for Class 1 piping under API 570?"*
- *"What does the NUPRC require for flare management in upstream operations?"*
- *"According to IOGP 2024, what are the leading indicators for process safety performance?"*

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| RAG Framework | LangChain |
| LLM | Groq (fast inference) |
| Backend | Flask + Flask-CORS |
| Vector Store | `YOUR_VECTOR_STORE (e.g. FAISS / ChromaDB)` |
| Embeddings | `YOUR_EMBEDDING_MODEL` |
| Frontend | `YOUR_FRONTEND` |
| Deployment | Render |

---

## 📁 Project Structure

```
├── app.py              # Flask backend — routes, health check, warm-up logic
├── rag_chain.py        # LangChain RAG pipeline (retrieval + generation)
├── vectorstore.py      # Vector store initialisation and loading
├── templates/
│   └── index.html      # Frontend UI
├── data/
│   ├── engineering/    # API 570 and piping documents
│   ├── regulatory/     # NUPRC Nigerian upstream regulations
│   └── safety/         # IOGP Safety Guidelines 2024
├── requirements.txt
└── README.md
```

---

## ⚙️ Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Install dependencies
pip install -r requirements.txt

# Add your API keys
cp .env.example .env
# Edit .env and add your credentials

# Launch the app
python app.py
```

---

## 🔑 Environment Variables

```env
GROQ_API_KEY=your_groq_api_key_here
FLASK_ENV=development        # Set to 'production' for deployment
PORT=5000
```

---

## 📡 API Reference

### `POST /api/query`
Submit a question to the RAG pipeline.

**Request body:**
```json
{
  "question": "What are inspection intervals for Class 1 piping?",
  "folder": "engineering"
}
```

**Folder options:** `engineering` | `regulatory` | `safety` | `all`

**Response:**
```json
{
  "question":   "What are inspection intervals for Class 1 piping?",
  "answer":     "According to API 570...",
  "reasoning":  "Retrieved 4 relevant chunks from engineering folder...",
  "sources":    ["API_570_section_6.pdf", "..."],
  "domain":     "engineering",
  "latency_ms": 842
}
```

---

### `GET /health`
Health check endpoint. Returns `warming_up` while the vectorstore loads, then `ok` once ready.

```json
{ "status": "ok", "vectorstore": "loaded", "groq": "connected" }
```

---

### `GET /api/folders`
Returns available document categories.

---

## 💡 Key Design Decisions

- **Background vectorstore loading** — The port binds immediately on startup so Render's health scanner doesn't time out. The vectorstore loads in a background thread; requests during warmup receive a graceful `503` with a retry message instead of crashing.
- **Folder-scoped retrieval** — Users can narrow queries to a specific regulation type, improving precision for domain-specific questions.
- **Metadata-aware chunking** — Each chunk retains its source document label so all answers are fully traceable to the original regulation or standard.
- **Hallucination control** — The LLM is instructed to answer only from retrieved context, critical for safety and compliance use cases where accuracy is non-negotiable.

---

## 🗺️ Roadmap

- [ ] Agentic query routing — automatically selecting the best folder based on question intent
- [ ] Multi-document comparison — how different standards address the same topic side by side
- [ ] Structured risk report generation from conversational input
- [ ] Streaming responses for faster perceived latency

---


[![LinkedIn]([https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/YOUR_PROFILE](https://www.linkedin.com/in/rasheed-adebayo-oyewole-gmnse/))

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

> ⚠️ **Disclaimer:** This system is a research and productivity tool. All regulatory and compliance decisions must be verified against official source documents and reviewed by qualified engineers or legal professionals.
