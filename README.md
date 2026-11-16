# Intelligent Financial & Regulatory Assistant (BFSI RAG on AWS Bedrock)

A production-grade Retrieval-Augmented Generation (RAG) system designed for the **Banking, Financial Services, and Insurance (BFSI)** domain. Built on **AWS Bedrock**, it answers complex regulatory, credit card, mutual fund, UPI, and health insurance queries with strict anti-hallucination guardrails, PII redaction, hybrid search, cross-encoder reranking, and full FinOps/compliance observability.

---

## Architecture Overview

```text
User Query (Streamlit UI / FastAPI REST API)
   │
   ├─► [Layer 1: Security Guardrails]
   │      ├── Prompt Injection & Jailbreak Filter (Regex/Heuristic)
   │      └── PII Masking (PAN, Card/Account Numbers, Indian Mobile, Email)
   │
   ├─► [Layer 2: Domain Relevance Gate]
   │      └── Pre-LLM FAISS L2 Distance Check (Blocks out-of-domain queries at $0 LLM cost)
   │
   ├─► [Layer 3: Two-Stage Hybrid Retrieval & Context Reduction]
   │      ├── Stage 1: Weighted Reciprocal Rank Fusion (60% FAISS Dense + 40% BM25 Sparse, k=8)
   │      └── Stage 2: FlashRank Cross-Encoder Reranker (ms-marco-MiniLM-L-12-v2) + Deduplication (top_n=4)
   │
   ├─► [Layer 4: Grounded Generation & Citation Check]
   │      └── AWS Bedrock (Amazon Nova Lite, temp=0.0) + Inline [Source: file, Page X] Verification
   │
   └─► [Layer 5: Compliance & FinOps Observability]
          └── Structured JSONL Audit Trail (Latency spans, token counts, USD cost, guardrail actions)
```

---

## Offline Benchmark Results (`evaluate.py`)

Evaluated using an automated **LLM-as-a-Judge + Deterministic Guardrail Suite** across 7 BFSI factual, multi-page synthesis, PII masking, prompt injection, and out-of-domain test cases:

| Metric | Score |
| :--- | :--- |
| **Average Faithfulness** | **91.4%** |
| **Average Answer Relevancy** | **98.6%** |
| **Source Recall Rate** | **100.0%** |
| **Inline Citation Compliance** | **100.0%** |
| **Security & Guardrail Accuracy** | **100.0%** |
| **Average End-to-End Latency** | **2,421.0 ms** |

---

## Project Structure

```text
bfsi-rag-chatbot/
├── data/                 # Source BFSI PDFs (RBI, NPCI, AMFI, Credit Card MITC, Health Insurance)
├── faiss_index/          # Pre-built 1024-d Amazon Titan V2 FAISS vector index (454 chunks)
├── logs/eval_report.json # Saved offline evaluation benchmark report
├── ingest.py             # Table-aware PDF parser (PyMuPDF4LLM) + chunking + Bedrock embedding pipeline
├── rag_engine.py         # Core RAG engine (PII redaction, guardrails, Hybrid RRF, FlashRank, Nova Lite)
├── api.py                # FastAPI production REST microservice (/api/v1/query, /health, /audit-logs)
├── app.py                # Streamlit interactive chat UI & real-time FinOps/Observability dashboard
├── evaluate.py           # Automated LLM-as-a-Judge & security guardrail evaluation script
└── requirements.txt      # Python dependencies
```

---

## Quickstart (Local)

1. **Install dependencies:**
   ```bash
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Configure AWS credentials** (requires Bedrock access to `amazon.titan-embed-text-v2:0` and `amazon.nova-lite-v1:0` in `ap-southeast-2`):
   ```bash
   aws configure
   ```
3. **Run FastAPI Microservice (`/docs`):**
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```
4. **Run Streamlit UI & Observability Dashboard:**
   ```bash
   streamlit run app.py
   ```