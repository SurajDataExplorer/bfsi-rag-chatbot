import os
import json
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_engine import (
    initialize_pipeline,
    ask_bfsi_assistant,
    AUDIT_LOG_PATH,
    BEDROCK_REGION,
    LLM_MODEL_ID,
    EMBED_MODEL_ID,
)

# Global container for preloaded pipeline artifacts
pipeline_state: Dict[str, Any] = {}


# ==============================================================================
# LIFESPAN: PRELOAD FAISS, BM25 & FLASHRANK ONCE AT SERVER STARTUP
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    retriever, db, ranker = initialize_pipeline(initial_k=8)
    pipeline_state["retriever"] = retriever
    pipeline_state["db"] = db
    pipeline_state["ranker"] = ranker
    yield
    pipeline_state.clear()


app = FastAPI(
    title="BFSI Intelligent Financial & Regulatory RAG API",
    description=(
        "Production-grade Retrieval-Augmented Generation microservice built on AWS Bedrock "
        "(Titan V2 + Nova Lite), Hybrid RRF Search (FAISS + BM25), FlashRank Cross-Encoder "
        "Reranking, PII Redaction, and Compliance Guardrails."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for decoupled frontend clients (React, Streamlit, mobile banking apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# PYDANTIC SCHEMAS (STRICT INPUT/OUTPUT CONTRACT)
# ==============================================================================
class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="User financial, policy, or regulatory question.",
        json_schema_extra={
            "example": "What is the late payment fee for PNB credit card if total payment due is Rs 8,000?"
        },
    )


class SourceEvidence(BaseModel):
    source: str
    page: Any
    rerank_score: float
    snippet: str


class QueryResponse(BaseModel):
    trace_id: str
    answer: str
    sources: List[SourceEvidence]
    redacted_pii: List[str]
    guardrail_status: str
    citation_verified: bool
    metrics: Dict[str, Any]


# ==============================================================================
# ENDPOINTS
# ==============================================================================
@app.get("/health", tags=["Operations"])
def health_check():
    """Readiness & liveness check for AWS load balancers and container orchestrators."""
    is_ready = all(k in pipeline_state for k in ("retriever", "db", "ranker"))
    if not is_ready:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized.")

    return {
        "status": "healthy",
        "region": BEDROCK_REGION,
        "embedding_model": EMBED_MODEL_ID,
        "llm_model": LLM_MODEL_ID,
        "indexed_chunks": len(pipeline_state["db"].docstore._dict),
    }


@app.post("/api/v1/query", response_model=QueryResponse, tags=["RAG Inference"])
def query_bfsi_assistant(payload: QueryRequest):
    """
    Executes the full BFSI RAG pipeline:
    Prompt Injection Defense -> PII Masking -> Domain Gate -> Hybrid RRF -> FlashRank -> Nova Lite.
    """
    if "retriever" not in pipeline_state:
        raise HTTPException(status_code=503, detail="Pipeline is still initializing.")

    try:
        result = ask_bfsi_assistant(
            query=payload.query,
            hybrid_retriever=pipeline_state["retriever"],
            vector_db=pipeline_state["db"],
            reranker=pipeline_state["ranker"],
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(exc)}")


@app.get("/api/v1/audit-logs", tags=["Observability & Compliance"])
def get_recent_audit_logs(limit: int = Query(default=20, ge=1, le=200)):
    """Returns the most recent structured JSONL compliance & FinOps traces."""
    if not os.path.exists(AUDIT_LOG_PATH):
        return {"count": 0, "logs": []}

    records = []
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    recent = records[-limit:][::-1]
    return {"count": len(recent), "logs": recent}


@app.get("/api/v1/eval-summary", tags=["Observability & Compliance"])
def get_evaluation_summary():
    """Returns the latest offline benchmark metrics from logs/eval_report.json."""
    eval_path = "logs/eval_report.json"
    if not os.path.exists(eval_path):
        raise HTTPException(status_code=404, detail="Evaluation report not found. Run evaluate.py first.")

    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Return top-level KPI summary without overwhelming payload
    return {k: v for k, v in data.items() if k != "detailed_results"}