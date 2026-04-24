"""
FastAPI REST API for the Custom Paraphrase Generator.

Endpoints:
- POST /paraphrase       — Paraphrase text using CPG
- POST /paraphrase/llm   — Paraphrase text using LLM baseline
- POST /evaluate         — Evaluate paraphrase quality metrics
- POST /compare          — Full CPG vs LLM comparison on same input
- GET  /health           — Health check

Run: uvicorn src.api:app --reload --port 8000
"""

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import CustomParaphraseGenerator
from src.llm_baseline import get_available_generator, BaseLLMGenerator
from src.evaluate import evaluate_paraphrase
from src.utils import preprocess_text, validate_input


# ── Pydantic Models ──────────────────────────────────────────

class ParaphraseRequest(BaseModel):
    text: str = Field(..., description="Input text to paraphrase (200-400 words)")
    model_path: Optional[str] = Field(None, description="Optional custom model path")

class ParaphraseResponse(BaseModel):
    paraphrased_text: str
    latency_ms: float
    input_words: int
    output_words: int
    length_ratio: float

class EvaluateRequest(BaseModel):
    input_text: str
    output_text: str
    reference_text: Optional[str] = None

class CompareResponse(BaseModel):
    cpg: dict
    llm: dict
    cpg_metrics: dict
    llm_metrics: dict


# ── App Setup ────────────────────────────────────────────────

app = FastAPI(
    title="Custom Paraphrase Generator API",
    description="CPG with LLM baseline comparison for TELUS AI",
    version="1.0.0",
)

# Lazy-loaded singletons
_cpg: Optional[CustomParaphraseGenerator] = None
_llm: Optional[BaseLLMGenerator] = None


def get_cpg(model_path: Optional[str] = None) -> CustomParaphraseGenerator:
    global _cpg
    if _cpg is None:
        _cpg = CustomParaphraseGenerator(model_path=model_path)
    return _cpg


def get_llm() -> BaseLLMGenerator:
    global _llm
    if _llm is None:
        _llm = get_available_generator()
    return _llm


# ── Endpoints ────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Custom Paraphrase Generator"}


@app.post("/paraphrase", response_model=ParaphraseResponse)
def paraphrase_cpg(req: ParaphraseRequest):
    """Paraphrase text using the Custom Paraphrase Generator (T5-based)."""
    text = preprocess_text(req.text)
    validation = validate_input(text)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail=validation["error"])

    cpg = get_cpg(model_path=req.model_path)
    result = cpg.paraphrase(text)

    return ParaphraseResponse(
        paraphrased_text=result["paraphrased_text"],
        latency_ms=result["latency_ms"],
        input_words=result["input_words"],
        output_words=result["output_words"],
        length_ratio=result["length_ratio"],
    )


@app.post("/paraphrase/llm", response_model=ParaphraseResponse)
def paraphrase_llm(req: ParaphraseRequest):
    """Paraphrase text using the LLM baseline (Gemini or Claude)."""
    text = preprocess_text(req.text)
    validation = validate_input(text)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail=validation["error"])

    try:
        llm = get_llm()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    result = llm.paraphrase(text)

    return ParaphraseResponse(
        paraphrased_text=result["paraphrased_text"],
        latency_ms=result["latency_ms"],
        input_words=result["input_words"],
        output_words=result["output_words"],
        length_ratio=result["length_ratio"],
    )


@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    """Evaluate paraphrase quality using all metrics."""
    metrics = evaluate_paraphrase(
        input_text=req.input_text,
        output_text=req.output_text,
        reference_text=req.reference_text,
    )
    return metrics


@app.post("/compare", response_model=CompareResponse)
def compare(req: ParaphraseRequest):
    """Run both CPG and LLM on the same input, evaluate both."""
    text = preprocess_text(req.text)
    validation = validate_input(text)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail=validation["error"])

    # Run CPG
    cpg = get_cpg(model_path=req.model_path)
    cpg_result = cpg.paraphrase(text)

    # Run LLM
    try:
        llm = get_llm()
        llm_result = llm.paraphrase(text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")

    # Evaluate both
    cpg_metrics = evaluate_paraphrase(text, cpg_result["paraphrased_text"])
    llm_metrics = evaluate_paraphrase(text, llm_result["paraphrased_text"])

    # Add latency to metrics
    cpg_metrics["latency_ms"] = cpg_result["latency_ms"]
    llm_metrics["latency_ms"] = llm_result["latency_ms"]

    return CompareResponse(
        cpg=cpg_result,
        llm=llm_result,
        cpg_metrics=cpg_metrics,
        llm_metrics=llm_metrics,
    )
