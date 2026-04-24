# Custom Paraphrase Generator (CPG)

A fine-tuned T5-based paraphrase generator with LLM baseline comparison, built for the TELUS AI internship assignment.

## Overview

This project builds a **Custom Paraphrase Generator (CPG)** that takes paragraphs of 200–400 words and produces high-quality paraphrases preserving semantic meaning while achieving meaningful lexical diversity. The CPG is compared against an LLM baseline (Google Gemini / Anthropic Claude) across quality metrics, diversity metrics, and system latency.

## Architecture

### CPG Model
- **Base model:** [`humarin/chatgpt_paraphraser_on_T5_base`](https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base) (220M params)
- **Fine-tuned on:** PAWS (paraphrase pairs) + QQP (Quora question pairs) — label=1 pairs only
- **Inference strategy:** Sentence-level decomposition → paraphrase each → rejoin
- **Decoding:** Diverse sampling (temperature=1.5, top_k=120, top_p=0.95) with 5 candidates, best picked by Jaccard distance

### LLM Baseline
- **Primary:** Google Gemini 2.0 Flash (free tier)
- **Fallback:** Anthropic Claude (claude-sonnet-4-6)

## Evaluation Metrics

| Metric | What it Measures | Ideal Range |
|--------|-----------------|-------------|
| BLEU | N-gram precision vs reference | 30–60 |
| ROUGE-1/2/L | Recall-oriented overlap | 0.4–0.7 |
| BERTScore F1 | Semantic similarity (BERT embeddings) | > 0.85 |
| Self-BLEU | Copying from input (lower = more diverse) | < 60 |
| Jaccard Similarity | Word-set overlap with input | 0.3–0.6 |
| Lexical Diversity | Unique words / total words | > 0.5 |
| Length Ratio | Output length / input length | ≥ 0.80 |
| Latency | Inference time (ms) | — |

## Project Structure

```
paraphrase-generator/
├── README.md
├── requirements.txt
├── Makefile
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── model.py              # CPG class (load, paraphrase, sentence processing)
│   ├── llm_baseline.py       # Gemini + Claude generators
│   ├── evaluate.py           # All evaluation metrics
│   ├── api.py                # FastAPI REST API
│   ├── run_comparison.py     # Main comparison script
│   ├── visualize_results.py  # Comparison charts
│   └── utils.py              # Text preprocessing & validation
├── notebooks/
│   ├── training.ipynb        # Fine-tuning notebook (Colab-ready)
│   └── evaluation.ipynb      # Full evaluation pipeline
├── data/
│   └── test_sample.txt       # Test passage (cover letter, 329 words)
├── results/                  # Evaluation outputs
└── docs/
    └── REPORT.md             # Full evaluation report
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Keys (for LLM baseline)
```bash
export GEMINI_API_KEY="your-key-here"       # Get free at https://aistudio.google.com/apikey
# OR
export ANTHROPIC_API_KEY="your-key-here"    # Get at https://console.anthropic.com
```

### 3. Run Comparison
```bash
python -m src.run_comparison
```

### 4. Start API Server
```bash
uvicorn src.api:app --reload --port 8000
```

### 5. Fine-tune (Optional — requires GPU)
Open `notebooks/training.ipynb` in Google Colab with T4 GPU runtime.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/paraphrase` | Paraphrase text using CPG |
| POST | `/paraphrase/llm` | Paraphrase text using LLM |
| POST | `/evaluate` | Evaluate paraphrase quality |
| POST | `/compare` | Full CPG vs LLM comparison |
| GET | `/health` | Health check |

## Key Design Decisions

1. **Why `humarin/chatgpt_paraphraser_on_T5_base`?** — Pre-trained on ChatGPT-generated paraphrases, producing genuinely diverse rewrites (unlike raw T5-small which copies input).
2. **Why sentence-level processing?** — T5-base has a 512-token context window; sentence-level keeps each unit within the model's comfort zone.
3. **Why sampling over beam search?** — Beam search is too conservative for paraphrasing; nucleus sampling with high temperature produces diverse candidates.
4. **Why Jaccard-based candidate selection?** — Picking the candidate with lowest word overlap ensures maximum lexical diversity while the model guarantees semantic preservation.

## Author

**Aman Kumar Singh**
B.Tech Materials Science & Engineering, IIT Kanpur (2023–2027)

## License

MIT
