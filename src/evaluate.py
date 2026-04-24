"""
Evaluation metrics for paraphrase quality assessment.

Metrics implemented:
- BLEU (sacrebleu)       — n-gram precision vs reference; 30-60 ideal for paraphrasing
- ROUGE-1, ROUGE-2, ROUGE-L — recall-oriented n-gram overlap
- BERTScore F1           — semantic similarity via BERT embeddings (MOST IMPORTANT)
- Self-BLEU              — BLEU of output vs INPUT (measures copying; lower = better)
- Jaccard Similarity     — word-set overlap; 0.3-0.6 ideal
- Lexical Diversity      — unique words / total words; higher = richer vocab
- Length Ratio           — output words / input words; must be >= 0.80

All metrics are computed between (input_text, output_text) or
(reference_text, output_text) depending on what makes sense.
"""

import time
from typing import Optional

import nltk
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn


# Ensure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute BLEU score using sacrebleu.

    BLEU measures n-gram precision of hypothesis against reference.
    For paraphrasing, moderate scores (30-60) are ideal — too high means
    copying, too low means the meaning may have drifted.
    """
    result = sacrebleu.sentence_bleu(hypothesis, [reference])
    return round(result.score, 2)


def compute_rouge(reference: str, hypothesis: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores.

    ROUGE is recall-oriented: how much of the reference is captured.
    Returns F1 scores (harmonic mean of precision and recall).
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1_f1": round(scores["rouge1"].fmeasure, 4),
        "rouge2_f1": round(scores["rouge2"].fmeasure, 4),
        "rougeL_f1": round(scores["rougeL"].fmeasure, 4),
    }


def compute_bert_score(reference: str, hypothesis: str) -> float:
    """Compute BERTScore F1 — semantic similarity using BERT embeddings.

    This is the MOST IMPORTANT metric for paraphrase evaluation because
    it captures meaning preservation even when lexical choices differ.
    A good paraphrase should have BERTScore > 0.85.
    """
    P, R, F1 = bert_score_fn(
        [hypothesis], [reference],
        lang="en",
        verbose=False,
        rescale_with_baseline=True,
    )
    return round(F1.item(), 4)


def compute_self_bleu(input_text: str, output_text: str) -> float:
    """Compute Self-BLEU: BLEU of output against INPUT (not reference).

    Measures how much the model copied from the input.
    Lower = better (more diverse paraphrase).
    A Self-BLEU of 94 means 94% copying — that's a failure.
    Target: < 60 for a real paraphrase.
    """
    return compute_bleu(reference=input_text, hypothesis=output_text)


def compute_jaccard(text_a: str, text_b: str) -> float:
    """Word-level Jaccard similarity.

    Jaccard = |A ∩ B| / |A ∪ B|
    Ideal range for paraphrasing: 0.3-0.6
    Too high = too much copying, too low = meaning may be lost.
    """
    set_a = set(text_a.lower().split())
    set_b = set(text_b.lower().split())
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return round(len(intersection) / len(union), 4)


def compute_lexical_diversity(text: str) -> float:
    """Lexical diversity = unique words / total words.

    Higher = richer vocabulary usage.
    Typically > 0.5 for well-written text.
    """
    words = text.lower().split()
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 4)


def compute_length_ratio(input_text: str, output_text: str) -> float:
    """Length ratio = output words / input words.

    Must be >= 0.80 per TELUS requirement.
    """
    in_words = len(input_text.split())
    out_words = len(output_text.split())
    if in_words == 0:
        return 0.0
    return round(out_words / in_words, 4)


def evaluate_paraphrase(
    input_text: str,
    output_text: str,
    reference_text: Optional[str] = None,
) -> dict:
    """Run all evaluation metrics on a paraphrase.

    Args:
        input_text: Original input text.
        output_text: Paraphrased output text.
        reference_text: Optional human reference paraphrase.
                       If None, input_text is used as reference for
                       BLEU, ROUGE, BERTScore (self-referential evaluation).

    Returns:
        dict with all metric scores.
    """
    ref = reference_text or input_text

    # Quality metrics (vs reference or input)
    bleu = compute_bleu(ref, output_text)
    rouge = compute_rouge(ref, output_text)
    bert_f1 = compute_bert_score(ref, output_text)

    # Diversity metrics (vs input)
    self_bleu = compute_self_bleu(input_text, output_text)
    jaccard = compute_jaccard(input_text, output_text)
    lex_div = compute_lexical_diversity(output_text)

    # Length check
    length_ratio = compute_length_ratio(input_text, output_text)

    return {
        # Quality
        "bleu": bleu,
        "rouge1_f1": rouge["rouge1_f1"],
        "rouge2_f1": rouge["rouge2_f1"],
        "rougeL_f1": rouge["rougeL_f1"],
        "bert_score_f1": bert_f1,
        # Diversity
        "self_bleu": self_bleu,
        "jaccard_similarity": jaccard,
        "lexical_diversity": lex_div,
        # Length
        "length_ratio": length_ratio,
        "input_words": len(input_text.split()),
        "output_words": len(output_text.split()),
    }


if __name__ == "__main__":
    # Quick test with a toy example
    inp = "The cat sat on the mat and looked out the window."
    out = "A feline rested upon the rug while gazing through the glass pane."

    print("Input:", inp)
    print("Output:", out)
    print()

    results = evaluate_paraphrase(inp, out)
    for k, v in results.items():
        print(f"  {k}: {v}")
