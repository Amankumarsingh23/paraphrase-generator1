"""
Main comparison script: CPG vs LLM on the test passage.

Runs both generators on the cover letter test passage, evaluates both
with all metrics, and saves results to results/evaluation_results.json.

Usage: python -m src.run_comparison
"""

import json
import os
import sys

from src.model import CustomParaphraseGenerator
from src.evaluate import evaluate_paraphrase
from src.utils import load_test_sample, preprocess_text, validate_input


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def run_cpg(text: str, model_path: str = None) -> dict:
    """Run the CPG on the test passage."""
    print("=" * 60)
    print("Running CPG (T5-base paraphraser)...")
    print("=" * 60)

    cpg = CustomParaphraseGenerator(model_path=model_path)
    result = cpg.paraphrase(text)

    print(f"\nCPG Output ({result['output_words']} words, {result['latency_ms']}ms):")
    print(result["paraphrased_text"][:200] + "...")
    print(f"Length ratio: {result['length_ratio']}")

    return result


def run_llm(text: str) -> dict:
    """Run the LLM baseline on the test passage."""
    print("\n" + "=" * 60)
    print("Running LLM Baseline...")
    print("=" * 60)

    try:
        from src.llm_baseline import get_available_generator
        gen = get_available_generator()
        result = gen.paraphrase(text)

        print(f"\nLLM Output ({result['output_words']} words, {result['latency_ms']}ms):")
        print(result["paraphrased_text"][:200] + "...")
        print(f"Length ratio: {result['length_ratio']}")
        print(f"Model: {result['model_name']}")

        return result
    except Exception as e:
        print(f"\nLLM unavailable: {e.__class__.__name__}: {e}")
        print("Skipping LLM comparison. Set GEMINI_API_KEY or ANTHROPIC_API_KEY.")
        return None


def evaluate_results(input_text: str, cpg_result: dict, llm_result: dict = None) -> dict:
    """Evaluate both outputs and build comparison."""
    print("\n" + "=" * 60)
    print("Evaluating paraphrase quality...")
    print("=" * 60)

    # Evaluate CPG
    print("\nComputing CPG metrics...")
    cpg_metrics = evaluate_paraphrase(input_text, cpg_result["paraphrased_text"])
    cpg_metrics["latency_ms"] = cpg_result["latency_ms"]

    print("CPG Metrics:")
    for k, v in cpg_metrics.items():
        print(f"  {k}: {v}")

    # Evaluate LLM
    llm_metrics = None
    if llm_result:
        print("\nComputing LLM metrics...")
        llm_metrics = evaluate_paraphrase(input_text, llm_result["paraphrased_text"])
        llm_metrics["latency_ms"] = llm_result["latency_ms"]

        print("LLM Metrics:")
        for k, v in llm_metrics.items():
            print(f"  {k}: {v}")

    return {"cpg_metrics": cpg_metrics, "llm_metrics": llm_metrics}


def save_results(input_text: str, cpg_result: dict, llm_result: dict,
                 cpg_metrics: dict, llm_metrics: dict):
    """Save all results to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {
        "input_text": input_text,
        "input_words": len(input_text.split()),
        "cpg": {
            "output_text": cpg_result["paraphrased_text"],
            "output_words": cpg_result["output_words"],
            "latency_ms": cpg_result["latency_ms"],
            "length_ratio": cpg_result["length_ratio"],
            "metrics": cpg_metrics,
        },
    }

    if llm_result and llm_metrics:
        results["llm"] = {
            "model_name": llm_result.get("model_name", "unknown"),
            "output_text": llm_result["paraphrased_text"],
            "output_words": llm_result["output_words"],
            "latency_ms": llm_result["latency_ms"],
            "length_ratio": llm_result["length_ratio"],
            "metrics": llm_metrics,
        }

    output_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    # Load and validate test passage
    text = preprocess_text(load_test_sample())
    validation = validate_input(text)
    print(f"Test passage: {validation['word_count']} words")
    assert validation["valid"], f"Invalid input: {validation['error']}"

    # Run both generators
    cpg_result = run_cpg(text, model_path="./cpg-finetuned-final/cpg-finetuned-final")
    llm_result = run_llm(text)

    # Evaluate
    metrics = evaluate_results(text, cpg_result, llm_result)

    # Save
    save_results(
        input_text=text,
        cpg_result=cpg_result,
        llm_result=llm_result,
        cpg_metrics=metrics["cpg_metrics"],
        llm_metrics=metrics["llm_metrics"],
    )

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    cpg_m = metrics["cpg_metrics"]
    print(f"\nCPG:  BERTScore={cpg_m['bert_score_f1']}, "
          f"Self-BLEU={cpg_m['self_bleu']}, "
          f"Jaccard={cpg_m['jaccard_similarity']}, "
          f"Latency={cpg_m['latency_ms']}ms")

    if metrics["llm_metrics"]:
        llm_m = metrics["llm_metrics"]
        print(f"LLM:  BERTScore={llm_m['bert_score_f1']}, "
              f"Self-BLEU={llm_m['self_bleu']}, "
              f"Jaccard={llm_m['jaccard_similarity']}, "
              f"Latency={llm_m['latency_ms']}ms")

    print("\nDone! Run `python -m src.visualize_results` to generate charts.")


if __name__ == "__main__":
    main()
