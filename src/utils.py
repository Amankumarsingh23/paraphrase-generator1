"""
Text utilities and validation helpers for the CPG pipeline.

Handles:
- Input validation (word count 200-400, non-empty)
- Text preprocessing (normalize whitespace, etc.)
- Loading the test sample passage
- Length ratio validation (output must be >= 80% of input)
"""

import os
import re


# Path to the test sample relative to project root
TEST_SAMPLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "test_sample.txt",
)


def load_test_sample() -> str:
    """Load the test passage from data/test_sample.txt."""
    with open(TEST_SAMPLE_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()


def preprocess_text(text: str) -> str:
    """Normalize whitespace and clean up text for model input.

    - Collapse multiple spaces/newlines into single spaces
    - Strip leading/trailing whitespace
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_words(text: str) -> int:
    """Count words in text (whitespace-split)."""
    return len(text.split())


def validate_input(text: str, min_words: int = 10, max_words: int = 400) -> dict:
    """Validate input text for the CPG.

    Args:
        text: Input text to validate.
        min_words: Minimum word count (default 10 for flexibility).
        max_words: Maximum word count per TELUS spec (400).

    Returns:
        dict with 'valid' (bool), 'word_count' (int), 'error' (str or None)
    """
    if not text or not text.strip():
        return {"valid": False, "word_count": 0, "error": "Input text is empty."}

    cleaned = preprocess_text(text)
    wc = count_words(cleaned)

    if wc < min_words:
        return {
            "valid": False,
            "word_count": wc,
            "error": f"Input too short: {wc} words (minimum {min_words}).",
        }

    if wc > max_words:
        return {
            "valid": False,
            "word_count": wc,
            "error": f"Input too long: {wc} words (maximum {max_words}).",
        }

    return {"valid": True, "word_count": wc, "error": None}


def validate_output_length(input_text: str, output_text: str, threshold: float = 0.80) -> dict:
    """Check that output meets the 80% minimum length requirement.

    Args:
        input_text: Original input text.
        output_text: Paraphrased output text.
        threshold: Minimum ratio (default 0.80 per TELUS spec).

    Returns:
        dict with 'meets_requirement', 'input_words', 'output_words', 'ratio'
    """
    in_words = count_words(input_text)
    out_words = count_words(output_text)
    ratio = out_words / in_words if in_words > 0 else 0.0

    return {
        "meets_requirement": ratio >= threshold,
        "input_words": in_words,
        "output_words": out_words,
        "ratio": round(ratio, 4),
        "threshold": threshold,
    }


if __name__ == "__main__":
    # Quick test
    sample = load_test_sample()
    print(f"Test sample loaded: {count_words(sample)} words")
    print(f"First 100 chars: {sample[:100]}...")
    print()

    validation = validate_input(sample)
    print(f"Validation: {validation}")
    print()

    # Test length validation
    short_output = " ".join(sample.split()[:200])
    length_check = validate_output_length(sample, short_output)
    print(f"Length check (short): {length_check}")
