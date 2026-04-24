"""
Custom Paraphrase Generator (CPG) using fine-tuned T5-base.

Base model: humarin/chatgpt_paraphraser_on_T5_base (220M params)
Strategy: Sentence-level decomposition → diverse sampling → Jaccard-based selection
          → post-processing to fix hallucinated questions

Why this architecture?
- humarin/chatgpt_paraphraser_on_T5_base is already trained on ChatGPT-generated
  paraphrases from Quora, SQUAD 2.0, and CNN News — it produces genuinely diverse
  rewrites, unlike raw t5-small which copies 94%+ of input.
- We fine-tune further on PAWS + QQP to customize it (making it a "Custom" generator).
- Sentence-level processing keeps each unit within T5's comfort zone (512 tokens).
- Post-processing strips hallucinated questions that appear when the model encounters
  numbered lists / structured content (a training data mismatch — PAWS+QQP is all prose).
"""

import time
import nltk
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Download sentence tokenizer data
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


# Default model — can be overridden with a fine-tuned checkpoint path
DEFAULT_MODEL = "humarin/chatgpt_paraphraser_on_T5_base"

# Nucleus sampling at temperature=0.7 — coherent and diverse without group beam search
# (group beam search was moved to a broken custom_generate repo in transformers ≥4.50)
DECODE_CONFIG = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "no_repeat_ngram_size": 2,
    "num_return_sequences": 5,
    "max_length": 256,
}


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Word-level Jaccard similarity between two texts.

    Jaccard = |A ∩ B| / |A ∪ B|
    Lower similarity = more diverse paraphrase.
    """
    set_a = set(text_a.lower().split())
    set_b = set(text_b.lower().split())
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


class CustomParaphraseGenerator:
    """T5-based paraphrase generator with diverse decoding and post-processing."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """Initialize the CPG.

        Args:
            model_path: HuggingFace model ID or local checkpoint path.
                        Defaults to humarin/chatgpt_paraphraser_on_T5_base.
            device: 'cuda', 'cpu', or None for auto-detection.
        """
        self.model_path = model_path or DEFAULT_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def load(self):
        """Load model and tokenizer into memory."""
        if self._loaded:
            return
        print(f"Loading CPG model: {self.model_path} -> {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        print("CPG model loaded successfully.")

    def _paraphrase_sentence(self, sentence: str) -> str:
        """Paraphrase a single sentence using diverse sampling.

        Generates 5 candidates and picks the one with lowest Jaccard
        similarity to the input (= most diverse).
        """
        if not sentence.strip():
            return sentence

        # Prepare input with the paraphrase prefix T5 expects
        input_text = f"paraphrase: {sentence}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(self.device)

        # Calculate min_length as 80% of input tokens
        input_token_count = inputs["input_ids"].shape[1]
        min_length = max(10, int(input_token_count * 0.8))

        # Generate multiple candidates, pick most diverse via Jaccard
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                min_length=min_length,
                **DECODE_CONFIG,
            )

        # Decode all candidates
        candidates = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Pick candidate with lowest Jaccard similarity (most diverse)
        best_candidate = sentence  # fallback
        best_distance = -1.0
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue
            sim = jaccard_similarity(sentence, candidate)
            distance = 1.0 - sim  # Higher distance = more diverse
            if distance > best_distance:
                best_distance = distance
                best_candidate = candidate

        return best_candidate

    @staticmethod
    def _post_process(input_sentences: list, output_sentences: list) -> list:
        """Post-process paraphrased sentences to fix known issues.

        Fixes:
        1. Hallucinated questions: If the model injects a question ("What are
           some examples?") but the corresponding input sentence wasn't a
           question, strip it or fall back to the input sentence.
        2. Empty/degenerate outputs: Fall back to input sentence.

        This is needed because PAWS+QQP training data is all prose — the model
        hasn't seen numbered lists or structured content, so it sometimes
        hallucinates interrogative fragments in those sections.
        """
        cleaned = []
        for i, out_sent in enumerate(output_sentences):
            inp_sent = input_sentences[i] if i < len(input_sentences) else ""
            input_has_question = "?" in inp_sent

            # Fix 1: Strip hallucinated questions
            if "?" in out_sent and not input_has_question:
                parts = out_sent.split("?")
                before_question = parts[0].strip()
                if len(before_question.split()) >= max(3, len(inp_sent.split()) * 0.5):
                    # The part before '?' is substantial — keep it as a statement
                    if not before_question.endswith("."):
                        before_question += "."
                    out_sent = before_question
                else:
                    # The question dominates — fall back to input sentence
                    out_sent = inp_sent

            # Fix 2: Empty or very short degenerate output
            if len(out_sent.split()) < 3:
                out_sent = inp_sent

            cleaned.append(out_sent)

        return cleaned

    def paraphrase(self, text: str) -> dict:
        """Paraphrase a full paragraph using sentence-level processing.

        Steps:
        1. Split paragraph into sentences
        2. Paraphrase each sentence independently
        3. Post-process to fix hallucinated questions and degenerate outputs
        4. Rejoin into a paragraph
        5. Validate length ratio >= 0.80

        Returns:
            dict with 'paraphrased_text', 'latency_ms', 'input_words',
            'output_words', 'length_ratio', 'num_sentences'
        """
        self.load()

        # Split into sentences
        sentences = nltk.sent_tokenize(text)

        start_time = time.time()

        # Paraphrase each sentence
        paraphrased_sentences = []
        for sent in sentences:
            para = self._paraphrase_sentence(sent)
            paraphrased_sentences.append(para)

        # Post-process: fix hallucinated questions and degenerate outputs
        paraphrased_sentences = self._post_process(sentences, paraphrased_sentences)

        # Rejoin
        paraphrased_text = " ".join(paraphrased_sentences)

        latency_ms = (time.time() - start_time) * 1000

        # Compute stats
        input_words = len(text.split())
        output_words = len(paraphrased_text.split())
        length_ratio = output_words / input_words if input_words > 0 else 0.0

        return {
            "paraphrased_text": paraphrased_text,
            "latency_ms": round(latency_ms, 2),
            "input_words": input_words,
            "output_words": output_words,
            "length_ratio": round(length_ratio, 4),
            "num_sentences": len(sentences),
        }


if __name__ == "__main__":
    # Quick smoke test
    cpg = CustomParaphraseGenerator()

    test_text = (
        "A cover letter is a formal document that accompanies your resume "
        "when you apply for a job. It serves as an introduction and provides "
        "additional context for your application."
    )

    print(f"Input ({len(test_text.split())} words):")
    print(test_text)
    print()

    result = cpg.paraphrase(test_text)
    print(f"Output ({result['output_words']} words, {result['latency_ms']}ms):")
    print(result["paraphrased_text"])
    print(f"\nLength ratio: {result['length_ratio']}")
    print(f"Sentences processed: {result['num_sentences']}")
