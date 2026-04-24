"""
LLM Baseline Paraphrase Generators.

Provides real LLM-based paraphrasing for comparison with the CPG.
- Primary: Google Gemini 2.0 Flash (free tier via google-genai SDK)
- Fallback: Anthropic Claude claude-sonnet-4-6

The TELUS task requires comparing CPG against "any LLM based generator."
A synonym dictionary is NOT an LLM — we use actual API-backed language models.

Usage:
    generator = get_available_generator()  # auto-detects API keys
    result = generator.paraphrase("Your text here...")
"""

import os
import time
from abc import ABC, abstractmethod


# Paraphrase system prompt — instructs the LLM to paraphrase properly
PARAPHRASE_PROMPT = """You are a professional paraphrasing assistant. Your task is to paraphrase the given text while following these rules strictly:

1. PRESERVE the complete meaning and all information from the original text.
2. Use different vocabulary, sentence structures, and phrasing.
3. The output MUST be at least 80% the length of the input (in word count).
4. Do NOT add new information not present in the original.
5. Do NOT omit any key points from the original.
6. Output ONLY the paraphrased text — no explanations, headers, or commentary.

Paraphrase the following text:"""


class BaseLLMGenerator(ABC):
    """Abstract base class for LLM paraphrase generators."""

    @abstractmethod
    def paraphrase(self, text: str) -> dict:
        """Paraphrase text and return result dict.

        Returns:
            dict with 'paraphrased_text', 'latency_ms', 'model_name',
            'input_words', 'output_words', 'length_ratio'
        """
        pass

    def _build_result(self, input_text: str, output_text: str,
                      latency_ms: float, model_name: str) -> dict:
        """Build standardized result dict."""
        input_words = len(input_text.split())
        output_words = len(output_text.split())
        return {
            "paraphrased_text": output_text,
            "latency_ms": round(latency_ms, 2),
            "model_name": model_name,
            "input_words": input_words,
            "output_words": output_words,
            "length_ratio": round(output_words / input_words, 4) if input_words > 0 else 0.0,
        }


class GeminiParaphraseGenerator(BaseLLMGenerator):
    """Google Gemini 2.0 Flash paraphrase generator.

    Uses the google-genai SDK (NOT the deprecated google-generativeai).
    Free tier — get API key at https://aistudio.google.com/apikey
    """

    # Try models in order until one works (free tier quotas vary per model)
    _FALLBACK_MODELS = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
    ]

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model_name = "gemini-2.0-flash"
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def paraphrase(self, text: str) -> dict:
        client = self._get_client()
        prompt = f"{PARAPHRASE_PROMPT}\n\n{text}"

        last_error = None
        for model in self._FALLBACK_MODELS:
            try:
                start_time = time.time()
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                latency_ms = (time.time() - start_time) * 1000
                output_text = response.text.strip()
                self.model_name = model
                return self._build_result(text, output_text, latency_ms, model)
            except Exception as e:
                print(f"  [{model}] failed: {e.__class__.__name__} — trying next model...")
                last_error = e

        raise RuntimeError(f"All Gemini models exhausted quota. Last error: {last_error}")


class ClaudeParaphraseGenerator(BaseLLMGenerator):
    """Anthropic Claude paraphrase generator.

    Uses claude-sonnet-4-6 model.
    Requires ANTHROPIC_API_KEY — get $5 free credits at https://console.anthropic.com
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model_name = "claude-sonnet-4-6"
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def paraphrase(self, text: str) -> dict:
        client = self._get_client()

        start_time = time.time()
        response = client.messages.create(
            model=self.model_name,
            max_tokens=2048,
            system=PARAPHRASE_PROMPT,
            messages=[{"role": "user", "content": text}],
        )
        latency_ms = (time.time() - start_time) * 1000

        output_text = response.content[0].text.strip()
        return self._build_result(text, output_text, latency_ms, self.model_name)


def get_available_generator() -> BaseLLMGenerator:
    """Auto-detect which LLM API key is available and return the generator.

    Priority: Gemini (free) > Claude (paid)

    Raises:
        RuntimeError: If no API key is found for any supported LLM.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if gemini_key:
        print(f"Using Gemini 2.0 Flash (API key found)")
        return GeminiParaphraseGenerator(api_key=gemini_key)
    elif anthropic_key:
        print(f"Using Claude claude-sonnet-4-6 (API key found)")
        return ClaudeParaphraseGenerator(api_key=anthropic_key)
    else:
        raise RuntimeError(
            "No LLM API key found. Set one of:\n"
            "  GEMINI_API_KEY  — Free at https://aistudio.google.com/apikey\n"
            "  ANTHROPIC_API_KEY — $5 free at https://console.anthropic.com\n"
        )


if __name__ == "__main__":
    # Quick test
    test = (
        "A cover letter is a formal document that accompanies your resume "
        "when you apply for a job. It serves as an introduction and provides "
        "additional context for your application."
    )

    try:
        gen = get_available_generator()
        result = gen.paraphrase(test)
        print(f"Model: {result['model_name']}")
        print(f"Output ({result['output_words']} words, {result['latency_ms']}ms):")
        print(result["paraphrased_text"])
        print(f"Length ratio: {result['length_ratio']}")
    except RuntimeError as e:
        print(f"Skipping LLM test: {e}")
