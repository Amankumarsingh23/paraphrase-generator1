.PHONY: install test compare visualize api clean help

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

test:  ## Quick smoke test of the CPG model
	python -m src.model

compare:  ## Run full CPG vs LLM comparison on test passage
	python -m src.run_comparison

visualize:  ## Generate comparison charts from results
	python -m src.visualize_results

api:  ## Start the FastAPI server on port 8000
	uvicorn src.api:app --reload --port 8000

clean:  ## Remove generated results and cache
	rm -rf results/*.json results/*.png
	rm -rf __pycache__ src/__pycache__
	rm -rf .pytest_cache
