PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin
QUERY  ?= How does BERT differ from GPT?

.PHONY: quickstart install index ingest embed query demo audit finetune eval eval-retrieval test lint clean

## quickstart: install deps, build the index and launch the demo UI
quickstart: install index demo

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)

install: $(BIN)/python
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -r requirements-dev.txt
	@test -f .env || (cp .env.example .env && echo "Created .env - add your GROQ_API_KEY before running queries.")

index: ingest embed

ingest:
	$(BIN)/python src/ingest.py

embed:
	$(BIN)/python src/embed.py --reset

query:
	$(BIN)/python src/pipeline.py --query "$(QUERY)"

demo:
	$(BIN)/python app.py

audit:
	$(BIN)/python src/bias_audit.py

finetune:
	$(BIN)/python src/finetune_embeddings.py --samples 100 --epochs 2

eval:
	$(BIN)/python src/evaluate.py --num_samples 20

eval-retrieval:
	$(BIN)/python src/evaluate.py --retrieval_only

test:
	$(BIN)/python -m pytest -q

lint:
	$(BIN)/ruff check .

clean:
	rm -rf data/raw data/processed vectorstore embeddings/finetuned results logs
