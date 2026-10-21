# Deployment

## Local

```bash
make install          # venv + pinned deps, creates .env from .env.example
$EDITOR .env          # set GROQ_API_KEY
make index            # fetch 54 Wikipedia articles, chunk, embed into ChromaDB (a few minutes on CPU)
make demo             # Gradio UI on http://127.0.0.1:7860
```

`GRADIO_SERVER_NAME` and `GRADIO_SERVER_PORT` override the bind address and port
(e.g. `GRADIO_SERVER_NAME=0.0.0.0` inside a container).

Requirements: Python 3.11+, ~2 GB disk for models and the index, 4 GB RAM (8 GB for fine-tuning).

## Using the fine-tuned embedding model

The index and the query encoder must use the same model. After `make finetune`:

```bash
echo "EMBEDDING_MODEL=./embeddings/finetuned" >> .env
python src/embed.py --reset      # rebuild the index with the new model
```

`embed.py` records the model name in the collection metadata and refuses to mix vectors from two
models; `--reset` drops and rebuilds the collection.

## Hugging Face Spaces (free, public)

1. Create a Space with the **Gradio** SDK.
2. In *Settings -> Variables and secrets*, add `GROQ_API_KEY` as a **secret** (never commit it).
3. Push this repository to the Space remote. Spaces does not persist the index between restarts,
   so build it on startup (first boot takes several minutes), for example in `app.py`:

   ```python
   if not os.path.exists(CHROMA_PERSIST_DIR):
       subprocess.run([sys.executable, "src/ingest.py"], check=True)
       subprocess.run([sys.executable, "src/embed.py"], check=True)
   ```

   and launch with `GRADIO_SERVER_NAME=0.0.0.0`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `GROQ_API_KEY is missing` | Copy `.env.example` to `.env` and set the key |
| `ChromaDB index not found` | `make index` |
| `Collection ... was built with ...` | You switched embedding models; `python src/embed.py --reset` |
| Groq `429` rate limits | The generator retries with exponential backoff; for evaluation, lower `--num_samples` |
| Port 7860 in use | `GRADIO_SERVER_PORT=7861 make demo` |

More in [troubleshooting.md](troubleshooting.md).
