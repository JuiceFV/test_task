How to run:
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Run `uv venv --python 3.11`
3. Run `source .venv/bin/activate`
4. Run `uv sync --extra dev`
5. Run `uv run src/test_task/build_index.py --kb_dir <path/to/domaindata>`
6. Run `uv run src/test_task/main.py --persist_dir <path/to/chroma_store> --top_k <k-candidates>`