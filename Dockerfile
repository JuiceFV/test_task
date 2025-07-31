FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y wget unzip && \
    wget https://isi-ml-public.s3.us-east-1.amazonaws.com/domaindata.zip && \
    unzip domaindata.zip -d /app/domaindata && \
    rm domaindata.zip && \
    apt-get purge -y --auto-remove wget unzip && \
    rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
RUN uv venv --python 3.11 && \
    source .venv/bin/activate && \
    uv sync --extra dev

RUN uv run src/test_task/build_index.py --kb_dir /app/domaindata --persist_dir /app/chroma_store

EXPOSE 8000

CMD uv run src/test_task/main.py --persist_dir /app/chroma_store --top_k 4
