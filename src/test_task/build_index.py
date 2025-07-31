import argparse
import hashlib
import json
import logging
import os
import time

import fitz
import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 100


def get_all_pdfs_paths(directory: str) -> list[str]:
    file_paths = []
    logger.info(f"Searching for PDF files in {directory}.")
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                abs_path = os.path.abspath(os.path.join(root, file))
                file_paths.append(abs_path)
    return file_paths


def load_and_index_many(pdf_paths: list[str], metadata_list: list[dict], persist_dir: str = "chroma_store") -> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    emb_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
    logger.info(f"Initializing text models: {splitter} and {emb_model}.")
    texts, ids, metadatas = [], [], []
    chunk_id_lookup = set()
    for path, meta in tqdm.tqdm(zip(pdf_paths, metadata_list), total=len(pdf_paths), desc="Processing PDFs"):
        # By sorting we only assume that metadata and PDF paths are aligned
        # this condition ensures that the metadata UUID matches the file path
        if meta["uuid"] not in path:
            raise ValueError(f"Metadata UUID {meta['uuid']} does not match file path {path}")
        try:
            doc = fitz.open(path)
        except Exception as e:
            logger.warning(f"Error opening {path}: {e}")
            continue

        # Extract text from each page and split into chunks
        # Further, titles could be a kind of clustering key, but here we just use it for context
        full_text = "\n".join([meta["title"]] + meta["industries"] + [page.get_text() for page in doc])
        chunks = splitter.create_documents([full_text])
        for chunk in chunks:
            content = chunk.page_content.strip()
            chunk_id = f"{meta['uuid']}:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"
            if chunk_id in chunk_id_lookup:
                logger.debug(f"Duplicate chunk ID found: {chunk_id}. Skipping this chunk.")
                continue
            texts.append(content)
            ids.append(chunk_id)
            chunk_id_lookup.add(chunk_id)
            metadatas.append(
                {
                    "uuid": meta["uuid"],
                    "date": meta["date"].split("T")[0],
                    "title": meta["title"],
                }
            )

    # Create local Chroma index
    logger.info(f"Creating Chroma index with {len(texts)} chunks.")
    db = Chroma.from_texts(texts, embedding=emb_model, metadatas=metadatas, ids=ids, persist_directory=persist_dir)
    return db


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a Chroma index from PDFs.")
    parser.add_argument("--kb_dir", type=str, required=True, help="Directory containing PDF files and metadata.")
    parser.add_argument("--persist_dir", type=str, default="chroma_store", help="Directory to persist Chroma index.")
    args = parser.parse_args()

    logger.info(f"Starting to build index from PDFs in {args.kb_dir}.")
    # By sorting ensure that metadata and PDF paths are aligned
    with open(os.path.join(args.kb_dir, "metadata.jsonl"), "r") as f:
        metadata = sorted([json.loads(line) for line in f], key=lambda x: x["uuid"])
    pdf_paths = sorted(get_all_pdfs_paths(args.kb_dir))

    logger.info(f"Found {len(pdf_paths)} PDF files and {len(metadata)} metadata entries.")
    logger.info(f"Persisting Chroma index to {args.persist_dir}.")
    start_time = time.time()
    load_and_index_many(pdf_paths, metadata, args.persist_dir)
    elapsed_time = time.time() - start_time
    logger.info(f"Indexing completed in {elapsed_time:.2f} seconds.")
