"""Fetch official LangChain and Qdrant documentation pages, chunk them,
embed locally, and upsert into a remote Qdrant collection.

    python ingest.py                # ingest every URL in data/urls.txt
    python ingest.py --recreate     # drop the collection first
    python ingest.py --limit 10     # smoke test on the first 10 pages
"""
import argparse
import re
import uuid

import requests
from bs4 import BeautifulSoup
from qdrant_client.models import Distance, VectorParams, PointStruct

from src import config, retriever

HEADERS = {"User-Agent": "Task0-DocsIngest/1.0 (+educational use)"}
TIMEOUT = 30


def read_urls(path: str):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def source_of(url: str) -> str:
    """Label each chunk so answers can say which product the fact came from."""
    if "qdrant.tech" in url:
        return "Qdrant"
    if "langgraph" in url:
        return "LangGraph"
    return "LangChain"


def extract(html: str):
    """Pull the readable body text out of a docs page."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""
    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        return title, ""

    text = main.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return title, text.strip()


def chunk(text: str, size: int, overlap: int):
    """Character-window split with overlap, snapped to the nearest line break."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        window = text[start:end]
        if end < len(text):
            cut = max(window.rfind("\n"), window.rfind(". "))
            if cut > size * 0.5:
                window = window[:cut + 1]
                end = start + cut + 1
        piece = window.strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", default=config.URLS_FILE)
    parser.add_argument("--recreate", action="store_true", help="drop the collection first")
    parser.add_argument("--limit", type=int, default=None, help="only ingest the first N URLs")
    args = parser.parse_args()

    config.require_env()
    client = retriever.get_client()
    dim = retriever.get_embedder().get_sentence_embedding_dimension()

    exists = client.collection_exists(config.COLLECTION_NAME)
    if args.recreate and exists:
        client.delete_collection(config.COLLECTION_NAME)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"Created collection '{config.COLLECTION_NAME}' (dim={dim})")

    urls = list(read_urls(args.urls))
    if args.limit:
        urls = urls[:args.limit]

    records, failed = [], []
    for i, url in enumerate(urls, start=1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            response.raise_for_status()
            title, text = extract(response.text)
            if len(text) < 200:
                failed.append((url, "page too short after extraction"))
                continue
            pieces = chunk(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            for j, piece in enumerate(pieces):
                records.append({
                    "text": piece,
                    "url": url,
                    "title": title,
                    "source": source_of(url),
                    "chunk_id": f"{i}-{j}",
                })
            print(f"[{i}/{len(urls)}] {len(pieces):3d} chunks  {url}")
        except Exception as exc:  # noqa: BLE001 - one bad page must not stop the run
            failed.append((url, str(exc)))
            print(f"[{i}/{len(urls)}] FAILED  {url}  ({exc})")

    if not records:
        raise SystemExit("Nothing ingested. Check data/urls.txt and your connection.")

    print(f"\nPrepared {len(records)} chunks from {len(urls) - len(failed)} pages. Embedding...")

    batch = 64
    for i in range(0, len(records), batch):
        window = records[i:i + batch]
        vectors = retriever.embed([r["text"] for r in window])
        client.upsert(
            collection_name=config.COLLECTION_NAME,
            points=[
                PointStruct(id=str(uuid.uuid4()), vector=v, payload=r)
                for v, r in zip(vectors, window)
            ],
        )
        print(f"  upserted {min(i + batch, len(records))}/{len(records)}")

    print(f"\nDone. {len(records)} chunks in '{config.COLLECTION_NAME}'.")
    if failed:
        print(f"\n{len(failed)} page(s) failed:")
        for url, reason in failed:
            print(f"  - {url}  ({reason})")


if __name__ == "__main__":
    main()
