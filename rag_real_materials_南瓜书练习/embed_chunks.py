"""把 chunks.json 的 857 个片段批量向量化，用 NumPy 存成 npy。

运行：
    EMBEDDING_API_KEY=xxx python embed_chunks.py

产物：
    chunk_vectors.npy   shape=(857, 1024)，已按行做 L2 归一化，可直接点积算余弦
    chunk_vector_ids.json  与向量行同顺序的 chunk_id 列表，用于回查 chunks.json
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).parent
CHUNKS_PATH = HERE / "chunks.json"
VECTORS_PATH = HERE / "chunk_vectors.npy"
IDS_PATH = HERE / "chunk_vector_ids.json"

EMBED_URL = os.environ.get("EMBED_URL", "http://192.168.124.18:8090/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding-0.6b")
EMBED_API_KEY = os.environ.get("EMBEDDING_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))


def embed_batch(texts: list[str]) -> list[list[float]]:
    response = requests.post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {EMBED_API_KEY}"},
        json={"model": EMBED_MODEL, "input": texts},
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()["data"]
    # data 里带 index，按 index 排序保证与输入顺序一致
    return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def main() -> None:
    if not EMBED_API_KEY:
        print("请先设置 EMBEDDING_API_KEY", file=sys.stderr)
        sys.exit(1)

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]
    print(f"待向量化片段: {len(texts)}")

    vectors: list[list[float]] = []
    start = time.time()
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        vectors.extend(embed_batch(batch))
        done = min(i + BATCH_SIZE, len(texts))
        print(f"  {done}/{len(texts)}  elapsed={time.time() - start:.1f}s", flush=True)

    matrix = normalize(np.asarray(vectors, dtype=np.float32))
    np.save(VECTORS_PATH, matrix)
    IDS_PATH.write_text(json.dumps([c["chunk_id"] for c in chunks]), encoding="utf-8")
    print(f"已保存 {VECTORS_PATH.name} shape={matrix.shape} 用时 {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
