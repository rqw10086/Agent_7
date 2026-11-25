"""BM25 与 NumPy 向量检索的对比：同一批问题分别检索，再交给同一个模型作答。

运行：
    EMBEDDING_API_KEY=xxx OPENAI_API_KEY=xxx python compare_bm25_vs_vector.py

依赖：先跑 embed_chunks.py 生成 chunk_vectors.npy / chunk_vector_ids.json。
检索参数沿用 search_bm25.py（top_k=3），模型接口沿用 search_bm25.call_llm。
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import requests
from search_bm25 import call_llm, format_hits, load_chunks, build_index, search

HERE = Path(__file__).parent
VECTORS_PATH = HERE / "chunk_vectors.npy"
IDS_PATH = HERE / "chunk_vector_ids.json"

EMBED_URL = os.environ.get("EMBED_URL", "http://192.168.124.18:8090/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding-0.6b")
EMBED_API_KEY = os.environ.get("EMBEDDING_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

QUESTIONS = [
    "书中介绍了哪三种模型评估方法？",
    "交叉验证法常用来比较哪些效果？",
    "小明的哥哥有什么兴趣爱好？",
    "课本里用哪几种办法来检验模型表现如何？",
]


def embed_query(query: str) -> np.ndarray:
    response = requests.post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {EMBED_API_KEY}"},
        json={"model": EMBED_MODEL, "input": query},
        timeout=60,
    )
    response.raise_for_status()
    vector = np.asarray(response.json()["data"][0]["embedding"], dtype=np.float32)
    return vector / np.linalg.norm(vector)


def load_vector_ids() -> list[str]:
    if not VECTORS_PATH.is_file() or not IDS_PATH.is_file():
        print("缺少 chunk_vectors.npy / chunk_vector_ids.json，请先运行 embed_chunks.py", file=sys.stderr)
        sys.exit(1)
    return json.loads(IDS_PATH.read_text(encoding="utf-8"))


def vector_hits(matrix: np.ndarray, chunks: list[dict], ids: list[str], query: str, top_k: int = 3) -> list:
    query_vector = embed_query(query)
    # 向量已按行归一化，点积即余弦相似度
    scores = matrix @ query_vector
    order = np.argsort(scores)[::-1][:top_k]
    by_id = {c["chunk_id"]: c for c in chunks}
    return [(int(i), float(scores[i]), by_id[ids[i]]) for i in order]


def run_one(bm25, chunks: list[dict], matrix: np.ndarray, ids: list[str], question: str) -> None:
    print("=" * 70)
    print(f"问题: {question}\n")

    bm_hits = search(bm25, chunks, question, top_k=3)
    vec_hits = vector_hits(matrix, chunks, ids, question, top_k=3)

    print("--- BM25 检索结果 ---")
    print(format_hits(bm_hits) if bm_hits else "（无命中）")
    print("\n--- 向量检索结果 ---")
    print(format_hits(vec_hits) if vec_hits else "（无命中）")

    print("\n--- BM25 检索 + 模型回答 ---")
    print(call_llm(question, format_hits(bm_hits)) if bm_hits else "（无命中，资料不足）")
    print("\n--- 向量检索 + 模型回答 ---")
    print(call_llm(question, format_hits(vec_hits)) if vec_hits else "（无命中，资料不足）")
    print()


def main() -> None:
    if not EMBED_API_KEY:
        print("请先设置 EMBEDDING_API_KEY", file=sys.stderr)
        sys.exit(1)

    chunks = load_chunks()
    bm25 = build_index(chunks)
    matrix = np.load(VECTORS_PATH)
    ids = load_vector_ids()
    print(f"语料片段数: {len(chunks)}  向量矩阵: {matrix.shape}\n")

    questions = sys.argv[1:] or QUESTIONS
    for question in questions:
        run_one(bm25, chunks, matrix, ids, question)


if __name__ == "__main__":
    main()
