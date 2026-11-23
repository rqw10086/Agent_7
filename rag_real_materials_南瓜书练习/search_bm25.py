"""rank_bm25 最小可用示例：用 chunks.json 做 BM25 检索。

运行：
    python search_bm25.py "哪三种模型评估方法？"
"""

import json
import re
import sys
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

HERE = Path(__file__).parent
CHUNKS_PATH = HERE / "chunks.json"

# 标点/空白统一换成空格，避免标点被当成独立 token 干扰打分
_PUNCT = re.compile(r"[\s\u3000-\u303f\uff00-\uffef。，、；：？！“”‘’（）《》〈〉【】…—\-·]+")


def tokenize(text: str) -> list[str]:
    """BM25 只认 token 列表，不认原始文本；语料和查询必须用同一套分词。"""
    return [w for w in jieba.cut(_PUNCT.sub(" ", text)) if w.strip()]


def load_chunks() -> list[dict]:
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_index(chunks: list[dict]) -> BM25Okapi:
    # corpus 是「每个片段的词列表」组成的列表，顺序必须与 chunks 一致
    tokenized_chunks = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized_chunks)


def search(bm25: BM25Okapi, chunks: list[dict], query: str, top_k: int = 3):
    query_tokens = tokenize(query)
    # get_scores 返回与 chunks 同顺序的分数数组
    scores = bm25.get_scores(query_tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(i, scores[i], chunks[i]) for i in top_idx if scores[i] > 0]


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "交叉验证法常用来比较哪些效果？"

    chunks = load_chunks()
    bm25 = build_index(chunks)

    print(f"语料片段数: {len(chunks)}")
    print(f"查询: {query}\n")

    hits = search(bm25, chunks, query, top_k=3)
    if not hits:
        print("没有命中（查询词在语料里没出现）")
        return
 
    for rank, (i, score, chunk) in enumerate(hits, start=1):
        snippet = chunk["text"].replace("\n", " ")[:1200]
        print(f"[{rank}] score={score:.3f}  {chunk['chunk_id']}  pdf_page={chunk['pdf_page']}")
        print(f"    {snippet}\n")


if __name__ == "__main__":
    main()