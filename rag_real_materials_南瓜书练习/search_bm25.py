"""rank_bm25 最小可用示例：用 chunks.json 做 BM25 检索，可选 --rag 让模型基于检索片段作答。

运行：
    python search_bm25.py "哪三种模型评估方法？"
    python search_bm25.py --rag "书中介绍了哪三种模型评估方法？"
"""

import json
import re
import sys
from pathlib import Path

import jieba
from openai import OpenAI
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


def format_hits(hits: list) -> str:
    """把检索结果渲染成可读文本，同时作为模型提示词里的证据块。"""
    lines = []
    for rank, (i, score, chunk) in enumerate(hits, start=1):
        snippet = chunk["text"].replace("\n", " ")[:1200]
        lines.append(f"[{rank}] score={score:.3f}  {chunk['chunk_id']}  pdf_page={chunk['pdf_page']}")
        lines.append(f"    {snippet}")
        lines.append("")
    return "\n".join(lines).rstrip()


def call_llm(question: str, evidence: str) -> str:
    """沿用 ReAct.py 的接口（OpenAI 客户端、base_url、模型名），单轮问答，不带工具循环。"""
    # 默认会自动读取环境变量中的 OPENAI_API_KEY
    client = OpenAI()
    client.base_url = "http://192.168.124.18:18082/v1"
    prompt = f"""你是资料问答助手。只能依据下面给出的资料片段回答用户问题。

规则：
1. 只使用给定片段中的信息，不得使用片段之外的知识，不得编造。
2. 回答中要标明来源页码（pdf_page），必要时附上 chunk_id。
3. 如果片段不足以回答问题，明确说明「资料不足，无法回答」，不要猜测或补充。

资料片段：
{evidence}

问题：{question}"""
    response = client.chat.completions.create(
        model="qwen38-gyro-m",
        messages=[
            {"role": "system", "content": "你是一个严格依据给定资料作答的助手。"},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("模型没有返回文本内容")
    return content


def answer_with_rag(bm25: BM25Okapi, chunks: list[dict], query: str) -> None:
    """检索前 3 条片段，连同问题一起交给模型，打印模型回答与证据。"""
    hits = search(bm25, chunks, query, top_k=3)
    if not hits:
        print("没有命中（查询词在语料里没出现），资料不足，无法回答")
        return

    evidence = format_hits(hits)
    print("证据：")
    print(evidence)
    print("\n模型回答：")
    print(call_llm(query, evidence))


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--rag"]
    use_rag = "--rag" in sys.argv[1:]
    query = args[0] if args else "交叉验证法常用来比较哪些效果？"

    chunks = load_chunks()
    bm25 = build_index(chunks)

    print(f"语料片段数: {len(chunks)}")
    print(f"查询: {query}\n")

    if use_rag:
        answer_with_rag(bm25, chunks, query)
        return

    hits = search(bm25, chunks, query, top_k=3)
    if not hits:
        print("没有命中（查询词在语料里没出现）")
        return
 
    print(format_hits(hits))


if __name__ == "__main__":
    main()