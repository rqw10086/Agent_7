import json


try:
    # PyMuPDF 的新包名是 pymupdf（pip install pymupdf）
    import pymupdf as fitz
except ImportError:
    # 兼容旧版本：只提供 fitz 这个模块名
    import fitz  # type: ignore[import-not-found]

with fitz.open("human_code/rag_real_materials_南瓜书练习/pumpkin_book.pdf") as doc:
    pages = []
    for page_number, page in enumerate(doc, start=1):
        pages.append({
            "pdf_page": page_number,
            "text": page.get_text("text"),
        })

# 请补三件事：
# 1. 打印总页数
print(f"总页数: {len(pages)}")
# 2. 找到并打印 PDF 第 18 页的记录
if len(pages) >= 18:
    print("第 18 页的记录:")
    print(pages[17])  # 索引从 0 开始，所以第 18 页的索引是 17

# 3. 把 pages 保存为 JSON 文件，保留可读中文


with open("human_code/rag_real_materials_南瓜书练习/pages.json", "w", encoding="utf-8") as f:
    json.dump(pages, f, ensure_ascii=False, indent=2)


# ============ 切片实验（起点参数，不是最佳参数）============
# 1. 读入 JSON，逐页处理，不做跨页拼接
with open("human_code/rag_real_materials_南瓜书练习/pages.json", "r", encoding="utf-8") as f:
    pages = json.load(f)

# 2. 每片目标长度：约 400 字，优先在句号/换行处断开
MAX_CHUNK_CHARS = 400
# 断句优先使用的边界符（句号、问号、感叹号、分号、换行）
BOUNDARIES = "。！？；\n"


def split_into_sentences(text: str) -> list[str]:
    """按边界符切句，边界符保留在句子末尾，保证句子可读。"""
    sentences: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in BOUNDARIES:
            sentences.append(buf)
            buf = ""
    if buf:
        sentences.append(buf)
    return sentences


def chunk_page_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """单页内贪心合并句子；遇到超长句再按长度硬切。"""
    chunks: list[str] = []
    current = ""
    for sentence in split_into_sentences(text):
        # 超长句：先把已攒的片段收尾，再按 max_chars 硬切
        while len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if len(current) + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current += sentence
    if current.strip():
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


# 3. 每片保存 chunk_id / source / pdf_page / text
#    例如 pumpkin-p018-c01 = 第 18 页第 1 个片段
chunks: list[dict] = []
for page in pages:
    for index, text in enumerate(chunk_page_text(page["text"]), start=1):
        chunks.append({
            "chunk_id": f"pumpkin-p{page['pdf_page']:03d}-c{index:02d}",
            "source": "pumpkin_book.pdf",
            "pdf_page": page["pdf_page"],
            "text": text,
        })

# 4. 保存 chunks.json，并检查第 18 页的片段是否被切散
with open("human_code/rag_real_materials_南瓜书练习/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"总片段数: {len(chunks)}")
page18 = [c for c in chunks if c["pdf_page"] == 18]
print(f"第 18 页片段数: {len(page18)}")
for c in page18:
    print("-" * 40)
    print(f"{c['chunk_id']} | {c['source']} | 第 {c['pdf_page']} 页 | 长度 {len(c['text'])}")
    print(c["text"])  



