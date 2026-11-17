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