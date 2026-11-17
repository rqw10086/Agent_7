"""用 MinerU / Docling / PyMuPDF4LLM 三种方案切同一本《南瓜书》，结果各存一个文件。

运行环境是 human_code/.venv-slice（含 mineru、docling、pymupdf4llm），
且必须在 LocalAgent 目录下运行（脚本里用的是 human_code/... 相对路径）：

    cd /Users/ruanwen/PycharmProjects/LocalAgent
    human_code/.venv-slice/bin/python human_code/rag_real_materials_南瓜书练习/chunk_three_ways.py

只跑其中几个方案 / 指定页码：

    ... chunk_three_ways.py --tools docling,pymupdf4llm --pages 1-20

三个方案对「切片」的处理方式不同，对比时要记住这一点：

- MinerU：CLI 只输出整页 markdown（公式转 LaTeX、表格还原、页眉页脚剔除），本身不切块。
  这里按 `<!-- page N of M -->` 还原页码后，复用 extract_pages.py 里那套贪心切句逻辑切到约 400 字。
- Docling：用它自带的 HybridChunker（max_tokens=400），块可跨页，页码来自 provenance。
- PyMuPDF4LLM：用它自带的 to_chunks（layout 模式，max_tokens=400），块可跨页，页码来自 metadata。
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path("human_code/rag_real_materials_南瓜书练习")
PDF_PATH = BASE_DIR / "pumpkin_book.pdf"
SOURCE = "pumpkin_book.pdf"
VENV_BIN = Path("human_code/.venv-slice/bin")

# 与 extract_pages.py 保持一致的目标长度，方便横向比较
MAX_CHUNK_CHARS = 400
BOUNDARIES = "。！？；\n"
# MinerU 在 markdown 里插入的页标记，例如 <!-- page 18 of 196 -->
PAGE_MARKER = re.compile(r"<!--\s*page\s+(\d+)\s+of\s+(\d+)\s*-->")


# ---------------------------------------------------------------- 公共工具


def split_into_sentences(text: str) -> list[str]:
    """按边界符切句，边界符留在句尾，保证句子可读。"""
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


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """单页内贪心合并句子；超长句先收尾已攒片段再按长度硬切。"""
    chunks: list[str] = []
    current = ""
    for sentence in split_into_sentences(text):
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


def save_chunks(tool: str, chunks: list[dict], elapsed: float) -> Path:
    """每个方案单独一个文件，字段顺序统一，中文不转义。"""
    out_path = BASE_DIR / f"chunks_{tool}.json"
    payload = {
        "tool": tool,
        "source": SOURCE,
        "chunk_count": len(chunks),
        "elapsed_seconds": round(elapsed, 1),
        "chunks": chunks,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    total_chars = sum(len(c["text"]) for c in chunks)
    print(
        f"[{tool}] 片段数 {len(chunks)} | 总字符 {total_chars} | "
        f"平均 {total_chars // max(len(chunks), 1)} | 用时 {elapsed:.1f}s -> {out_path}"
    )
    return out_path


def parse_pages_arg(value: str) -> list[int] | None:
    """把 '1-20' / '18,19' / 'all' 解析成 1 基页码列表；all 返回 None。"""
    if value.strip().lower() == "all":
        return None
    pages: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    return sorted(set(pages))


# ---------------------------------------------------------------- MinerU


def mineru_binary() -> Path:
    """优先用当前解释器同目录的 mineru，其次用 PATH 上的。"""
    candidate = Path(sys.executable).parent / "mineru"
    if candidate.exists():
        return candidate
    found = shutil.which("mineru")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "找不到 mineru 命令。请用 human_code/.venv-slice/bin/python 运行本脚本。"
    )


def run_mineru_cli(*args: str, timeout: int) -> str:
    result = subprocess.run(
        [str(mineru_binary()), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"mineru 命令失败: {' '.join(args)}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def ensure_mineru_server() -> None:
    """MinerU 4.x 的 parse 走本地服务；没起就顺手起一个。"""
    status = subprocess.run(
        [str(mineru_binary()), "server", "status"],
        capture_output=True,
        text=True,
    )
    if "未在运行" in status.stdout or status.returncode != 0:
        print("[mineru] 服务未运行，正在启动 mineru server start（首次会加载模型）")
        run_mineru_cli("server", "start", timeout=300)
        time.sleep(5)


MINERU_BATCH_PAGES = 20
MINERU_TOTAL_PAGES = 196
MINERU_ATTEMPTS = 3


def mineru_parse_batch(pages_arg: str, tmp_out: Path) -> str:
    """解析一个页码批次；parse-server 偶尔会被健康检查重启，重试时服务会自动拉起。"""
    last_error: Exception | None = None
    for attempt in range(1, MINERU_ATTEMPTS + 1):
        try:
            ensure_mineru_server()
            # --no-marker 会连页码标记一起去掉，这里保留标记以便还原 pdf_page
            run_mineru_cli(
                "parse",
                str(PDF_PATH),
                "-p",
                pages_arg,
                "--wait",
                "1800",
                "-o",
                str(tmp_out),
                timeout=3600,
            )
            return tmp_out.read_text(encoding="utf-8")
        except RuntimeError as error:
            last_error = error
            print(f"[mineru] 批次 {pages_arg} 第 {attempt} 次失败，准备重试")
            time.sleep(10)
    raise RuntimeError(f"MinerU 批次 {pages_arg} 连续 {MINERU_ATTEMPTS} 次失败") from last_error


def chunk_with_mineru(pages: list[int] | None) -> tuple[list[dict], float]:
    started = time.time()
    tmp_out = BASE_DIR / ".mineru_markdown_tmp.md"
    target_pages = pages if pages is not None else list(range(1, MINERU_TOTAL_PAGES + 1))

    # 分批解析：单次请求覆盖整本书会让 parse-server 超时被重启，进而断连
    batches: list[str] = []
    for start in range(0, len(target_pages), MINERU_BATCH_PAGES):
        batch = target_pages[start:start + MINERU_BATCH_PAGES]
        batches.append(f"{batch[0]}-{batch[-1]}")

    markdown_parts: list[str] = []
    for pages_arg in batches:
        markdown_parts.append(mineru_parse_batch(pages_arg, tmp_out))
        print(f"[mineru] 批次 {pages_arg} 完成")
    tmp_out.unlink(missing_ok=True)
    markdown = "\n".join(markdown_parts)
    elapsed = time.time() - started

    # 按页标记切段；没有标记（例如整篇只有一页）时退化成单段
    marks = list(PAGE_MARKER.finditer(markdown))
    chunks: list[dict] = []
    if not marks:
        segments = [(1, markdown)]
    else:
        segments = []
        for index, match in enumerate(marks):
            end = marks[index + 1].start() if index + 1 < len(marks) else len(markdown)
            segments.append((int(match.group(1)), markdown[match.end():end]))

    for pdf_page, page_markdown in segments:
        for index, text in enumerate(chunk_text(page_markdown), start=1):
            chunks.append(
                {
                    "chunk_id": f"mineru-p{pdf_page:03d}-c{index:02d}",
                    "source": SOURCE,
                    "tool": "mineru",
                    "pdf_page": pdf_page,
                    "page_start": pdf_page,
                    "page_end": pdf_page,
                    "text": text,
                }
            )
    return chunks, elapsed


# ---------------------------------------------------------------- Docling


def chunk_with_docling(pages: list[int] | None) -> tuple[list[dict], float]:
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter

    started = time.time()
    converter = DocumentConverter()
    # Docling 的 page_range 只能给连续区间，散页请求先按区间转，再按实际页码过滤
    convert_kwargs: dict[str, object] = {} if pages is None else {"page_range": (pages[0], pages[-1])}
    document = converter.convert(PDF_PATH, **convert_kwargs).document
    chunker = HybridChunker(max_tokens=MAX_CHUNK_CHARS)

    chunks: list[dict] = []
    counter: dict[int, int] = {}
    for raw in chunker.chunk(document):
        page_numbers = sorted(
            {prov.page_no for item in raw.meta.doc_items for prov in (item.prov or [])}
        )
        if pages is not None and not any(p in pages for p in page_numbers):
            continue
        page_start = page_numbers[0] if page_numbers else 1
        counter[page_start] = counter.get(page_start, 0) + 1
        chunks.append(
            {
                "chunk_id": f"docling-p{page_start:03d}-c{counter[page_start]:02d}",
                "source": SOURCE,
                "tool": "docling",
                "pdf_page": page_start,
                "page_start": page_start,
                "page_end": page_numbers[-1] if page_numbers else page_start,
                "pdf_pages": page_numbers,
                "headings": raw.meta.headings or [],
                "text": raw.text,
            }
        )
    return chunks, time.time() - started


# ---------------------------------------------------------------- PyMuPDF4LLM


def chunk_with_pymupdf4llm(pages: list[int] | None) -> tuple[list[dict], float]:
    import pymupdf4llm

    # to_chunks 只在 layout 模式下可用，不开会抛 NotImplementedError
    pymupdf4llm.use_layout(True)
    started = time.time()
    # 它默认按「4 字符 ≈ 1 token」估算，中文实际约 1 字符 ≈ 1 token，
    # 直接传 max_tokens 会让片长比另外两个方案大 4 倍；这里改成按字符计数，口径统一。
    kwargs: dict[str, object] = {
        "max_tokens": MAX_CHUNK_CHARS,
        "tokenizer": lambda text: len(text),
    }
    if pages is not None:
        kwargs["pages"] = [p - 1 for p in pages]  # pymupdf4llm 的 pages 是 0 基
    parsed = pymupdf4llm.to_chunks(str(PDF_PATH), **kwargs)

    chunks: list[dict] = []
    counter: dict[int, int] = {}
    for chunk in parsed.chunks:
        meta = chunk.metadata
        page_start = int(meta.page_start)
        counter[page_start] = counter.get(page_start, 0) + 1
        chunks.append(
            {
                "chunk_id": f"pymupdf4llm-p{page_start:03d}-c{counter[page_start]:02d}",
                "source": SOURCE,
                "tool": "pymupdf4llm",
                "pdf_page": page_start,
                "page_start": page_start,
                "page_end": int(meta.page_end),
                "section_path": list(meta.section_path or []),
                "types": list(meta.types or []),
                "token_count": meta.token_count,
                "text": chunk.text,
            }
        )
    return chunks, time.time() - started


# ---------------------------------------------------------------- 汇总核对

RUNNERS = {
    "mineru": chunk_with_mineru,
    "docling": chunk_with_docling,
    "pymupdf4llm": chunk_with_pymupdf4llm,
}


def summarize(tool: str, chunks: list[dict]) -> None:
    """核对第 18 页（第一题证据页）是否被切散，以及证据句是否完整留在同一片里。"""
    page18 = [c for c in chunks if c["page_start"] <= 18 <= c["page_end"]]
    keywords = ["留出法", "交叉验证法", "自助法"]
    whole = [c["chunk_id"] for c in page18 if all(k in c["text"] for k in keywords)]
    print(f"  第 18 页片段数: {len(page18)}")
    print(f"  三种评估方法同片命中: {whole if whole else '无（证据被切散）'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="三种 PDF 切片方案对比")
    parser.add_argument("--tools", default="mineru,docling,pymupdf4llm")
    parser.add_argument("--pages", default="all", help="例如 1-20 或 18,19；默认 all")
    args = parser.parse_args()

    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"缺少 {PDF_PATH}（约 4 MB，未入库）。下载地址见 {BASE_DIR / 'README_先读我.md'}。"
        )

    pages = parse_pages_arg(args.pages)
    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    unknown = [t for t in tools if t not in RUNNERS]
    if unknown:
        raise ValueError(f"未知方案: {unknown}，可选 {list(RUNNERS)}")

    for tool in tools:
        chunks, elapsed = RUNNERS[tool](pages)
        save_chunks(tool, chunks, elapsed)
        summarize(tool, chunks)


if __name__ == "__main__":
    main()
