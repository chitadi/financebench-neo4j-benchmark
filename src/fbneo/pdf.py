from __future__ import annotations

from pathlib import Path

from .types import PageText


def find_pdf(pdf_dir: Path, doc_name: str) -> Path | None:
    candidates = [
        pdf_dir / f"{doc_name}.pdf",
        pdf_dir / f"{doc_name}.PDF",
        pdf_dir / doc_name,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    doc_lower = doc_name.lower()
    for path in pdf_dir.glob("*.pdf"):
        if path.stem.lower() == doc_lower:
            return path
    return None


def extract_pages(pdf_path: Path, doc_name: str) -> list[PageText]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required. Install with `pip install -e .`.") from exc

    pages: list[PageText] = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            text = page.get_text("text") or ""
            pages.append(PageText(doc_name=doc_name, page_num=page_num, text=text.strip()))
    return pages

