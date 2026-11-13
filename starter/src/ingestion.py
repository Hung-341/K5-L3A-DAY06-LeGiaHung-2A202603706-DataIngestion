"""Read local PDFs and normalize extracted text."""
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while retaining paragraph breaks."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(
        re.sub(r"[\t\f\v ]+", " ", line).strip()
        for line in text.split("\n")
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical PDF page and retain its source and page number."""
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy PDF: {path}")

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ValueError(f"Không đọc được PDF {path}: {exc}") from exc

    pages = []
    for page_number, pdf_page in enumerate(reader.pages, start=1):
        try:
            text = normalize_text(pdf_page.extract_text() or "")
        except Exception as exc:
            raise ValueError(
                f"Lỗi trích xuất {path}, trang PDF {page_number}: {exc}"
            ) from exc

        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
            )
        )

    return pages