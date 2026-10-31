
from io import BytesIO

import pymupdf
from docx import Document


def extract_pdf_text(content: bytes) -> dict:
    """Extract readable text from each PDF page."""

    with pymupdf.open(stream=content, filetype="pdf") as pdf:
        if pdf.is_encrypted:
            raise ValueError("Password-protected PDFs are not supported.")

        total_pages = len(pdf)
        pages = []

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text,
                })

    if not pages:
        raise ValueError(
            "No readable text found. The PDF may require OCR."
        )

    return {
        "total_pages": total_pages,
        "pages_with_text": len(pages),
        "extracted_content": pages,
    }


def extract_docx_text(content: bytes) -> dict:
    """Extract text from Word document paragraphs and tables."""

    document = Document(BytesIO(content))
    sections = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            sections.append({
                "text": text,
                "source": "paragraph",
            })

    for table_number, table in enumerate(document.tables, start=1):
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            text = " | ".join(cells)

            if text.replace("|", "").strip():
                sections.append({
                    "text": text,
                    "source": f"table_{table_number}",
                })

    if not sections:
        raise ValueError("No readable text found in the DOCX document.")

    return {
        "sections_with_text": len(sections),
        "extracted_content": sections,
    }


def extract_document(filename: str, content: bytes) -> dict:
    """Choose the correct text extractor based on file extension."""

    extension = filename.rsplit(".", 1)[-1].lower()

    if extension == "pdf":
        return {
            "file_type": "pdf",
            **extract_pdf_text(content),
        }

    if extension == "docx":
        return {
            "file_type": "docx",
            **extract_docx_text(content),
        }

    raise ValueError("Only PDF and DOCX files are supported.")
