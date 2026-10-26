"""Small PDF extraction utility for the bundled sample policies."""

from pathlib import Path

from document_pipeline import pdf_to_text

DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"


def main():
    for filename in ("intern.pdf", "employee.pdf", "manager.pdf", "board.pdf"):
        path = DOCUMENTS_DIR / filename
        if not path.exists():
            print(f"Missing: {filename}")
            continue
        text = pdf_to_text(path)
        print(f"\n{filename}\nExtracted {len(text)} characters\n{text[:300]}")


if __name__ == "__main__":
    main()
