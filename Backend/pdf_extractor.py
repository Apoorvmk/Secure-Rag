import pymupdf
from pathlib import Path

DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"

def extract_pdf(file_path):
    text = ""

    with pymupdf.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text() + "\n"

    return text


for filename in ["intern.pdf", "employee.pdf", "manager.pdf", "board.pdf"]:

    file_path = DOCUMENTS_DIR / filename

    text = extract_pdf(file_path)

    print(f"\n{filename}")
    print(f"Extracted {len(text)} characters")
    print(text[:300])