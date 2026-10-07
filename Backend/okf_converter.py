import pymupdf
from pathlib import Path

DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"
OKF_DIR = DOCUMENTS_DIR / "okf"

DOCUMENTS = {
    "intern.pdf": {
        "type": "Company Policy",
        "title": "Intern Policy",
        "access_level": "intern"
    },
    "employee.pdf": {
        "type": "Company Policy",
        "title": "Employee Policy",
        "access_level": "employee"
    },
    "manager.pdf": {
        "type": "Company Policy",
        "title": "Manager Policy",
        "access_level": "manager"
    },
    "board.pdf": {
        "type": "Company Policy",
        "title": "Board Policy",
        "access_level": "board"
    }
}


def extract_text(pdf_path):
    text = ""

    with pymupdf.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text() + "\n"

    return text.strip()


def create_okf_document(filename, metadata):

    pdf_path = DOCUMENTS_DIR / filename

    text = extract_text(pdf_path)

    okf_content = f"""---
type: {metadata["type"]}
title: {metadata["title"]}
description: Company policy document for the {metadata["access_level"]} access tier.
access_level: {metadata["access_level"]}
source: {filename}
---

# {metadata["title"]}

{text}
"""

    output_path = OKF_DIR / filename.replace(".pdf", ".md")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(okf_content)

    print(f"Created: {output_path}")


# Create OKF directory
OKF_DIR.mkdir(exist_ok=True)

# Convert PDFs
for filename, metadata in DOCUMENTS.items():

    pdf_path = DOCUMENTS_DIR / filename

    if not pdf_path.exists():
        print(f"Missing: {filename}")
        continue

    create_okf_document(filename, metadata)