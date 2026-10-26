from pathlib import Path
import yaml

from document_pipeline import pdf_to_text

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
    return pdf_to_text(Path(pdf_path))


def create_okf_document(filename, metadata):

    pdf_path = DOCUMENTS_DIR / filename

    text = extract_text(pdf_path)

    frontmatter = yaml.safe_dump({
        "type": metadata["type"], "title": metadata["title"],
        "description": f"Company policy document for the {metadata['access_level']} access tier.",
        "access_level": metadata["access_level"], "source": filename,
    }, sort_keys=False, allow_unicode=True).strip()
    okf_content = f"---\n{frontmatter}\n---\n\n# {metadata['title']}\n\n{text}\n"

    output_path = OKF_DIR / filename.replace(".pdf", ".md")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(okf_content)

    print(f"Created: {output_path}")


def main():
    OKF_DIR.mkdir(parents=True, exist_ok=True)
    for filename, metadata in DOCUMENTS.items():
        if not (DOCUMENTS_DIR / filename).exists():
            print(f"Missing: {filename}")
            continue
        create_okf_document(filename, metadata)


if __name__ == "__main__":
    main()
