import os
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")


def load_pdfs(directory: str = DOCUMENTS_DIR) -> list[Document]:
    """Load all PDFs from the documents directory."""
    documents = []
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {directory}")

    for filename in pdf_files:
        filepath = os.path.join(directory, filename)
        print(f"Loading: {filename}")
        try:
            # Try to decrypt if needed
            import pikepdf
            try:
                with pikepdf.open(filepath, allow_overwriting_input=True) as pdf:
                    pdf.save(filepath)
            except Exception:
                pass  # Not encrypted or already handled

            reader = PdfReader(filepath)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": page_num + 1,
                        }
                    ))
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    print(f"\nLoaded {len(documents)} pages from {len(pdf_files)} documents")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks


def load_and_chunk() -> list[Document]:
    """Full pipeline — load PDFs and return chunks."""
    documents = load_pdfs()
    chunks = chunk_documents(documents)
    return chunks


if __name__ == "__main__":
    chunks = load_and_chunk()
    print(f"\nSample chunk:")
    print(f"Source: {chunks[0].metadata['source']} — Page {chunks[0].metadata['page']}")
    print(f"Content: {chunks[0].page_content[:300]}")