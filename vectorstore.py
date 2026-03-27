"""
vectorstore.py
──────────────
Responsibility: take chunks from ingest.py, embed them, store in FAISS.

Think of this file as the "librarian" —
  ingest.py  : cuts the books into index cards
  vectorstore.py : reads every card, writes a number-summary on the back,
                   then files them in a cabinet (FAISS) you can search instantly.
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ingest import load_and_chunk

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(__file__)
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

# ── Embedding model ───────────────────────────────────────────────────────────
# WHY BGE and not something else?
#   - It's free (runs locally, no API calls)
#   - It's built for asymmetric retrieval: short query vs long document chunk
#   - It understands that "inspection frequency" ≈ "examination intervals"
#     which is exactly the kind of paraphrasing in technical O&G documents
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the embedding model.
    First call downloads it (~400MB), every call after uses the cache.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,  # required for BGE — makes similarity scores comparable
            "batch_size": 32,              # process 32 chunks at a time to avoid memory spikes
        },
    )


def build_vectorstore(chunks: list) -> FAISS:
    """
    Embed all chunks and build a FAISS index in memory.

    What happens here step by step:
      1. Each chunk's text is passed through the BGE model
      2. BGE outputs a vector (list of 768 numbers) for each chunk
      3. FAISS organises all those vectors into an index structure
         optimised for nearest-neighbour search
      4. The chunk text + metadata is stored alongside each vector
         so you can retrieve the original content after search
    """
    embeddings = get_embeddings()

    print(f"Embedding {len(chunks)} chunks — this takes ~1-2 min on first run...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("Embedding complete.")

    return vectorstore


def save_vectorstore(vectorstore: FAISS) -> None:
    """Persist the FAISS index to disk so you don't re-embed on every restart."""
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"Index saved to: {FAISS_INDEX_PATH}")


def load_vectorstore() -> FAISS:
    """
    Load a previously saved FAISS index from disk.
    This is what runs in production — fast, no re-embedding needed.
    """
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,  # safe because we wrote this index ourselves
    )
    print(f"Index loaded from: {FAISS_INDEX_PATH}")
    return vectorstore


def get_vectorstore() -> FAISS:
    """
    Smart loader — the one function everything else should call.

    Logic:
      - If a saved index exists on disk → load it (fast, ~2 seconds)
      - If not → build it from scratch, then save it for next time

    This means:
      First run  : slow (~1-2 min) — embeds everything, saves to disk
      Every other run : fast (~2s) — just loads from disk
    """
    if os.path.exists(FAISS_INDEX_PATH):
        print("Found existing FAISS index — loading from disk...")
        return load_vectorstore()
    else:
        print("No index found — building from scratch (first run)...")
        chunks      = load_and_chunk()
        vectorstore = build_vectorstore(chunks)
        save_vectorstore(vectorstore)
        return vectorstore


def rebuild_vectorstore() -> FAISS:
    """
    Force a full rebuild — use this when you add new PDFs to your documents folder.
    Deletes the old index and builds a fresh one.
    """
    import shutil
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)
        print("Old index deleted.")

    chunks      = load_and_chunk()
    vectorstore = build_vectorstore(chunks)
    save_vectorstore(vectorstore)
    return vectorstore


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Build (or load) the index
    vectorstore = get_vectorstore()

    # Run a few test searches so you can verify retrieval is working
    test_queries = [
        ("inspection intervals Class 1 piping",          "engineering"),
        ("Nigerian petroleum upstream licensing",         "regulatory"),
        ("lost time injury frequency rate 2024",          "safety"),
        ("risk based inspection corrosion monitoring",    None),          # cross-domain
    ]

    print("\n── Retrieval smoke test ─────────────────────────────────────")
    embeddings = get_embeddings()

    for query, folder in test_queries:
        label = f"[{folder.upper()}]" if folder else "[ALL]"
        print(f"\n{label} Query: '{query}'")

        search_kwargs = {"k": 3, "fetch_k": 12, "lambda_mult": 0.65}
        if folder:
            search_kwargs["filter"] = {"folder": folder}

        results = vectorstore.max_marginal_relevance_search(query, **search_kwargs)

        if not results:
            print("  ✗ No results — check folder filter matches metadata exactly")
            continue

        for i, doc in enumerate(results, 1):
            meta = doc.metadata
            print(
                f"  {i}. {meta.get('source','?')} p{meta.get('page','?')} "
                f"[{meta.get('domain','?')}]"
                + (f" §{meta['section']}" if meta.get('section') else "")
            )
            print(f"     {doc.page_content[:120].strip()}...")