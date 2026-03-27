import os
from groq import Groq
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from ingest import load_and_chunk

load_dotenv()

# ---- Load and index documents once at startup ----
print("Indexing documents...")
CHUNKS = load_and_chunk()
RETRIEVER = BM25Retriever.from_documents(CHUNKS, k=5)
print(f"Index ready — {len(CHUNKS)} chunks available\n")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def retrieve(question: str) -> list:
    """Retrieve top 5 relevant chunks for a question."""
    return RETRIEVER.invoke(question)


def build_context(docs: list) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    sections = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", "?")
        sections.append(
            f"[Source {i}: {source} — Page {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(sections)


def answer_question(question: str) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks
    2. Build context
    3. Call Groq LLM with context + question
    4. Return answer + sources
    """
    # Step 1 — Retrieve
    docs = retrieve(question)

    if not docs:
        return {
            "question": question,
            "answer":   "I could not find relevant information in the available documents.",
            "sources":  [],
        }

    # Step 2 — Build context
    context = build_context(docs)

    # Step 3 — Generate answer
    prompt = f"""You are OpsRAG, an expert AI assistant for oil and gas safety, inspection, and operations.

You have been provided with relevant excerpts from industry documents. Use ONLY this information to answer the question.
Always cite the source document and page number in your answer.
Be precise and technical — your users are engineers and safety professionals.
If the provided context does not contain enough information to answer the question fully, say so clearly.
Never make up information that is not in the context.

CONTEXT:
{context}

QUESTION:
{question}

Provide a clear, structured answer with source citations."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq error: {e}")
        answer = "I encountered an error generating a response. Please try again."

    # Step 4 — Extract clean sources
    sources = list({
        f"{doc.metadata.get('source', 'Unknown')} — Page {doc.metadata.get('page', '?')}"
        for doc in docs
    })

    return {
        "question": question,
        "answer":   answer,
        "sources":  sources,
    }


# ---- Test locally ----
if __name__ == "__main__":
    test_questions = [
        "What are the key safety performance indicators for oil and gas operations?",
        "What is risk-based inspection and when should it be applied?",
        "What are the requirements for offshore pipeline integrity management?",
    ]

    for question in test_questions:
        print(f"\nQ: {question}")
        print("-" * 60)
        result = answer_question(question)
        print(f"A: {result['answer']}")
        print(f"\nSources:")
        for s in result['sources']:
            print(f"  - {s}")
        print("=" * 60)