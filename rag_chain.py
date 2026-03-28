import os
from groq import Groq
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ingest import load_and_chunk, DOCUMENTS_DIR

load_dotenv()

# ── Embedding model ───────────────────────────────────────────────────────────
# all-MiniLM-L6-v2 is fast and free — but for technical O&G text,
# BAAI/bge-base-en-v1.5 gives noticeably better semantic matching on
# clause-level retrieval (API 570 numbered sections, regulatory definitions).
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "faiss_index")

# ── Domain system prompts ─────────────────────────────────────────────────────
# Each domain gets a tailored reasoning persona so the LLM answers
# in the right register — regulatory vs engineering vs safety stats.
DOMAIN_PERSONAS = {
    "Engineering": (
        "You are a senior piping and inspection engineer with deep knowledge of "
        "API 570, ASME codes, and piping integrity management. "
        "When answering, reference specific clause numbers where available, "
        "explain the engineering rationale behind requirements, and flag any "
        "conditions that would change the answer (e.g. service class, fluid category)."
    ),
    "Regulatory": (
        "You are a Nigerian upstream oil and gas regulatory specialist with expertise "
        "in DPR/NUPRC regulations and Nigerian petroleum law. "
        "When answering, cite the specific regulation, section, or guideline number. "
        "Distinguish between mandatory requirements and recommended practices. "
        "Note any penalties or compliance deadlines where relevant."
    ),
    "Safety": (
        "You are a safety performance analyst specialising in IOGP metrics and "
        "process safety for upstream oil and gas operations. "
        "When answering, interpret statistical trends carefully — do not overstate "
        "causation from correlations. Reference the specific indicator name and year "
        "from the IOGP data. Contextualise numbers against industry benchmarks."
    ),
    "Mixed": (
        "You are OpsRAG, an expert AI assistant covering oil and gas safety, "
        "piping inspection, regulatory compliance, and performance management. "
        "You have deep knowledge of API 570, Nigerian upstream regulations, and "
        "IOGP safety performance indicators. "
        "When answering, cite specific sources, clause numbers, and page references. "
        "Distinguish between regulatory requirements, engineering standards, and "
        "statistical observations."
    ),
}

# ── Reasoning prompt template ─────────────────────────────────────────────────
REASONING_PROMPT = """
{persona}

You are answering a question using ONLY the document excerpts provided below.
Think step by step before giving your final answer. Structure your reasoning as follows:

<reasoning>
1. Identify what the question is specifically asking
2. Review each source excerpt and note what is directly relevant
3. Identify any gaps, conflicts, or conditions that affect the answer
4. Synthesise the relevant information into a coherent answer
</reasoning>

<answer>
[Your final structured answer here. Be precise and technical.
Cite sources as (Source N: filename, Page X) inline within the answer.
If context is insufficient, say exactly what is missing.]
</answer>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOCUMENT EXCERPTS:
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUESTION: {question}
"""


# ── Vector store ──────────────────────────────────────────────────────────────

def build_or_load_vectorstore() -> FAISS:
    """
    Load FAISS index from disk if it exists, otherwise embed all chunks
    and persist to disk. This way re-runs are instant.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            # BGE models perform better with this prefix for asymmetric retrieval
            "normalize_embeddings": True,
        },
    )

    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}...")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("Index loaded.\n")
    else:
        print("No index found — embedding documents (first run, takes ~1–2 min)...")
        chunks = load_and_chunk()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
        print(f"Index built and saved to {FAISS_INDEX_PATH}\n")

    return vectorstore


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(
    vectorstore: FAISS,
    question: str,
    k: int = 6,
    folder_filter: str | None = None,
) -> list:
    """
    Semantic retrieval with optional folder filter.

    folder_filter: 'engineering' | 'regulatory' | 'safety' | None
    When None, searches across all folders.

    Uses MMR (Maximal Marginal Relevance) instead of pure similarity —
    MMR balances relevance with diversity so you don't get 6 chunks
    from the same paragraph.
    """
    search_kwargs = {
        "k": k,
        "fetch_k": k * 4,   # MMR candidate pool — fetch more, then diversify
        "lambda_mult": 0.65, # 0 = max diversity, 1 = max relevance; 0.65 is a good balance
    }

    if folder_filter:
        search_kwargs["filter"] = {"folder": folder_filter.lower()}

    docs = vectorstore.max_marginal_relevance_search(
        question,
        **search_kwargs,
    )
    return docs


def detect_domain(docs: list) -> str:
    """
    Infer which domain persona to use based on retrieved chunks.
    If chunks span multiple domains, use the Mixed persona.
    """
    domains = {doc.metadata.get("domain", "Mixed") for doc in docs}
    if len(domains) == 1:
        return domains.pop()
    return "Mixed"


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(docs: list) -> str:
    """
    Format retrieved chunks into a numbered context block.
    Surfaces section numbers and table flags so the LLM uses them.
    """
    sections = []
    for i, doc in enumerate(docs, 1):
        meta     = doc.metadata
        source   = meta.get("source", "Unknown")
        page     = meta.get("page", "?")
        domain   = meta.get("domain", "")
        section  = meta.get("section", "")
        is_table = meta.get("is_table", False)

        header_parts = [f"Source {i}: {source} — Page {page} [{domain}]"]
        if section:
            header_parts.append(f"Clause/Section: {section}")
        if is_table:
            header_parts.append("⚠ Table data")

        header = " | ".join(header_parts)
        sections.append(f"[{header}]\n{doc.page_content}")

    return "\n\n---\n\n".join(sections)


# ── Main RAG pipeline ─────────────────────────────────────────────────────────

def answer_question(
    vectorstore: FAISS,
    client: Groq,
    question: str,
    folder_filter: str | None = None,
) -> dict:
    """
    Full semantic RAG pipeline with chain-of-thought reasoning:
    1. Semantic retrieval (MMR) with optional folder filter
    2. Domain detection → persona selection
    3. Reasoning prompt → Groq LLaMA
    4. Return structured result with sources
    """
    # Step 1 — Retrieve
    docs = retrieve(vectorstore, question, k=6, folder_filter=folder_filter)

    if not docs:
        return {
            "question": question,
            "answer":   "No relevant documents found. Try rephrasing or removing the folder filter.",
            "reasoning": "",
            "sources":  [],
            "domain":   "Unknown",
        }

    # Step 2 — Domain + context
    domain  = detect_domain(docs)
    persona = DOMAIN_PERSONAS[domain]
    context = build_context(docs)

    # Step 3 — Build prompt
    prompt = REASONING_PROMPT.format(
        persona=persona,
        context=context,
        question=question,
    )

    # Step 4 — Groq call
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=900,
            temperature=0.2,   # Low temp for factual/technical answers
        )
        raw_output = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq error: {e}")
        return {
            "question": question,
            "answer":   "Error generating response. Please try again.",
            "reasoning": "",
            "sources":  [],
            "domain":   domain,
        }

    # Step 5 — Parse reasoning vs answer
    reasoning, answer = _parse_output(raw_output)

    # Step 6 — Deduplicated sources
    sources = []
    seen    = set()
    for doc in docs:
        meta   = doc.metadata
        label  = f"{meta.get('source','?')} — Page {meta.get('page','?')} [{meta.get('domain','')}]"
        if meta.get("section"):
            label += f" § {meta['section']}"
        if label not in seen:
            seen.add(label)
            sources.append(label)

    return {
        "question": question,
        "answer":   answer,
        "reasoning": reasoning,
        "sources":  sources,
        "domain":   domain,
    }


def _parse_output(raw: str) -> tuple[str, str]:
    """Extract <reasoning> and <answer> blocks from model output."""
    import re

    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", raw, re.DOTALL)
    answer_match    = re.search(r"<answer>(.*?)</answer>",    raw, re.DOTALL)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    answer    = answer_match.group(1).strip()    if answer_match    else raw

    return reasoning, answer


# ── Startup + test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    vectorstore = build_or_load_vectorstore()
    client      = Groq(api_key=os.getenv("GROQ_API_KEY"))

    test_cases = [
        # (question, folder_filter)
        ("What are the required inspection intervals for Class 1 piping systems?", "engineering"),
        ("What are the licensing requirements for upstream petroleum operations in Nigeria?", "regulatory"),
        ("What was the lost time injury frequency rate trend in the 2024 IOGP report?", "safety"),
        ("How does risk-based inspection relate to regulatory compliance requirements?", None),  # cross-domain
    ]

    for question, folder in test_cases:
        label = f"[{folder.upper()}]" if folder else "[ALL DOMAINS]"
        print(f"\n{'='*65}")
        print(f"{label} Q: {question}")
        print("="*65)

        result = answer_question(vectorstore, client, question, folder_filter=folder)

        print(f"\nDomain detected: {result['domain']}")
        if result["reasoning"]:
            print(f"\n── Reasoning ──\n{result['reasoning']}")
        print(f"\n── Answer ──\n{result['answer']}")
        print(f"\n── Sources ──")
        for s in result["sources"]:
            print(f"  • {s}") 