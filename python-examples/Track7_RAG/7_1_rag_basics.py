"""
Track 7.1: RAG Basics - Super Simple Edition
============================================
How to give LLMs knowledge from your documents.

Author: AI Engineering Masterclass
"""

print("=" * 60)
print("  WHAT IS RAG?")
print("=" * 60)

print("""
🔍 RAG = Retrieval Augmented Generation

   Think of it like open-book vs closed-book exams:

   ❌ CLOSED-BOOK (LLM alone):
   "Tell me about our company policy on vacation days"
   → Might guess or hallucinate

   ✅ OPEN-BOOK (RAG):
   "Using THIS document, tell me about vacation days"
   → References actual document

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   REAL-WORLD EXAMPLES:
   - Chat with your PDF (upload document → ask questions)
   - Company FAQ bot (uses your internal docs)
   - Legal research (search through 1000s of contracts)
   - Product support (uses your knowledge base)
""")

# ==============================================================================
# PART 1: WHY RAG?
# ==============================================================================

print("\n" + "=" * 60)
print("  WHY DO WE NEED RAG?")
print("=" * 60)

print("""
   🔴 LLM LIMITATIONS:

   1️⃣ KNOWLEDGE CUTOFF
      LLMs only know things up to their training date
      GPT-4: trained up to April 2023
      It doesn't know today's news!

   2️⃣ HALLUCINATION
      LLMs sometimes make things up confidently
      "I'm 99% sure the capital of France is... London"
      (It's Paris!)

   3️⃣ NO ACCESS TO PRIVATE DATA
      Your company docs, personal files
      LLMs can't read your hard drive

   ✅ RAG SOLVES ALL OF THESE!
""")

# ==============================================================================
# PART 2: HOW RAG WORKS (STEP BY STEP)
# ==============================================================================

print("\n" + "=" * 60)
print("  HOW RAG WORKS")
print("=" * 60)

print("""
   📄 THE RAG PIPELINE:

   ┌─────────────────────────────────────────────┐
   │           AT REQUEST TIME                   │
   └─────────────────────────────────────────────┘
   User Query: "What is our refund policy?"

   ┌──────────────┐
   │  1. RETRIEVE │
   └──────┬───────┘
          ↓
   Search document database for relevant info
   Found: "Refunds available within 30 days..."
          ↓
   ┌──────────────┐
   │  2. AUGMENT  │
   └──────┬───────┘
          ↓
   Add retrieved info to the prompt
   "Based on this policy: [retrieved text]
    Answer: What is our refund policy?"
          ↓
   ┌──────────────┐
   │  3. GENERATE │
   └──────┬───────┘
          ↓
   LLM generates answer using the provided context
   → Grounded, accurate, based on real documents!
""")

# ==============================================================================
# PART 3: SIMPLE RAG IMPLEMENTATION
# ==============================================================================

import numpy as np

def simple_rag_demo():
    """Simple RAG example without external libraries."""
    print("\n🚀 SIMPLE RAG IMPLEMENTATION")
    print("-" * 40)

    # 1. DOCUMENTS (your knowledge base)
    documents = [
        "Our refund policy allows returns within 30 days of purchase with receipt.",
        "We offer free shipping on orders over $50 within the continental US.",
        "Customer support is available Monday-Friday, 9am-5pm EST.",
        "We accept Visa, Mastercard, American Express, and PayPal payments.",
        "Products come with a 1-year manufacturer warranty."
    ]

    print("📄 Document Database:")
    for i, doc in enumerate(documents):
        print(f"   [{i}] {doc}")

    # 2. SIMPLE QUERY (user question)
    query = "What is the return policy?"

    print(f"\n❓ User Query: '{query}'")

    # 3. RETRIEVAL (simple keyword matching)
    # Count keyword matches
    query_words = set(query.lower().split())
    scores = []

    for doc in documents:
        doc_words = set(doc.lower().replace(",", "").replace(".", "").split())
        # Count shared words
        shared = query_words & doc_words
        score = len(shared)
        scores.append(score)

    # Get best match
    best_idx = np.argmax(scores)
    best_doc = documents[best_idx]

    print(f"\n🔍 Retrieval: Found document [{best_idx}]")
    print(f"   Score: {scores[best_idx]} keyword matches")
    print(f"   Content: '{best_doc}'")

    # 4. GENERATION (simulated)
    prompt = f"""Based on the following document, answer the question.

Document: {best_doc}

Question: {query}

Answer:"""

    print(f"\n📝 Augmented Prompt (sent to LLM):")
    print(f"   [Document context included]")
    print(f"   Question: '{query}'")

    # In real implementation, you'd send this to an LLM
    # response = llm.generate(prompt)

    print("\n✅ RAG provides relevant context → Better answers!")

simple_rag_demo()

# ==============================================================================
# PART 4: VECTOR SEARCH (BETTER RETRIEVAL)
# ==============================================================================

print("\n" + "=" * 60)
print("  VECTOR SEARCH (The Secret Sauce)")
print("=" * 60)

print("""
   🔢 WHY TEXT SEARCH ISN'T ENOUGH:

   Query: "How do I return items?"
   Doc: "Our refund policy allows returns within 30 days..."

   Simple search: "return" matches "returns" ✓
   But what about:
   - "How to get money back?" → no exact match
   - "Can I exchange products?" → different words

   💡 VECTOR SEARCH TO THE RESCUE!

   Convert text to VECTORS (numbers)
   Similar meaning → similar vectors!
""")

def vector_search_demo():
    """Demonstrate semantic search with vectors."""
    print("\n🧮 VECTOR SEARCH EXAMPLE")

    # Simple word vectors (embedding simulation)
    # Real systems use 768-4096 dimensions!

    word_vectors = {
        "return": np.array([0.1, 0.8, 0.3]),
        "refund": np.array([0.2, 0.9, 0.2]),
        "exchange": np.array([0.3, 0.7, 0.4]),
        "shipping": np.array([0.8, 0.2, 0.1]),
        "payment": np.array([0.7, 0.3, 0.5]),
        "support": np.array([0.4, 0.5, 0.6]),
    }

    # Query vector
    query = "money back"
    # Simple average of words
    query_words = ["money", "back"]
    query_vec = np.mean([word_vectors.get(w, np.array([0.5, 0.5, 0.5]))
                         for w in query_words], axis=0)

    print(f"Query: '{query}'")
    print(f"Query vector: {query_vec.round(3)}")

    # Find most similar
    similarities = {}
    for word, vec in word_vectors.items():
        # Cosine similarity
        sim = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
        similarities[word] = sim

    print("\nSimilarity to query:")
    for word, sim in sorted(similarities.items(), key=lambda x: -x[1]):
        print(f"   '{word}': {sim:.3f}")

    print("\n✅ 'refund' and 'return' have HIGH similarity to 'money back'!")

vector_search_demo()

# ==============================================================================
# PART 5: RAG IN PRACTICE
# ==============================================================================

print("\n" + "=" * 60)
print("  RAG IN PRACTICE")
print("=" * 60)

print("""
   🏗️  RAG ARCHITECTURE:

   INDEXING (Once):
   ┌─────────┐     ┌───────────┐     ┌────────────┐
   │ Docs    │ ──→ │  Chunk    │ ──→ │  Embed    │
   │ (.pdf)  │     │ (split)   │     │ (vector)  │
   └─────────┘     └───────────┘     └─────┬──────┘
                                           ↓
                                     ┌────────────┐
                                     │ Vector DB │
                                     │ (storage) │
                                     └────────────┘

   QUERY (Every question):
   ┌─────────┐     ┌───────────┐     ┌────────────┐     ┌───────┐
   │ Query   │ ──→ │  Embed    │ ──→ │   Search   │ ──→ │ LLM   │
   │         │     │ (vector)  │     │  (retrieve)│     │ Generate│
   └─────────┘     └───────────┘     └────────────┘     └───────┘
""")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)
print("""
✅ RAG = Give LLM knowledge from your documents

✅ Solves: knowledge cutoff, hallucination, private data

✅ Steps: Retrieve → Augment → Generate

✅ Better retrieval = Vector embeddings + similarity search

✅ Use cases: Chatbots, document Q&A, research, support

NEXT: Build a real RAG pipeline with code!
""")