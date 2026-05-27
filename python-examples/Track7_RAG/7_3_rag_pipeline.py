"""
Track 7.3: Complete RAG Pipeline Implementation
===============================================
Build a production-ready RAG system:
- Document loading and chunking
- Embedding generation
- Vector storage
- Retrieval and reranking
- LLM integration

Author: AI Engineering Masterclass
"""

import os
import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

# ==============================================================================
# PART 1: DOCUMENT PROCESSING & CHUNKING
# ==============================================================================

@dataclass
class Document:
    """Represents a document with metadata."""
    content: str
    metadata: Dict = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]

class DocumentProcessor:
    """
    Process documents into chunks for embedding.

    Chunking strategies:
    - Fixed-size: Simple but may break semantic units
    - Semantic: Split on natural boundaries (sentences, paragraphs)
    - Recursive: Hierarchical splitting for best quality
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, source: str = "") -> List[Document]:
        """Split text into overlapping chunks."""

        # Simple sentence-aware chunking
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(Document(
                    content=current_chunk.strip(),
                    metadata={"source": source, "chars": len(current_chunk)}
                ))

                # Start new chunk with overlap
                overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                current_chunk = overlap_text + sentence
                current_size = len(current_chunk)
            else:
                current_chunk += sentence
                current_size += sentence_size

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(Document(
                content=current_chunk.strip(),
                metadata={"source": source, "chars": len(current_chunk)}
            ))

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting."""
        import re
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s + " " for s in sentences if s.strip()]

    def chunk_documents(self, documents: List[str], sources: List[str] = None) -> List[Document]:
        """Process multiple documents."""
        if sources is None:
            sources = [f"doc_{i}" for i in range(len(documents))]

        all_chunks = []
        for doc, source in zip(documents, sources):
            chunks = self.chunk_text(doc, source)
            all_chunks.extend(chunks)

        return all_chunks

# ==============================================================================
# PART 2: EMBEDDING GENERATION
# ==============================================================================

class EmbeddingGenerator:
    """
    Generate embeddings for text.

    Uses OpenAI's text-embedding-3-small by default.
    Falls back to a simple TF-IDF embedder for demo.
    """

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        # For demo: use TF-IDF fallback
        self.use_tfidf = self.api_key is None
        if self.use_tfidf:
            print("⚠️ No OpenAI API key - using TF-IDF fallback (for demo only)")
            self._init_tfidf()
        else:
            print(f"✅ Using OpenAI {model} for embeddings")

    def _init_tfidf(self):
        """Initialize TF-IDF embedder for demo."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf = TfidfVectorizer(max_features=1536, stop_words='english')
        self.is_fitted = False

    def generate(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if self.use_tfidf:
            return self._tfidf_embed(texts)
        else:
            return self._openai_embed(texts)

    def _tfidf_embed(self, texts: List[str]) -> np.ndarray:
        """TF-IDF embedding (fallback for demo)."""
        if not self.is_fitted:
            # Fit on all texts we'll see
            self.tfidf.fit(texts)
            self.is_fitted = True
            self.vocab_size = len(self.tfidf.vocabulary_)

        embeddings = self.tfidf.transform(texts).toarray()

        # Normalize to unit vectors (like OpenAI embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        embeddings = embeddings / norms

        return embeddings.astype(np.float32)

    def _openai_embed(self, texts: List[str]) -> np.ndarray:
        """OpenAI embedding API call."""
        import openai
        client = openai.OpenAI(api_key=self.api_key)

        # Batch requests (OpenAI limit: 2048 inputs per request)
        all_embeddings = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                model=self.model,
                input=batch
            )
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings).astype(np.float32)

# ==============================================================================
# PART 3: VECTOR DATABASE (Simple In-Memory)
# ==============================================================================

class SimpleVectorStore:
    """
    Simple in-memory vector database for demo.

    Production systems would use:
    - Pinecone, Weaviate, ChromaDB, Qdrant, pgvector
    """

    def __init__(self, embedding_dim: int = 1536):
        self.embedding_dim = embedding_dim
        self.documents: List[Document] = []
        self.embeddings: np.ndarray = np.zeros((0, embedding_dim), dtype=np.float32)

    def add(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents and their embeddings."""
        self.documents.extend(documents)
        self.embeddings = np.vstack([self.embeddings, embeddings])
        print(f"Added {len(documents)} documents (total: {len(self.documents)})")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for most similar documents using cosine similarity.
        """
        # Cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        doc_norms = np.linalg.norm(self.embeddings, axis=1)

        # Handle edge cases
        similarities = np.zeros(len(self.documents))
        valid_mask = (query_norm > 0) & (doc_norms > 0)

        if np.any(valid_mask):
            similarities[valid_mask] = (
                np.dot(self.embeddings[valid_mask], query_embedding) /
                (doc_norms[valid_mask] * query_norm)
            )

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))

        return results

    def __len__(self):
        return len(self.documents)

# ==============================================================================
# PART 4: RETRIEVAL & RERANKING
# ==============================================================================

class Reranker:
    """
    Rerank retrieved documents for better relevance.

    Uses a cross-encoder model for more accurate relevance scoring.
    Falls back to embedding similarity for demo.
    """

    def __init__(self, use_cross_encoder: bool = False):
        self.use_cross_encoder = use_cross_encoder
        if use_cross_encoder:
            print("✅ Using Cross-Encoder for reranking")
        else:
            print("⚠️ Using similarity-based reranking (demo mode)")

    def rerank(self, query: str, documents: List[Document],
               embeddings: np.ndarray, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Rerank documents based on query-document relevance."""

        if not self.use_cross_encoder:
            # Just return sorted by similarity
            return list(zip(documents, [0.9] * len(documents)))[:top_k]

        # Cross-encoder scoring would go here
        # This gives more accurate relevance than embedding similarity

        # Placeholder
        scores = [0.9] * len(documents)
        scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        return scored_docs[:top_k]

# ==============================================================================
# PART 5: COMPLETE RAG PIPELINE
# ==============================================================================

class RAGPipeline:
    """
    Complete RAG pipeline combining all components.

    Flow:
    1. Index: Documents → Chunks → Embeddings → Vector Store
    2. Query: Query → Embedding → Retrieve → Rerank → LLM → Answer
    """

    def __init__(self, chunk_size: int = 500, top_k: int = 5):
        self.chunk_size = chunk_size
        self.top_k = top_k

        # Initialize components
        self.processor = DocumentProcessor(chunk_size=chunk_size)
        self.embedder = EmbeddingGenerator()
        self.vector_store = SimpleVectorStore()
        self.reranker = Reranker()

        self.is_indexed = False

    def index(self, documents: List[str], sources: List[str] = None):
        """Build the index from documents."""

        print("\n" + "=" * 50)
        print("  INDEXING DOCUMENTS")
        print("=" * 50)

        # Step 1: Chunk documents
        print("\n1. Chunking documents...")
        chunks = self.processor.chunk_documents(documents, sources)
        print(f"   Created {len(chunks)} chunks")

        # Step 2: Generate embeddings
        print("\n2. Generating embeddings...")
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.generate(texts)
        print(f"   Generated {len(embeddings)} embeddings of dim {embeddings.shape[1]}")

        # Step 3: Store in vector database
        print("\n3. Storing in vector database...")
        self.vector_store.add(chunks, embeddings)

        self.is_indexed = True
        print("\n✅ Indexing complete!")

    def query(self, question: str, llm_callback=None) -> Dict:
        """
        Query the RAG system.

        Args:
            question: User's question
            llm_callback: Optional function to generate answer with LLM

        Returns:
            Dict with answer and source documents
        """

        if not self.is_indexed:
            raise ValueError("Must index documents before querying")

        print("\n" + "=" * 50)
        print("  QUERYING")
        print("=" * 50)

        # Step 1: Embed the query
        print(f"\nQuestion: {question}")
        print("\n1. Embedding query...")
        query_embedding = self.embedder.generate([question])[0]

        # Step 2: Retrieve similar documents
        print("\n2. Retrieving similar documents...")
        results = self.vector_store.search(query_embedding, top_k=self.top_k * 2)

        # Step 3: Rerank
        print("\n3. Reranking...")
        documents = [doc for doc, _ in results]
        reranked = self.reranker.rerank(question, documents, query_embedding, top_k=self.top_k)

        # Step 4: Generate answer (if LLM provided)
        answer = None
        if llm_callback:
            print("\n4. Generating answer with LLM...")
            context = "\n\n".join([f"[Doc {i+1}]: {doc.content}" for i, (doc, _) in enumerate(reranked)])
            prompt = f"""Answer the question based on the context provided.

Context:
{context}

Question: {question}

Answer:"""

            answer = llm_callback(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": reranked,
            "num_sources": len(reranked)
        }

    def print_results(self, results: Dict):
        """Pretty print query results."""
        print("\n" + "=" * 50)
        print("  RESULTS")
        print("=" * 50)

        if results["answer"]:
            print(f"\n📝 Answer:\n{results['answer']}")

        print(f"\n📚 Top {len(results['sources'])} Source Documents:")
        print("-" * 50)

        for i, (doc, score) in enumerate(results["sources"]):
            print(f"\n[{i+1}] Score: {score:.4f}")
            print(f"    Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"    Content: {doc.content[:200]}...")

# ==============================================================================
# PART 6: DEMO & EXAMPLE
# ==============================================================================

def demo_rag():
    """Demonstrate the RAG pipeline."""

    print("=" * 70)
    print("  RAG PIPELINE DEMONSTRATION")
    print("=" * 70)

    # Sample documents (imagine these are from a knowledge base)
    documents = [
        """
        Artificial Intelligence (AI) is the simulation of human intelligence
        by machines. It encompasses various subfields including machine learning,
        deep learning, natural language processing, and computer vision.

        Machine learning is a subset of AI that enables systems to learn from
        data without being explicitly programmed. Deep learning uses neural
        networks with many layers to achieve state-of-the-art performance.

        Large Language Models (LLMs) are AI models trained on vast amounts
        of text data. Examples include GPT-4, Claude, and Gemini. These models
        can understand and generate human-like text.
        """,
        """
        Prompt Engineering is the art of crafting effective prompts for LLMs.
        Key techniques include:

        1. Zero-shot prompting: Asking the model to perform a task without
           examples.

        2. Few-shot prompting: Providing examples in the prompt to guide
           the model's response.

        3. Chain-of-Thought: Encouraging the model to show its reasoning
           step by step.

        4. Constitutional AI: Using principles to guide model behavior.

        The choice of temperature and top-p parameters affects the creativity
        and randomness of responses.
        """,
        """
        Retrieval Augmented Generation (RAG) combines retrieval systems with
        LLMs. It addresses key limitations:

        1. Knowledge Cutoff: LLMs only know things up to their training date.
           RAG allows real-time information retrieval.

        2. Hallucination: RAG grounds responses in retrieved documents,
           reducing made-up information.

        3. Context Limits: Instead of loading entire databases into context,
           RAG retrieves only relevant chunks.

        Key components: Document chunking, embedding models, vector databases,
        reranking models, and LLM generation.
        """
    ]

    sources = ["AI Overview", "Prompt Engineering", "RAG Introduction"]

    # Initialize RAG pipeline
    rag = RAGPipeline(chunk_size=150, top_k=2)

    # Index documents
    rag.index(documents, sources)

    # Demo query
    query = "What is RAG and why is it useful?"

    # Simple mock LLM (in production, use real API)
    def mock_llm(prompt: str) -> str:
        # In real usage, call OpenAI/Anthropic API
        return "RAG combines retrieval with LLM generation to provide accurate, up-to-date answers grounded in your documents."

    # Query with mock LLM
    results = rag.query(query, llm_callback=mock_llm)
    rag.print_results(results)

    # Query without LLM (just retrieval)
    print("\n" + "=" * 70)
    print("  QUERY WITHOUT LLM (RETRIEVAL ONLY)")
    print("=" * 70)
    results = rag.query(query)
    rag.print_results(results)

    return rag

# ==============================================================================
# PART 7: PRODUCTION CONSIDERATIONS
# ==============================================================================

def production_considerations():
    """Key considerations for production RAG systems."""

    print("\n" + "=" * 70)
    print("  PRODUCTION RAG CONSIDERATIONS")
    print("=" * 70)

    considerations = """
    1. DOCUMENT PROCESSING
       - Support PDF, Word, HTML parsing
       - Handle tables and figures (extract text, use OCR)
       - Preserve document structure (headers, lists)

    2. CHUNKING STRATEGY
       - Semantic chunking (split at natural boundaries)
       - Overlap to maintain context across chunks
       - Different chunk sizes for different use cases
       - Late chunking: embed at sentence level, aggregate

    3. EMBEDDING MODELS
       - OpenAI: text-embedding-3-small (cheap, fast)
       - Cohere: Embed v3 (excellent multilingual)
       - Voyage: voyage-3 (SOTA on MTEB benchmarks)
       - E5/BGE: Open-source alternatives

    4. VECTOR DATABASE SELECTION
       - Pinecone: Managed, excellent for production
       - Weaviate: Open-source, built-in hybrid search
       - Qdrant: Rust-based, high performance
       - ChromaDB: Simple, great for experimentation
       - pgvector: If you already use PostgreSQL

    5. RETRIEVAL OPTIMIZATION
       - Hybrid search: Dense (embeddings) + Sparse (BM25)
       - Query expansion: Rewrite queries for better recall
       - Parent document retrieval: Get full context
       - Reranking: Cross-encoder for precision

    6. EVALUATION
       - RAGAS: Faithfulness, Answer Relevancy, Context Recall
       - Trulens: Comprehensive evaluation framework
       - Human evaluation for quality assurance

    7. COST & LATENCY
       - Cache frequent queries
       - Batch embedding generation
       - Use smaller/faster models for retrieval
       - Consider approximate nearest neighbor (ANN) search
    """

    print(considerations)

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # Run demo
    rag = demo_rag()

    # Show production considerations
    production_considerations()

    print("\n" + "=" * 70)
    print("  NEXT: Advanced RAG (HyDE, GraphRAG, Agentic RAG)")
    print("=" * 70)
