"""
AeroLex — Dense Vector Retriever

Performs semantic search using vector similarity.
This is the CORE of RAG — finding relevant chunks for a query.

How Dense Retrieval Works:
1. User query → embed with SAME model as documents
2. Query vector → Qdrant similarity search
3. Top-K most similar chunks returned
4. These chunks = context for LLM

Why "Dense"?
- Dense = every dimension has a value (1024 or 1536 numbers)
- Contrast: Sparse vectors have mostly zeros (BM25 — Phase 3d)
- Dense = semantic meaning captured
- Sparse = exact keyword matching

Official Docs:
- Qdrant Search: https://qdrant.tech/documentation/concepts/search/
- Voyage Embeddings: https://docs.voyageai.com/docs/embeddings
"""

from src.utils.logger import get_logger
from src.utils.exception_handler import handle_exception, RetrievalError
from src.retrieval.qdrant_store import QdrantStore
from config.settings import settings

logger = get_logger(__name__)


class DenseRetriever:
    """
    Semantic search using dense vector embeddings.

    Supports all 4 embedding models:
    - voyage-3-large (primary — best RAG quality)
    - text-embedding-3-small (OpenAI)
    - intfloat/e5-large-v2 (local fallback)
    - BAAI/bge-m3 (multilingual)
    """

    def __init__(
        self,
        collection: str = "aerolex_voyage",
        embedding_model: str = "voyage"
    ):
        """
        Args:
            collection: Qdrant collection to search
            embedding_model: Which embedder to use for queries
                           "voyage", "openai", "e5", "bge"
        """
        self.collection      = collection
        self.embedding_model = embedding_model
        self.store           = QdrantStore(collection=collection)
        self._embedder       = None  # Lazy load

        logger.info(
            f"DenseRetriever initialized | "
            f"Collection: {collection} | "
            f"Embedder: {embedding_model}"
        )

    def _load_embedder(self):
        """Lazy load the appropriate embedder."""
        if self._embedder is not None:
            return self._embedder

        if self.embedding_model == "voyage":
            from src.embeddings.voyage_embedder import VoyageEmbedder
            self._embedder = VoyageEmbedder()

        elif self.embedding_model == "openai":
            from src.embeddings.openai_embedder import OpenAIEmbedder
            self._embedder = OpenAIEmbedder()

        elif self.embedding_model == "e5":
            from src.embeddings.e5_embedder import E5Embedder
            self._embedder = E5Embedder()

        elif self.embedding_model == "bge":
            from src.embeddings.local_embedder import LocalEmbedder
            self._embedder = LocalEmbedder()

        else:
            raise RetrievalError(
                message=f"Unknown embedding model: {self.embedding_model}",
                context="DenseRetriever._load_embedder()"
            )

        return self._embedder

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a search query using the configured model.

        CRITICAL: Must use same model as document embeddings!
        Documents embedded with voyage → query must use voyage.

        Args:
            query: User's search query

        Returns:
            list[float]: Query embedding vector
        """
        embedder = self._load_embedder()

        # Use asymmetric query embedding where supported
        if self.embedding_model == "voyage":
            return embedder.embed_query(query)  # input_type="query"
        elif self.embedding_model == "e5":
            return embedder.embed_query(query)  # "query: " prefix
        else:
            # Symmetric models — embed normally
            result = embedder.client.embeddings.create(
                model="text-embedding-3-small",
                input=[query]
            ) if self.embedding_model == "openai" else None

            if self.embedding_model == "openai":
                return result.data[0].embedding
            else:
                return embedder.model.encode([query], normalize_embeddings=True)[0].tolist()

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.3,
        filters: dict = None,
    ) -> list[dict]:
        """
        Retrieve most relevant chunks for a query.

        Args:
            query: User's search query
            top_k: Number of chunks to retrieve
            score_threshold: Minimum similarity (0-1)
                            0.3 = fairly relevant
                            0.7 = very similar
            filters: Optional metadata filters
                    e.g., {"source": "ecfr", "part_number": "91"}

        Returns:
            list[dict]: Retrieved chunks sorted by relevance
        """
        logger.info(
            f"Dense retrieval | "
            f"Query: '{query[:50]}...' | "
            f"Top-K: {top_k} | "
            f"Threshold: {score_threshold}"
        )

        try:
            # Step 1: Embed query
            query_vector = self.embed_query(query)
            logger.debug(f"Query embedded | Dims: {len(query_vector)}")

            # Step 2: Search Qdrant
            results = self.store.search(
                query_vector=query_vector,
                top_k=top_k,
                score_threshold=score_threshold,
                filters=filters,
            )

            top_score = f"{results[0]['score']:.4f}" if results else "0"
            logger.info(
                f"Dense retrieval complete | "
                f"Results: {len(results)} | "
                f"Top score: {top_score}"
            )

            return results

        except Exception as e:
            raise RetrievalError(
                message=f"Dense retrieval failed for query: {query[:50]}",
                context="DenseRetriever.retrieve()",
                original_error=e
            )

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
        filters: dict = None,
    ) -> dict:
        """
        Retrieve chunks and format as RAG context.

        Returns formatted context ready for LLM prompt.

        Args:
            query: User query
            top_k: Number of chunks
            score_threshold: Minimum similarity
            filters: Metadata filters

        Returns:
            dict: {
                "query": original query,
                "chunks": list of retrieved chunks,
                "context": formatted string for LLM,
                "citations": list of citations
            }
        """
        chunks    = self.retrieve(query, top_k, score_threshold, filters)
        citations = []
        context_parts = []

        for i, chunk in enumerate(chunks):
            citation = chunk.get("citation", "Unknown")
            text     = chunk.get("text", "")
            score    = chunk.get("score", 0)
            citations.append(citation)

            context_parts.append(
                f"[{i+1}] {citation} (relevance: {score:.2f})\n{text}"
            )

        context = "\n\n".join(context_parts)

        return {
            "query":     query,
            "chunks":    chunks,
            "context":   context,
            "citations": citations,
            "num_chunks": len(chunks),
        }


# ── Module-level test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔍 AeroLex — Dense Retrieval Test")
    print("="*60)

    # Test queries — real aviation compliance questions!
    test_queries = [
        "What must a pilot do before beginning a flight?",
        "What are the requirements for operating in Class B airspace?",
        "What are the fuel requirements for flight?",
    ]

    retriever = DenseRetriever(
        collection="aerolex_voyage",
        embedding_model="voyage"
    )

    for query in test_queries:
        print(f"\n{'─'*60}")
        print(f"📝 Query: {query}")
        print(f"{'─'*60}")

        result = retriever.retrieve_with_context(
            query=query,
            top_k=3,
            score_threshold=0.2,
        )

        print(f"Found {result['num_chunks']} relevant chunks:\n")
        for i, chunk in enumerate(result["chunks"]):
            print(f"  [{i+1}] Score: {chunk['score']:.4f}")
            print(f"       Citation: {chunk['citation']}")
            print(f"       Text: {chunk['text'][:150]}...")
            print()

    print("="*60)
    print("✅ Dense Retriever working!")
    print("="*60)