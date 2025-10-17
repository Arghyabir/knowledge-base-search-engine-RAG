from sentence_transformers import SentenceTransformer
from embeddings_store import EmbeddingsStore

MODEL_NAME = "all-MiniLM-L6-v2"

class Retriever:
    def __init__(self, index_dir="vector_store"):
        self.model = SentenceTransformer(MODEL_NAME)
        self.store = EmbeddingsStore(index_dir=index_dir, model=self.model)

    def retrieve(self, query, k=5):
        q_emb = self.model.encode(query)
        results = self.store.search(q_emb, k=k)
        return results
