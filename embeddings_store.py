import os
import faiss
import numpy as np
import pickle

class EmbeddingsStore:
    def __init__(self, index_dir="vector_store", dim=384, model=None):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.index_path = os.path.join(index_dir, "faiss.index")
        self.meta_path = os.path.join(index_dir, "metadata.pkl")
        self.dim = dim
        self.model = model
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            self.next_id = len(self.metadata)
        else:
            # Flat L2 index
            self.index = faiss.IndexFlatL2(self.dim)
            self.metadata = []  # list of dicts parallel to index ids
            self.next_id = 0

    def add(self, vector, metadata, text=None):
        v = np.array([vector]).astype('float32')
        self.index.add(v)
        entry = {"metadata": metadata, "text": text}
        self.metadata.append(entry)
        self.next_id += 1

    def search(self, vector, k=5):
        v = np.array([vector]).astype('float32')
        D, I = self.index.search(v, k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append({"score": float(dist), "metadata": meta["metadata"], "text": meta["text"]})
        return results

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
