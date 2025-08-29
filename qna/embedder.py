import numpy as np
import faiss # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore

def build_index(df):
    embedder = SentenceTransformer('all-distilroberta-v1')
    embeddings = embedder.encode(df['description'].tolist(), convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return embedder, index
