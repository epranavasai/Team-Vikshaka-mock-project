import numpy as np

def retrieve_product(comment, embedder, index, df, top_k=1, threshold=0.42):
    query_embedding = embedder.encode([comment], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    similarities, indices = index.search(query_embedding, top_k)
    best_score = similarities[0][0]

    print(f"Debug: Best similarity score = {best_score}")
    print(f"Debug: Retrieved indices = {df.iloc[indices[0]].to_dict()}")

    if best_score >= threshold:
        return df.iloc[indices[0]].to_dict()
    else:
        return None
