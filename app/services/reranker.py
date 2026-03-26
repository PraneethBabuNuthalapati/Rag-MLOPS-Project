from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, documents, top_k=3):
    """
    Rerank retrieved documents using cross-encoder
    """

    if not documents:
        return []

    pairs = [(query, doc) for doc in documents]

    scores = reranker_model.predict(pairs)

    # combine docs + scores
    scored_docs = list(zip(documents, scores))

    # sort by score descending
    ranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    # return top_k documents
    return [doc for doc, _ in ranked[:top_k]]