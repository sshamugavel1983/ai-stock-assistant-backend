from sentence_transformers import SentenceTransformer, util

# Load SBERT model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


def rerank_results(query, retrieved_docs):
    """Re-rank retrieved documents based on similarity to query."""
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    doc_embeddings = [sbert_model.encode(doc.page_content, convert_to_tensor=True) for doc in retrieved_docs]

    scores = [util.pytorch_cos_sim(query_embedding, doc_emb).item() for doc_emb in doc_embeddings]

    # Sort by highest similarity score
    ranked_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked_docs[:5]]  # Return top 5 ranked docs
