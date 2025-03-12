from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


def create_vector_store(documents):
    """Convert documents to FAISS embeddings."""
    embedding_function = OpenAIEmbeddings()

    if not documents:
        raise ValueError("No documents available for FAISS!")

    vector_db = FAISS.from_documents(documents, embedding=embedding_function)
    return vector_db


def get_retriever(vector_db):
    """Retrieve similar documents from FAISS."""
    return vector_db.as_retriever(search_kwargs={"k": 10})  # Retrieve top 10 for re-ranking
