from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from retriever import get_stock_news
from vector_store import create_vector_store, get_retriever
from ranker import rerank_results
from langchain_core.documents import Document

# Step 1: Fetch stock news
news_articles = get_stock_news("TSLA")
news_docs = [Document(page_content=article["title"], metadata={"source": article["link"]}) for article in news_articles]

# Step 2: Create vector database and retrieve top matches
vector_db = create_vector_store(news_docs)
retriever = get_retriever(vector_db)
retrieved_docs = retriever.get_relevant_documents("Latest Tesla News")

# Step 3: Re-rank retrieved documents
top_docs = rerank_results("Latest Tesla News", retrieved_docs)

# Step 4: Define LLM-based RAG system
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="You are a financial AI. Answer the question using the context:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
)


def get_rag_response(query):
    llm = ChatOpenAI(model_name="gpt-4")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt}
    )
    return qa_chain.run(query)
