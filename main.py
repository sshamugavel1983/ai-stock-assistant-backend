from fastapi import FastAPI
from rag_pipeline import get_rag_response

app = FastAPI()

@app.get("/query")
def query_stock_assistant(question: str):
    response = get_rag_response(question)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
