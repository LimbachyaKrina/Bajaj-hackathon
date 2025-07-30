# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from rag_pipeline import run_pipeline
import uvicorn

app = FastAPI()

class RAGRequest(BaseModel):
    documents: str  # PDF Blob URL
    questions: List[str]

@app.post("/api/v1/hackrx/run")
async def run_rag(request: RAGRequest):
    result = run_pipeline(request.documents, request.questions)
    return {"answers": result}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
