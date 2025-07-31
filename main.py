# # main.py
# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import List
# from rag_pipeline import run_pipeline
# import uvicorn

# app = FastAPI()

# class RAGRequest(BaseModel):
#     documents: str  # PDF Blob URL
#     questions: List[str]

# @app.post("/api/v1/hackrx/run")
# async def run_rag(request: RAGRequest):
#     result = run_pipeline(request.documents, request.questions)
#     return {"answers": result}

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# main.py
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
from rag_pipeline import run_pipeline
from dotenv import load_dotenv
import os
import uvicorn

# Load API Key from .env
load_dotenv()
API_KEY = os.getenv("HACKRX_API_KEY")

app = FastAPI()

class RAGRequest(BaseModel):
    documents: str  # PDF Blob URL
    questions: List[str]

@app.post("/hackrx/run")
async def run_rag(
    request: RAGRequest,
    authorization: str = Header(...)
):
    # Check if Authorization header is valid
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Run pipeline
    result = run_pipeline(request.documents, request.questions)
    return {"answers": result}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

