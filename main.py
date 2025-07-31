from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
from rag_pipeline import run_pipeline
from dotenv import load_dotenv
import os
import uvicorn

# Load environment variables
load_dotenv()
API_KEY = os.getenv("HACKRX_API_KEY")
IS_PRODUCTION = os.getenv("ENVIRONMENT") == "production"

# In development mode, ensure API_KEY is set
if not IS_PRODUCTION and not API_KEY:
    raise ValueError("HACKRX_API_KEY environment variable not set in development mode.")

app = FastAPI()

class RAGRequest(BaseModel):
    documents: str  # PDF Blob URL
    questions: List[str]

@app.post("/hackrx/run")
async def run_rag(
    request: RAGRequest,
    authorization: Optional[str] = Header(None)
):
    # Check if Authorization header is missing
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Bearer token"
        )

    # Check if header starts with "Bearer "
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Bearer token"
        )

    # Extract token
    token = authorization.split("Bearer ")[1].strip()

    # Check for empty token
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Empty Bearer token"
        )

    # In development mode, validate token against API_KEY
    if not IS_PRODUCTION and token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid API key"
        )

    # Run pipeline
    try:
        result = run_pipeline(request.documents, request.questions)
        return {"answers": result}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
