import os
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from dotenv import load_dotenv
from main import run_agent

load_dotenv()

app = FastAPI(
    title="AI Retail Brain API",
    description="API for accessing the AI Retail Agent",
    version="1.0.0",
    docs_url="/docs", 
    openapi_url="/openapi.json"
)

# Define the expected request body
class Question(BaseModel):
    question: str
    password: str
    session_id: str = "default_user"

# Securely load the password from the .env file
API_PASSWORD = os.getenv("API_PASSWORD", "test")

@app.get("/")
def root():
    return {"message": "Retail Brain API is running smoothly!"}

@app.post("/ask")
def ask_ai(data: Question):
    # 1. Professional Security Check
    if data.password != API_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized access. Invalid password."
        )

    # 2. Run the Agent with the Session ID passed correctly
    try:
        # Pass both arguments so the agent uses the correct memory locker
        answer = run_agent(data.question, data.session_id)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"The AI Agent encountered an error: {str(e)}"
        )