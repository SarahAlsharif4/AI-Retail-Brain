import os
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from dotenv import load_dotenv

# Import the run_agent function from your main.py file
from main import run_agent

# Load the environment variables from the .env file
load_dotenv()

app = FastAPI(
    title="AI Retail Brain API",
    description="API for accessing the AI Retail Agent",
    version="1.0.0",
    docs_url="/docs", # You can visit localhost:8000/docs to test the API!
    openapi_url="/openapi.json"
)

# Define the expected request body
class Question(BaseModel):
    question: str
    password: str

# Securely load the password from the .env file (defaults to "test" if not found)
API_PASSWORD = os.getenv("API_PASSWORD", "test")

@app.get("/")
def root():
    return {"message": "Retail Brain API is running smoothly!"}

@app.post("/ask")
def ask_ai(data: Question):
    # 1. Professional Security Check (Returns a 401 Unauthorized status)
    if data.password != API_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized access. Invalid password."
        )

    # 2. Run the Agent with Error Handling (Try-Except block)
    try:
        answer = run_agent(data.question)
        return {"answer": answer}
    except Exception as e:
        # If the AI fails (e.g., OpenAI servers are down), return a 500 status instead of crashing
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"The AI Agent encountered an error: {str(e)}"
        )