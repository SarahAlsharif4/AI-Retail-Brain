from fastapi import FastAPI
from pydantic import BaseModel


from main import run_agent

app = FastAPI(
    title="AI Retail Brain",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

@app.get("/")
def root():
    return {"message": "API is running"}

class Question(BaseModel):
    question: str
    password: str

API_Password = "test"

@app.post("/ask")
def ask_ai(data: Question):
    if data.password != API_Password:
        return {"error": "Unauthorized"}

    answer = run_agent(data.question)
    return {"answer": answer}