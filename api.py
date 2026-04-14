from fastapi import FastAPI
from pydantic import BaseModel


from main import run_agent

app = FastAPI()

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