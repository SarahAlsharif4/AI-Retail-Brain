# 🧠 AI-Retail-Brain

End-to-end AI decision system for retail analytics that combines machine learning forecasting, LLM-based reasoning, and API deployment to deliver real-time business insights.

---

## 🚀 Overview

AI-Retail-Brain is a data-driven decision system designed to simulate a real-world business intelligence engine.  
It combines machine learning forecasting with LLM-based reasoning to generate actionable insights for retail operations.

The system processes sales data, predicts future demand, and answers business questions through an intelligent AI agent exposed via an API.

---

## ✨ Features

- 📊 Sales forecasting using Machine Learning (Linear Regression)
- 🧠 AI agent for business Q&A (LLM-powered)
- 📈 Automated insights generation (top/lowest categories, trends)
- ⚙️ Decision system for inventory recommendations
- 🌐 REST API using FastAPI
- 🔐 Protected endpoint (controlled access)
- ☁️ Deployed online

---

## 🏗️ System Architecture

The system consists of multiple integrated components:

1. **Data Processing**
   - Cleans and aggregates sales data
   - Converts daily records into monthly insights

2. **Machine Learning Model**
   - Predicts next month’s sales using Linear Regression

3. **Business Logic Layer**
   - Translates predictions into actionable decisions

4. **AI Agent (LLM)**
   - Answers user questions based on available data
   - Controlled via prompt engineering for accuracy

5. **API Layer**
   - Exposes the system through REST endpoints

---

## 🧪 Example Capabilities

The system can answer questions like:

- "What should we do next month?"
- "Which category performs best?"
- "Who buys beauty products more?"
- "What are our product categories?"

---

## 🔌 API Usage

### Endpoint:
POST /ask

### Request Body

```json
{
  "question": "What should we do next month?",
  "password": "YOUR_ACCESS_PASSWORD"
}
```

> 🔐 **Access Note:**  
> This API is protected with a password to prevent unauthorized usage of the underlying LLM service, which may incur costs.  
> Access credentials are shared only upon request.

---

## 🌍 Live Demo

👉 https://ai-retail-brain.onrender.com/docs

> ⚠️ Note: The service may take a few seconds to respond on the first request due to free hosting (cold start).

---

## 🛠️ Tech Stack

- Python  
- Pandas  
- Scikit-learn  
- FastAPI  
- OpenAI API  
- LangChain   
- Render (Deployment)  

---

## 📁 Project Structure

```
AI-Retail-Brain/
│
├── main.py
├── api.py
├── requirements.txt
├── data/
│   └── sales.csv
└── README.md
```

---

## 🔐 Security & Access

This project uses a simple authentication layer to control access.

- The API is not publicly open to prevent misuse  
- LLM usage may incur costs  
- Access is intentionally restricted  

> 📩 If you would like to test the system, feel free to reach out for access.

---

## 🧠 Purpose of the Project

This project was built to demonstrate:

- End-to-end AI system design  
- Integration of machine learning with LLMs  
- Real-world business decision automation  
- API development and deployment  

It is intended as a **portfolio project** to showcase technical and system design skills.
