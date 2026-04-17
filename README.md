# 🧠 AI-Retail-Brain

An end-to-end autonomous AI decision system for retail analytics. This project combines machine learning forecasting, an LLM-based LangChain Agent, and a secure FastAPI backend to deliver real-time, data-driven business insights.

---

## 🚀 Overview

AI-Retail-Brain is designed to simulate a real-world business intelligence engine. It goes beyond basic data analysis by integrating an **Autonomous AI Agent** capable of reasoning, selecting analytical tools, and answering complex business questions dynamically. 

The system processes retail sales data, predicts future demand using predictive modeling, and exposes its intelligence through a secure REST API.

---

## ✨ Features

- **🤖 Autonomous AI Agent:** Powered by LangChain and OpenAI, capable of dynamic tool-calling.
- **🧠 Multi-Session Memory:** Uses a session-based architecture to remember user context and history within a single conversation.
- **📊 Predictive Analytics:** Machine Learning sales forecasting (Linear Regression).
- **📈 Automated Insights:** On-demand generation of top/lowest performing categories, pricing trends, and demographic splits.
- **🌐 RESTful API:** Robust backend built with FastAPI, including error handling and documentation.
- **🔐 Secure Architecture:** Environment variable protection (`.env`) and password-gated API endpoints.
- **☁️ Cloud Deployed:** Live and accessible via Render.

---

## 🏗️ System Architecture

The system consists of interconnected layers designed for scalability and accuracy:

1. **Data Processing Layer (Pandas)**
   - Cleans, aggregates, and transforms daily transaction records into actionable monthly metrics.
2. **Predictive Modeling Layer (Scikit-learn)**
   - Utilizes Linear Regression to predict next month’s sales volume and calculate growth trajectories.
3. **Agentic AI Layer (LangChain & OpenAI)**
   - Acts as the "Brain." Uses an `AgentExecutor` combined with a `sessions_memory` dictionary.
   - This "Locker System" maps unique `session_id`s to specific `ConversationBufferMemory` objects, ensuring the agent maintains state for multiple users simultaneously without mixing data.
4. **API & Deployment Layer (FastAPI & Render)**
   - Exposes the agent's capabilities through secure, documented REST endpoints.

---

## 🧪 Example Capabilities

You can ask the Retail Brain questions such as:
- *"Based on the growth percentage, should we increase our inventory for next month?"*
- *"What is our best-selling product category?"*
- *"What were the lowest monthly sales recorded?"*
- *"Give me a demographic split of our sales."*

---

## 🔌 API Usage

### Endpoint
`POST /ask`

### Request Body

```json
{
  "question": "What is my name?",
  "password": "YOUR_ACCESS_PASSWORD",
  "session_id": "sarah_session_01"
}
```

> 🔐 **Access Note:** This API is password-protected to prevent unauthorized usage of the underlying OpenAI API and manage costs. Access credentials are provided upon request.

> 💡 **Tip:** Use the same `session_id` to maintain a conversation. If you change the ID or the server restarts, the Agent will start with a fresh memory.

---

## 🧠 Memory & Persistence

To demonstrate session management without the overhead of external databases, this project implements **In-Memory Storage**:
- **Session Isolation:** Each user is assigned a private memory buffer based on their `session_id`.
- **Volatility Note:** Because the system uses Render's free tier, memory is cleared if the server "spins down" (after 15 minutes of inactivity) or during new deployments.
  
---

## 🌍 Live Demo

👉 **[Test the API via Swagger UI](https://ai-retail-brain.onrender.com/docs)**

> ⚠️ *Note: The service is hosted on a free tier. It may take 50-80 seconds to wake up (cold start) on your first request.*

---

## 🛠️ Tech Stack

- **Core:** Python
- **AI & NLP:** LangChain (Agents, Tools, Memory), OpenAI API (gpt-4o-mini)
- **Data & ML:** Pandas, Scikit-learn
- **Backend:** FastAPI, Uvicorn, Python-dotenv
- **Deployment:** Render

---

## 📊 Dataset

This project utilizes a publicly available retail sales dataset from Kaggle:
🔗 [Retail Sales Dataset](https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset)

The dataset contains structured transaction data, including:
- Product categories (Electronics, Clothing, Beauty)
- Customer demographics (Age, Gender)
- Pricing and quantity metrics

*Note: This dataset is used strictly for demonstration and educational purposes.*

---


## 📁 Project Structure

```text
AI-Retail-Brain/
│
├── api.py                 # FastAPI application and endpoint routing
├── main.py                # Core ML logic, Tool definitions, and LangChain Agent
├── requirements.txt       # Production dependencies
├── .gitignore             # Security exclusions
├── .env                   # Template for required environment variables
├── data/
│   └── sales.csv          # Retail dataset
└── README.md
