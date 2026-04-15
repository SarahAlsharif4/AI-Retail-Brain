import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# DATA PREP
df = pd.read_csv("data/sales.csv")
df.columns = df.columns.str.strip().str.replace(" ", "_")
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.to_period("M")
monthly_sales = df.groupby("Month")["Quantity"].sum().reset_index()
monthly_sales["Month_num"] = range(len(monthly_sales))

# Remove incomplete month
monthly_sales = monthly_sales.iloc[:-1]

# MODEL PREP
X = monthly_sales[["Month_num"]]
y = monthly_sales["Quantity"]
model = LinearRegression()
model.fit(X, y)
next_month = pd.DataFrame([[len(monthly_sales)]], columns=["Month_num"])
prediction = model.predict(next_month)
predicted_value = prediction[0]
last_month_sales = monthly_sales["Quantity"].iloc[-1]



# ======================
# SAFE INSIGHTS (NO HALLUCINATION)
# ======================
def get_top_category():
    return df.groupby("Product_Category")["Total_Amount"].sum().idxmax()

def get_lowest_category():
    return df.groupby("Product_Category")["Total_Amount"].sum().idxmin()

def get_top_gender():
    return df.groupby("Gender")["Total_Amount"].sum().idxmax()

def get_lowest_gender():
    return df.groupby("Gender")["Total_Amount"].sum().idxmin()

def get_top_combo():
    grouped = df.groupby(["Gender", "Product_Category"])["Total_Amount"].sum()
    top = grouped.idxmax()
    return f"{top[0]} - {top[1]}"

def get_lowest_combo():
    grouped = df.groupby(["Gender", "Product_Category"])["Total_Amount"].sum()
    low = grouped.idxmin()
    return f"{low[0]} - {low[1]}"

def average_price_by_category():
    return df.groupby("Product_Category")["Price_per_Unit"].mean().round(2).to_dict()

def sales_by_gender_and_category():
    result = df.groupby(["Gender", "Product_Category"])["Total_Amount"].sum()
    return {f"{g} - {c}": v for (g, c), v in result.items()}

def monthly_sales_dict():
    return df.groupby(df["Date"].dt.to_period("M"))["Quantity"].sum().to_dict()

def best_month():
    monthly = df.groupby(df["Date"].dt.to_period("M"))["Quantity"].sum()
    return monthly.idxmax(), monthly.max()

def monthly_trend():
    monthly = df.groupby(df["Date"].dt.to_period("M"))["Quantity"].sum()
    return monthly.to_dict()

def last_data_month():
    return df["Date"].dt.to_period("M").max()

# ======================
# SAFE GROWTH CALCULATIONS
# ======================
if last_month_sales == 0:
    growth_percentage = 0
else:
    growth_rate = (predicted_value - last_month_sales) / last_month_sales
    growth_percentage = round(growth_rate * 100, 2)

# ======================
# TOOLS DEFINITION (The Agent's Actions)
# ======================

@tool
def get_sales_forecast_tool():
    """Provides the predicted sales quantity and growth percentage for the upcoming month."""
    return {
        "predicted_sales": round(predicted_value, 2),
        "growth_percentage": f"{growth_percentage}%",
        "last_month_actual_sales": last_month_sales
    }

@tool
def get_business_performance_insights():
    """Provides high-level insights including top/lowest product categories, genders, and best-performing combinations."""
    return {
        "top_category": get_top_category(),
        "lowest_category": get_lowest_category(),
        "top_gender": get_top_gender(),
        "lowest_gender": get_lowest_gender(),
        "best_performing_combo": get_top_combo(),
        "worst_performing_combo": get_lowest_combo()
    }

@tool
def get_pricing_and_historical_trends():
    """Provides average price per category and historical monthly sales trends for deeper analysis."""
    return {
        "average_prices": average_price_by_category(),
        "monthly_historical_sales": monthly_sales_dict(),
        "best_month_recorded": best_month(),
        "last_data_point_month": str(last_data_month())
    }

@tool
def get_demographic_sales_split():
    """Provides a detailed breakdown of total sales amounts grouped by gender and product category."""
    return sales_by_gender_and_category()

# ======================
# PRINT BASIC OUTPUT
# ======================
if __name__ == "__main__":
    print("Prediction:", predicted_value)
    print("Growth %:", growth_percentage)

    # ======================
    # AUTO INSIGHTS
    # ======================
    top_category = get_top_category()
    lowest_category = get_lowest_category()
    top_gender = get_top_gender()
    lowest_gender = get_lowest_gender()
    top_combo = get_top_combo()
    lowest_combo = get_lowest_combo()

    print("\n=== AUTO INSIGHTS ===")
    print("Top Category:", top_category)
    print("Lowest Category:", lowest_category)
    print("Top Gender:", top_gender)
    print("Lowest Gender:", lowest_gender)
    print("Top Combo:", top_combo)
    print("Lowest Combo:", lowest_combo)

    # ======================
# NEW: START THE AGENT (Moved outside so FastAPI can see it)
# ======================
print("\n--- Initializing AI Retail Agent ---")

# 1. Setup the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2. Define Tools list
tools = [
    get_sales_forecast_tool, 
    get_business_performance_insights, 
    get_pricing_and_historical_trends,
    get_demographic_sales_split
]

# 3. Setup Memory and Prompt
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a highly analytical retail business assistant. You must NEVER guess or hallucinate numbers. ALWAYS use your provided tools to check the exact data before answering. If you don't know the exact number from the tools, say 'I don't have that data'."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 4. Create Agent Executor
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    memory=memory, 
    verbose=True, 
    handle_parsing_errors=True
)

# 1. Dictionary to hold chat memory for different sessions
sessions_memory = {}

def run_agent(user_query: str, session_id: str = "default") -> str:
    """
    Runs the agent using a specific memory buffer based on session_id.
    """
    # 2. If this is a new session, create a new memory object for it
    if session_id not in sessions_memory:
        sessions_memory[session_id] = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
    
    # 3. Assign the specific session's memory to the agent executor
    # This ensures the agent "remembers" the correct person
    agent_executor.memory = sessions_memory[session_id]
    
    # 4. Invoke the agent
    try:
        result = agent_executor.invoke({"input": user_query})
        return result["output"]
    except Exception as e:
        return f"Error running agent: {str(e)}"

# ======================
# PRINT BASIC OUTPUT & TERMINAL CHAT (Hidden from FastAPI)
# ======================
if __name__ == "__main__":
    print("Prediction:", predicted_value)
    print("Growth %:", growth_percentage)

    print("\n=== AUTO INSIGHTS ===")
    print("Top Category:", get_top_category())
    print("Lowest Category:", get_lowest_category())
    print("Top Gender:", get_top_gender())
    print("Lowest Gender:", get_lowest_gender())
    print("Top Combo:", get_top_combo())
    print("Lowest Combo:", get_lowest_combo())

    # 5. TEST THE AGENT
    print("\n--- AI AGENT IS LIVE ---")

    # 6. INTERACTIVE CHAT LOOP
    print("\n--- You can now chat with the Retail Brain! (Type 'exit' to stop) ---")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit", "stop"]:
            print("Closing Retail Brain. Goodbye!")
            break
            
        try:
            response = agent_executor.invoke({"input": user_input})
            print(f"\nAgent: {response['output']}")
        except Exception as e:
            print(f"\nError: {e}")