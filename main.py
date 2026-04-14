import pandas as pd
from sklearn.linear_model import LinearRegression
from openai import OpenAI
from langchain_openai import ChatOpenAI
from datetime import datetime


# ======================
# LOAD DATA
# ======================
df = pd.read_csv("data/sales.csv")

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Convert Date
df["Date"] = pd.to_datetime(df["Date"])

# ======================
# MONTHLY MODEL
# ======================
df["Month"] = df["Date"].dt.to_period("M")
monthly_sales = df.groupby("Month")["Quantity"].sum().reset_index()
monthly_sales["Month_num"] = range(len(monthly_sales))

# Remove incomplete month
monthly_sales = monthly_sales.iloc[:-1]

X = monthly_sales[["Month_num"]]
y = monthly_sales["Quantity"]

model = LinearRegression()
model.fit(X, y)

next_month = pd.DataFrame([[len(monthly_sales)]], columns=["Month_num"])
prediction = model.predict(next_month)

predicted_value = prediction[0]
last_month_sales = monthly_sales["Quantity"].iloc[-1]

current_month = datetime.now().strftime("%Y-%m")

# ======================
# SAFE GROWTH
# ======================
if last_month_sales == 0:
    growth_percentage = 0
else:
    growth_rate = (predicted_value - last_month_sales) / last_month_sales
    growth_percentage = round(growth_rate * 100, 2)

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
    # LLM EXPLANATION
    # ======================
    client = OpenAI()

    insight_prompt = f"""
    Explain these business insights:

    Top category: {top_category}
    Lowest category: {lowest_category}
    Top gender: {top_gender}
    Lowest gender: {lowest_gender}
    Top combo: {top_combo}
    Lowest combo: {lowest_combo}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": insight_prompt}]
    )

    print("\nInsight Explanation:")
    print(response.choices[0].message.content)
    
# ======================
# ADVANCED AGENT
# ======================
llm = ChatOpenAI(model="gpt-4o-mini")

# ======================
# AGENT FUNCTION
# ======================
def run_agent(user_question):
    top_category = get_top_category()
    lowest_category = get_lowest_category()
    top_gender = get_top_gender()
    lowest_gender = get_lowest_gender()
    top_combo = get_top_combo()
    lowest_combo = get_lowest_combo()

    agent_prompt = f"""
You are an AI business agent.

User question: {user_question}

Available data:

Sales:
- Predicted sales: {predicted_value:.2f}
- Last month sales: {last_month_sales}
- Growth: {growth_percentage}%
- Best month: {best_month()}

Products:
- Categories: {df["Product_Category"].unique().tolist()}

Pricing:
- Avg price by category: {average_price_by_category()}

Advanced:
- Sales by gender & category: {sales_by_gender_and_category()}

Insights:
- Top category: {top_category}
- Lowest category: {lowest_category}

Historical:
- Monthly sales: {monthly_sales_dict()}

Time context:
- Current system month: {current_month}
- Last available data month: {last_data_month()}

Instructions:

1. Understand the user intent
2. If sales → give decision using growth %
3. If trend → explain only (no action)
4. If product/category → use product data
5. If cross-domain → use gender & category data
6. If question unclear → ask for clarification
7. If data not available → say clearly
8. If question is about past months → use historical monthly data
9. Do NOT use predicted values for past questions
10. Always prioritize inventory decisions over general advice
11. Marketing is secondary, not primary
12. If question is about multiple months → analyze historical trend, not prediction
13. If question is about next month → refer to prediction, not calendar assumption
14. ONLY use relevant data based on the question
15. DO NOT mention prediction or growth unless the question is about future, forecast, or sales performance
16. DO NOT include extra information that was not asked

Rules:
- No hallucination
- Do NOT guess numbers
- If spelling mistakes exist → infer meaning
- Avoid repeating same answers
- Max 2 sentences
- Be precise and practical
- Do NOT assume current month unless provided
- Use only the dataset time for answers
- If unknown → say "Not specified in the data"
"""

    response = llm.invoke(agent_prompt)
    return response.content

    