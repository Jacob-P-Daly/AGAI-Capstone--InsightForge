import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'c:\Users\jacob\OneDrive\Desktop\AGAI\capstones\BI_Assistant\insightforge.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def make_source(text):
    lines = text.split('\n')
    result = [line + '\n' for line in lines[:-1]]
    if lines[-1]:
        result.append(lines[-1])
    return result

def code_cell(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": make_source(src)}

def md_cell(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": make_source(src)}

# ── 1. Updated router cell (index 15) ────────────────────────────────────────
router_code = r"""# ── Conversational memory ────────────────────────────────────────────────────
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

conversational_rag = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    verbose=False
)

# ── Keyword lists ─────────────────────────────────────────────────────────────
# Visual keywords are checked FIRST so "show me a bar chart of total sales"
# goes to the chart generator rather than the Pandas agent.
VISUAL_KEYWORDS = [
    "show", "plot", "chart", "graph", "visuali", "draw",
    "bar chart", "line chart", "histogram", "scatter",
    "display a", "display the", "generate a chart", "generate a plot",
]

PANDAS_KEYWORDS = [
    "total", "sum", "average", "mean", "median", "max", "min",
    "count", "how many", "percentage", "percent", "%",
    "standard deviation", "variance", "number of",
    "rank", "top", "bottom", "highest", "lowest",
    "calculate", "compute", "exact",
    "trend", "over time", "over the", "by year", "by month", "by quarter",
    "monthly", "quarterly", "annually", "annual", "yearly",
    "year", "month", "quarter", "2022", "2023", "2024", "2025", "2026",
    "growth", "grew", "increase", "increased", "decrease", "decreased",
    "change", "changed", "evolution", "progress",
    "compare", "comparison", "versus", " vs ", "difference",
    "correlation", "relationship between", "breakdown", "distribution",
    "by region", "by product", "by gender", "by age",
    "which region", "which product",
]

def route_query(question: str) -> str:
    """Route to 'visual', 'pandas', or 'rag'."""
    q_lower = question.lower()
    if any(kw in q_lower for kw in VISUAL_KEYWORDS):
        return "visual"
    if any(kw in q_lower for kw in PANDAS_KEYWORDS):
        return "pandas"
    return "rag"

# ── Unified query function ────────────────────────────────────────────────────
def ask_insightforge(question: str, use_memory: bool = True, verbose: bool = True):
    """
    Main interface. Returns a string for text answers, or (fig, description)
    for chart requests.
    """
    route = route_query(question)

    if verbose:
        route_label = {"visual": "Chart Generator", "pandas": "Pandas Agent",
                       "rag": "RAG Chain"}[route]
        print(f"\n{'='*55}")
        print(f"Query:  {question}")
        print(f"Route:  {route_label}")
        print("=" * 55)

    try:
        if route == "visual":
            fig, description = generate_chart(question)
            if verbose and fig:
                fig.show() if hasattr(fig, 'show') else plt.show()
                print(f"\n{description}")
            return fig, description
        elif route == "pandas":
            result = pandas_agent.invoke(question)
            answer = result["output"]
        else:
            if use_memory:
                result = conversational_rag.invoke({"question": question})
                answer = result["answer"]
            else:
                result = rag_chain.invoke({"query": question})
                answer = result["result"]
    except Exception as e:
        answer = f"Error processing query: {e}"

    if verbose:
        print(f"\nAnswer:\n{answer}")
    return answer

# ── Tests ─────────────────────────────────────────────────────────────────────
ask_insightforge("What is the average sales by region?")
ask_insightforge("Which products tend to sell well?")
"""

# ── 2. New chart generator cell (to insert at index 16) ──────────────────────
chart_gen_code = r"""# ── Cell 7b: Dynamic chart generator ─────────────────────────────────────────
# Interprets a natural-language visualization query with Claude, extracts
# structured chart parameters, then renders a matplotlib figure using the
# project's consistent color maps.

import json as _json

# Color maps (self-contained so this cell works independently of Cell 10)
_PRODUCT_COLORS = {"Widget A": "#4C72B0", "Widget B": "#DD8452",
                   "Widget C": "#55A868", "Widget D": "#C44E52"}
_REGION_COLORS  = {"East": "#8172B2", "North": "#937860",
                   "South": "#DA8BC3", "West": "#CCB974"}
_GENDER_COLORS  = {"Female": "#E377C2", "Male": "#7F7F7F"}
_AGE_COLORS     = {"18\u201330": "#AEC6E8", "31\u201345": "#4C72B0",
                   "46\u201360": "#1C4E8A", "61\u201370": "#0A2444"}
_DIM_COLORS = {
    "Product":         _PRODUCT_COLORS,
    "Region":          _REGION_COLORS,
    "Customer_Gender": _GENDER_COLORS,
    "Age_Group":       _AGE_COLORS,
}

_PARSE_PROMPT = """You are a data visualization parameter extractor.
Given a user query about a sales dataset, return ONLY a JSON object — no explanation.

Dataset columns:
  Product          (Widget A, Widget B, Widget C, Widget D)
  Region           (East, North, South, West)
  Year             (2022–2028, integer)
  Month            (1–12, integer)
  Quarter          (1–4, integer)
  Customer_Gender  (Male, Female)
  Age_Group        (18–30, 31–45, 46–60, 61–70)
  Sales            (numeric, dollars)
  Customer_Satisfaction (0–5 float)

JSON fields to return:
  chart_type   : "bar" | "barh" | "line"
  dimension    : column name to group by (the x-axis or category axis)
  metric       : "Sales" | "Customer_Satisfaction"
  aggregation  : "sum" | "mean"
  filter_col   : column to pre-filter on, or null
  filter_val   : value to filter for, or null
  title        : short descriptive title for the chart

Rules:
- Use "line" when the dimension is Year, Month, or Quarter (time-based).
- Use "barh" when there are 5+ categories or the query says "horizontal".
- Use "bar" for ≤4 categories.
- Default metric to "Sales" unless satisfaction is explicitly mentioned.
- Default aggregation to "sum" for Sales, "mean" for Customer_Satisfaction.

Query: "{question}"
"""

def generate_chart(question: str):
    """
    Parse a visualization query, build a chart, and return (fig, description).
    Returns (None, error_message) if parsing or rendering fails.
    """
    # ── Step 1: ask Claude to extract parameters ──────────────────────────────
    try:
        raw = llm.invoke(_PARSE_PROMPT.format(question=question)).content.strip()
        # Strip markdown fences if present
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.lower().startswith("json"):
                raw = raw[4:]
        params = _json.loads(raw)
    except Exception as e:
        return None, f"Could not interpret the visualization request: {e}"

    # ── Step 2: aggregate the data ────────────────────────────────────────────
    try:
        dim      = params["dimension"]
        metric   = params["metric"]
        agg      = params.get("aggregation", "sum")
        ctype    = params.get("chart_type", "bar")
        title    = params.get("title", f"{agg.capitalize()} {metric} by {dim}")
        filt_col = params.get("filter_col")
        filt_val = params.get("filter_val")

        plot_df = df.copy()
        if filt_col and filt_val is not None:
            plot_df = plot_df[plot_df[filt_col].astype(str) == str(filt_val)]
            if plot_df.empty:
                return None, f"No data found for {filt_col} = {filt_val}."

        grouped = plot_df.groupby(dim, observed=True)[metric].agg(agg)
    except Exception as e:
        return None, f"Data aggregation failed: {e}"

    # ── Step 3: render the chart ──────────────────────────────────────────────
    try:
        color_map = _DIM_COLORS.get(dim, {})

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(9, 4))

        if ctype == "line":
            x_labels = grouped.index.astype(str)
            ax.plot(x_labels, grouped.values,
                    color="#4C72B0", linewidth=2, marker="o", markersize=4)
            ax.fill_between(range(len(grouped)), grouped.values,
                            alpha=0.1, color="#4C72B0")
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels,
                               rotation=45 if len(x_labels) > 6 else 0)
        elif ctype == "barh":
            g_sorted = grouped.sort_values(ascending=True)
            colors   = [color_map.get(str(k), "#4C72B0") for k in g_sorted.index]
            ax.barh(g_sorted.index.astype(str), g_sorted.values,
                    color=colors, edgecolor="white")
        else:  # bar
            colors = [color_map.get(str(k), "#4C72B0") for k in grouped.index]
            ax.bar(grouped.index.astype(str), grouped.values,
                   color=colors, edgecolor="white")
            ax.tick_params(axis="x", rotation=0 if len(grouped) <= 6 else 45)

        ylabel = f"{'Total' if agg == 'sum' else 'Average'} {metric}"
        if metric == "Sales":
            ylabel += " ($)"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel)
        if filt_col:
            ax.set_xlabel(f"(filtered to {filt_col} = {filt_val})")
        plt.tight_layout()

        description = (
            f"{title}. "
            f"Showing {agg} of {metric} grouped by {dim}"
            f"{f', filtered to {filt_col}={filt_val}' if filt_col else ''}."
        )
        return fig, description

    except Exception as e:
        return None, f"Chart rendering failed: {e}"

# ── Quick test ────────────────────────────────────────────────────────────────
print("Chart generator ready.")
print("Example: ask_insightforge('Show me a bar chart of total sales by region')")
"""

# ── 3. Updated Streamlit app (cell 33 → 34 after insertion) ──────────────────
streamlit_app = r"""streamlit_app = """
streamlit_app += '"""'
streamlit_app += r"""
import os
import json as _json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dotenv import load_dotenv
from sklearn.linear_model import Ridge

from langchain_core.documents import Document
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

load_dotenv()

# ── Consistent color maps ─────────────────────────────────────────────────────
PRODUCT_COLORS = {"Widget A": "#4C72B0", "Widget B": "#DD8452",
                  "Widget C": "#55A868", "Widget D": "#C44E52"}
REGION_COLORS  = {"East": "#8172B2", "North": "#937860",
                  "South": "#DA8BC3", "West": "#CCB974"}
GENDER_COLORS  = {"Female": "#E377C2", "Male": "#7F7F7F"}
AGE_COLORS     = {"18\u201330": "#AEC6E8", "31\u201345": "#4C72B0",
                  "46\u201360": "#1C4E8A", "61\u201370": "#0A2444"}
DIM_COLORS = {"Product": PRODUCT_COLORS, "Region": REGION_COLORS,
              "Customer_Gender": GENDER_COLORS, "Age_Group": AGE_COLORS}

st.set_page_config(page_title="InsightForge BI Assistant",
                   page_icon="\U0001f4ca", layout="wide")
st.title("\U0001f4ca InsightForge \u2014 AI Business Intelligence Assistant")
st.markdown("Powered by Claude + LangChain + FAISS")
st.divider()

@st.cache_resource
def load_resources():
    import onnxruntime, fastembed
    df = pd.read_csv("sales_data.csv", parse_dates=["Date"])
    df["Year"]    = df["Date"].dt.year
    df["Month"]   = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Age_Group"] = pd.cut(df["Customer_Age"], bins=[17,30,45,60,70],
                              labels=["18\u201330","31\u201345","46\u201360","61\u201370"])
    api_key = os.getenv("ANTHROPIC_API_KEY")
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001",
                        anthropic_api_key=api_key, temperature=0, max_tokens=2048)
    embeddings = FastEmbedEmbeddings()
    if os.path.exists("faiss_sales_index"):
        vectorstore = FAISS.load_local("faiss_sales_index", embeddings,
                                       allow_dangerous_deserialization=True)
    else:
        def row_to_doc(row):
            return Document(page_content=(
                f"On {row['Date'].date()}, a {row['Customer_Age']}-year-old "
                f"{row['Customer_Gender']} customer purchased {row['Product']} "
                f"in {row['Region']}. Sale: ${row['Sales']}. "
                f"Satisfaction: {row['Customer_Satisfaction']:.2f}."))
        vectorstore = FAISS.from_documents(
            [row_to_doc(r) for _, r in df.iterrows()], embeddings)
    retriever    = vectorstore.as_retriever(search_kwargs={"k": 5})
    rag_chain    = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    pandas_agent = create_pandas_dataframe_agent(
        llm=llm, df=df,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False, allow_dangerous_code=True)
    return df, llm, rag_chain, pandas_agent

@st.cache_data
def build_forecast(_df):
    monthly = (_df.groupby(_df["Date"].dt.to_period("M"))["Sales"]
                  .sum().reset_index())
    monthly.columns = ["YearMonth","Sales"]
    monthly["Date"]  = monthly["YearMonth"].dt.to_timestamp()
    monthly = monthly.sort_values("Date").reset_index(drop=True)
    monthly["t"]     = np.arange(len(monthly))
    monthly["month"] = monthly["Date"].dt.month
    monthly["sin12"] = np.sin(2*np.pi*monthly["month"]/12)
    monthly["cos12"] = np.cos(2*np.pi*monthly["month"]/12)
    monthly["sin6"]  = np.sin(2*np.pi*monthly["month"]/6)
    monthly["cos6"]  = np.cos(2*np.pi*monthly["month"]/6)
    X = monthly[["t","sin12","cos12","sin6","cos6"]].values
    y = monthly["Sales"].values
    model = Ridge(alpha=1.0); model.fit(X, y)
    y_fit     = model.predict(X)
    resid_std = (y - y_fit).std()
    ci_95     = 1.96 * resid_std
    last_date      = monthly["Date"].max()
    forecast_dates = pd.date_range(last_date + pd.DateOffset(months=1),
                                   periods=12, freq="MS")
    ft = np.arange(len(monthly), len(monthly)+12); fm = forecast_dates.month
    Xf = np.column_stack([ft, np.sin(2*np.pi*fm/12), np.cos(2*np.pi*fm/12),
                              np.sin(2*np.pi*fm/6),  np.cos(2*np.pi*fm/6)])
    fv = model.predict(Xf)
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecast": fv.round(2),
                                 "Lower": (fv-ci_95).round(2), "Upper": (fv+ci_95).round(2)})
    return monthly, y_fit, forecast_df, ci_95, model.score(X,y), \
           ("upward" if model.coef_[0]>0 else "downward"), model.coef_[0]

_PARSE_PROMPT = """You are a data visualization parameter extractor.
Return ONLY a JSON object for this query — no explanation.

Dataset columns: Product (Widget A/B/C/D), Region (East/North/South/West),
Year (2022-2028), Month (1-12), Quarter (1-4), Customer_Gender (Male/Female),
Age_Group (18-30/31-45/46-60/61-70), Sales (numeric), Customer_Satisfaction (0-5).

JSON fields: chart_type ("bar"|"barh"|"line"), dimension (column to group by),
metric ("Sales"|"Customer_Satisfaction"), aggregation ("sum"|"mean"),
filter_col (or null), filter_val (or null), title (short string).

Rules: line for time dims (Year/Month/Quarter); barh for 5+ categories;
default metric=Sales, agg=sum for Sales / mean for satisfaction.

Query: "{question}"
"""

def generate_chart(llm, df, question):
    try:
        raw = llm.invoke(_PARSE_PROMPT.format(question=question)).content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.lower().startswith("json"): raw = raw[4:]
        params = _json.loads(raw)
    except Exception as e:
        return None, f"Could not interpret the visualization request: {e}"
    try:
        dim      = params["dimension"]
        metric   = params["metric"]
        agg      = params.get("aggregation", "sum")
        ctype    = params.get("chart_type", "bar")
        title    = params.get("title", f"{agg.capitalize()} {metric} by {dim}")
        filt_col = params.get("filter_col")
        filt_val = params.get("filter_val")
        plot_df  = df.copy()
        if filt_col and filt_val is not None:
            plot_df = plot_df[plot_df[filt_col].astype(str) == str(filt_val)]
            if plot_df.empty:
                return None, f"No data found for {filt_col} = {filt_val}."
        grouped   = plot_df.groupby(dim, observed=True)[metric].agg(agg)
        color_map = DIM_COLORS.get(dim, {})
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(9, 4))
        if ctype == "line":
            x_labels = grouped.index.astype(str)
            ax.plot(x_labels, grouped.values, color="#4C72B0",
                    linewidth=2, marker="o", markersize=4)
            ax.fill_between(range(len(grouped)), grouped.values,
                            alpha=0.1, color="#4C72B0")
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45 if len(x_labels)>6 else 0)
        elif ctype == "barh":
            g = grouped.sort_values(ascending=True)
            ax.barh(g.index.astype(str), g.values,
                    color=[color_map.get(str(k),"#4C72B0") for k in g.index],
                    edgecolor="white")
        else:
            ax.bar(grouped.index.astype(str), grouped.values,
                   color=[color_map.get(str(k),"#4C72B0") for k in grouped.index],
                   edgecolor="white")
            ax.tick_params(axis="x", rotation=0 if len(grouped)<=6 else 45)
        ylabel = f"{'Total' if agg=='sum' else 'Average'} {metric}"
        if metric == "Sales": ylabel += " ($)"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        desc = (f"{title}. Showing {agg} of {metric} by {dim}"
                f"{f', filtered to {filt_col}={filt_val}' if filt_col else ''}.")
        return fig, desc
    except Exception as e:
        return None, f"Chart rendering failed: {e}"

with st.spinner("Loading InsightForge..."):
    df, llm, rag_chain, pandas_agent = load_resources()

VISUAL_KW = ["show", "plot", "chart", "graph", "visuali", "draw",
             "bar chart", "line chart", "histogram", "display a", "display the"]
PANDAS_KW = [
    "total","sum","average","mean","median","max","min","count","how many",
    "percentage","percent","%","number of","rank","top","bottom","highest",
    "lowest","calculate","compute","trend","over time","over the","by year",
    "by month","by quarter","monthly","quarterly","annually","annual","yearly",
    "year","month","quarter","2022","2023","2024","2025","2026","growth","grew",
    "increase","decreased","change","compare","versus"," vs ","difference",
    "breakdown","distribution","by region","by product","by gender","by age",
    "forecast","predict","next year","next month",
]

def route_and_answer(q):
    q_l = q.lower()
    if any(kw in q_l for kw in VISUAL_KW):
        fig, desc = generate_chart(llm, df, q)
        return {"type": "chart", "fig": fig, "text": desc}, "Chart Generator"
    if any(kw in q_l for kw in PANDAS_KW):
        return {"type": "text",
                "text": pandas_agent.invoke(q)["output"]}, "Pandas Agent"
    return {"type": "text",
            "text": rag_chain.invoke({"query": q})["result"]}, "RAG Chain"

with st.sidebar:
    st.header("\U0001f4c8 Dataset Summary")
    st.metric("Total Records",    f"{len(df):,}")
    st.metric("Date Range",       f"{df['Date'].min().year}\u2013{df['Date'].max().year}")
    st.metric("Total Revenue",    f"${df['Sales'].sum():,.0f}")
    st.metric("Avg Satisfaction", f"{df['Customer_Satisfaction'].mean():.2f} / 5")
    st.divider()
    st.subheader("Columns")
    st.write(list(df.columns[:7]))

tab1, tab2, tab3, tab4 = st.tabs([
    "\U0001f4ac Ask InsightForge",
    "\U0001f4ca Visualisations",
    "\U0001f52e Forecast",
    "\U0001f5c2\ufe0f Raw Data",
])

with tab1:
    st.subheader("Ask a business question")
    st.markdown(
        "*Try: 'What is total revenue by region?' or "
        "'Show me a bar chart of average satisfaction by product' or "
        "'Which products tend to sell well?'*"
    )
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("is_chart"):
                st.caption(msg["content"])
            else:
                st.write(msg["content"])

    if prompt := st.chat_input("Ask InsightForge..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result, route = route_and_answer(prompt)
            if result["type"] == "chart":
                if result["fig"]:
                    st.pyplot(result["fig"])
                    plt.close(result["fig"])
                st.caption(result["text"])
                st.caption(f"Answered via: {route}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["text"],
                     "is_chart": True})
            else:
                st.write(result["text"])
                st.caption(f"Answered via: {route}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["text"]})

with tab2:
    st.subheader("Sales Analytics Dashboard")
    plt.style.use("seaborn-v0_8-whitegrid")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Monthly Sales Trend**")
        mp = df.groupby(df["Date"].dt.to_period("M"))["Sales"].sum()
        mp.index = mp.index.to_timestamp()
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(mp.index, mp.values, color="#4C72B0")
        ax.fill_between(mp.index, mp.values, alpha=0.15, color="#4C72B0")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    with col2:
        st.markdown("**Total Sales by Product**")
        prod = df.groupby("Product")["Sales"].sum().sort_values()
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.barh(prod.index, prod.values,
                color=[PRODUCT_COLORS[p] for p in prod.index], edgecolor="white")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Total Revenue by Region**")
        reg = df.groupby("Region")["Sales"].sum().sort_values(ascending=True)
        total_rev = reg.sum()
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.barh(reg.index, reg.values,
                       color=[REGION_COLORS[r] for r in reg.index], edgecolor="white")
        for bar, val in zip(bars, reg.values):
            ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
                    f"{val/total_rev*100:.1f}%", va="center", fontsize=9)
        ax.set_xlim(0, reg.max()*1.2)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    with col4:
        st.markdown("**Avg Satisfaction by Age Group**")
        age_sat = df.groupby("Age_Group", observed=True)["Customer_Satisfaction"].mean()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(age_sat.index.astype(str), age_sat.values,
               color=[AGE_COLORS.get(str(g), "#4C72B0") for g in age_sat.index],
               edgecolor="white")
        ax.set_ylim(0, 5); ax.set_ylabel("Avg Satisfaction")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

with tab3:
    st.subheader("\U0001f52e 12-Month Sales Forecast")
    monthly_hist, y_fit, forecast_df, ci_95, r2, trend_dir, monthly_coef = build_forecast(df)
    col1, col2, col3 = st.columns(3)
    col1.metric("Model R\u00b2", f"{r2:.3f}")
    col2.metric("Trend", trend_dir.capitalize(), delta=f"${abs(monthly_coef):,.0f}/month")
    col3.metric("Predicted next-12-month revenue", f"${forecast_df['Forecast'].sum():,.0f}")
    st.markdown("---")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly_hist["Date"], monthly_hist["Sales"], color="#4C72B0",
            linewidth=1.5, label="Actual")
    ax.plot(monthly_hist["Date"], y_fit, color="#55A868", linewidth=1,
            linestyle="--", alpha=0.7, label="Model fit")
    ax.plot(forecast_df["Date"], forecast_df["Forecast"], color="#DD8452",
            linewidth=2, label="Forecast")
    ax.fill_between(forecast_df["Date"], forecast_df["Lower"], forecast_df["Upper"],
                    color="#DD8452", alpha=0.2, label="95% confidence band")
    ax.axvline(monthly_hist["Date"].max(), color="grey", linestyle=":",
               linewidth=1.5, label="Forecast start")
    ax.set_title("Monthly Sales \u2014 Actual vs 12-Month Forecast", fontsize=13)
    ax.set_ylabel("Monthly Sales ($)")
    ax.legend(fontsize=9); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.markdown("**Monthly forecast breakdown**")
    disp = forecast_df.copy()
    disp["Date"] = disp["Date"].dt.strftime("%B %Y")
    disp.columns = ["Month","Forecast ($)","Lower 95% CI ($)","Upper 95% CI ($)"]
    st.dataframe(disp.set_index("Month"), use_container_width=True)

with tab4:
    st.subheader("Sales Data")
    st.dataframe(df.drop(columns=["Age_Group","YearMonth"], errors="ignore"), height=400)
"""
streamlit_app += '"""'
streamlit_app += r"""

with open("app.py", "w", encoding="utf-8") as f:
    f.write(streamlit_app)
print("Streamlit app written to app.py")
print("Launch with:  streamlit run app.py")
"""

# ── Apply all changes ─────────────────────────────────────────────────────────
# 1. Update router cell (15)
nb['cells'][15]['source'] = make_source(router_code)

# 2. Insert chart generator cell + markdown at index 16
nb['cells'].insert(16, code_cell(chart_gen_code))
nb['cells'].insert(16, md_cell(
    "## Cell 7b: Dynamic Chart Generator\n\n"
    "> Uses Claude to extract chart parameters from a natural-language query,\n"
    "> then renders a matplotlib figure using the project's consistent color maps."
))

# 3. Update Streamlit cell (was 33, now 35 after 2 inserts)
nb['cells'][35]['source'] = make_source(streamlit_app)

with open(r'c:\Users\jacob\OneDrive\Desktop\AGAI\capstones\BI_Assistant\insightforge.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

for idx, label in [(15,'Router'), (16,'MD 7b'), (17,'Chart gen'),
                   (35,'Streamlit')]:
    src = nb['cells'][idx]['source']
    print(f"{label:10s}: {len(src):3d} lines — {repr(src[0][:55])}")
