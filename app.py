# ---------------------------------------------------------------------
# app.py — EV Insights AI Analyzer (Definitive Final Version)
# ---------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------------------
# App Configuration
# ---------------------------
st.set_page_config(
    page_title="EV Insights AI Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)
# The custom theme from .streamlit/config.toml will be applied

# ---------------------------
# Helper: Typewriter Effect
# ---------------------------
def typewriter(text, speed=0.01):
    placeholder = st.empty()
    typed_text = ""
    for char in text:
        typed_text += char
        placeholder.markdown(typed_text)
        time.sleep(speed)

# ---------------------------
# Data Loading and Preparation
# ---------------------------
@st.cache_data(show_spinner="Loading and preparing data...")
def load_data(path="IEA-EV-dataEV salesHistoricalCars.csv"):
    if not os.path.exists(path): return None, None
    df = pd.read_csv(path)
    df_sales = df[df["parameter"] == "EV sales"].copy()
    df_analysis = df_sales[~df_sales['region'].isin(['World', 'Rest of the world'])].copy()
    df_predict_options = df_sales.copy()
    return df_analysis, df_predict_options

df_analysis_data, df_predict_options_data = load_data()

# ---------------------------
# Predictive Model Loading
# ---------------------------
@st.cache_resource
def load_predictive_model_artifacts():
    MODEL_DIR = "models"
    paths = [os.path.join(MODEL_DIR, f) for f in ["model.pkl", "scaler.pkl", "encoders.pkl", "poly.pkl"]]
    if not all(os.path.exists(p) for p in paths): return None
    try:
        return {
            "model": joblib.load(paths[0]), "scaler": joblib.load(paths[1]),
            "encoders": joblib.load(paths[2]), "poly": joblib.load(paths[3])
        }
    except Exception: return None

predictive_artifacts = load_predictive_model_artifacts()

# ---------------------------
# AI Brain for Chatbot (RESTORED and IMPROVED)
# ---------------------------
def get_offline_response(query: str, df: pd.DataFrame):
    if df is None: return "The data could not be loaded."
    q_lower = query.lower()

    # --- [IMPROVED] Intent 1: Highest Sales Year ---
    # This block is now more specific and must come BEFORE the general "highest" check.
    if any(word in q_lower for word in ["highest", "top", "best", "peak"]) and "year" in q_lower:
        top_year_data = df.groupby('year')['value'].sum()
        top_year = top_year_data.idxmax()
        top_sales = top_year_data.max()
        return f"Based on the analysis, the year with the highest EV sales was **{top_year}**, with approximately **{int(top_sales):,}** vehicles sold globally."

    # --- Intent 2: Top N Regions ---
    top_n_match = re.search(r"top (\d+)", q_lower)
    if any(word in q_lower for word in ["highest", "top", "biggest", "best", "largest"]):
        n = int(top_n_match.group(1)) if top_n_match else 1
        top_regions = df.groupby("region")["value"].sum().nlargest(n)
        if n == 1: return f"The region with the highest total EV sales is **{top_regions.index[0]}** with **{int(top_regions.iloc[0]):,}** vehicles sold."
        response = f"Here are the Top {n} regions by total EV sales:\n"
        for i, (region, total) in enumerate(top_regions.items()): response += f"\n{i+1}. **{region}**: {int(total):,} vehicles"
        return response
    
    # --- Intent 3: Current or Latest Sales ---
    elif any(word in q_lower for word in ["current", "latest"]):
        latest_year = df['year'].max()
        latest_sales = df[df['year'] == latest_year]['value'].sum()
        return f"The most current sales information in the dataset is for **{latest_year}**, where total sales were **{int(latest_sales):,}** vehicles."

    # --- [RESTORED] Intent 4: General EV Information ---
    elif any(phrase in q_lower for phrase in ["what is ev", "tell me about ev", "about evs"]):
        return (
            "An **Electric Vehicle (EV)** uses one or more electric motors for power, running on rechargeable batteries instead of a gasoline engine.\n\n"
            "Key benefits include zero tailpipe emissions, a quieter ride, and lower running costs."
        )

    # --- Intent 5: Sales for a Specific Year ---
    year_match = re.search(r"(\b20\d{2}\b)", q_lower)
    if year_match:
        year = int(year_match.group(1))
        year_sales = df[df['year'] == year]['value'].sum()
        if year_sales > 0: return f"Total EV sales in **{year}** were approximately **{int(year_sales):,}** vehicles."
        return f"I could not find any sales data for the year {year}."

    # --- Intent 6: Sales for a Specific Region ---
    for region in df['region'].unique():
        if region.lower() in q_lower:
            region_sales = df[df['region'] == region]['value'].sum()
            return f"Total historical sales for **{region}** are **{int(region_sales):,}** vehicles."
            
    # --- Fallback ---
    return "I'm not sure how to answer that. Try asking questions like 'What are the top 5 regions?' or 'Which year had the highest sales?'"

# ---------------------------
# Reusable Plotting Functions (No changes)
# ---------------------------
def plot_global_trend(df):
    fig, ax = plt.subplots()
    df_trend = df[df["parameter"] == "EV sales"].groupby("year")["value"].sum().reset_index()
    sns.lineplot(data=df_trend, x="year", y="value", marker="o", ax=ax)
    ax.set_title("EV Sales Trend")
    return fig

def plot_top_regions(df):
    fig, ax = plt.subplots()
    top_10 = df[df["parameter"] == "EV sales"].groupby("region")["value"].sum().nlargest(10)
    sns.barplot(x=top_10.values, y=top_10.index, ax=ax)
    ax.set_title("Top 10 Regions by Total Sales")
    return fig

# ---------------------------
# Scikit-learn Prediction Function (No changes)
# ---------------------------
def perform_prediction(artifacts, region, mode, powertrain, category, year):
    if not artifacts: return 0
    model, scaler, enc, poly = artifacts.values()
    def t(k, v):
        try: return int(enc[k].transform([v])[0])
        except (ValueError, KeyError): return 0
    X_initial = np.array([[t("region", region), t("mode", mode), t("powertrain", powertrain), t("category", category), year]])
    return float(np.expm1(model.predict(scaler.transform(poly.transform(X_initial)))[0]))

# ---------------------------
# --- Professional UI Rendering --- (No changes to the UI)
# ---------------------------
with st.sidebar:
    st.image("images/logo.png", width=150)
    st.header("Navigation")
    mode = st.radio("Choose a mode:", ["Home", "AI Chat Assistant", "Predict Sales", "Visualize Data", "Analyze Your CSV"])
    st.divider()
    st.write("Predictive Model Status:")
    if predictive_artifacts:
        st.success("Loaded Successfully ✅")
    else:
        st.warning("Missing Files ❌")

if mode == "Home":
    st.title("⚡ EV Insights AI Analyzer")
    st.subheader("A Smart, Interactive Analytics Tool for EV Sales Data")
    with st.container(border=True):
        st.markdown("### Welcome!\nThis application provides a comprehensive suite of tools to analyze and predict Electric Vehicle (EV) sales.")
    with st.container(border=True):
        st.subheader("Dataset at a Glance")
        col1, col2, col3 = st.columns(3)
        if df_analysis_data is not None:
            col1.metric("Total Regions Analyzed", df_analysis_data['region'].nunique())
            col2.metric("Data Available Until", df_analysis_data['year'].max())
            col3.metric("Total Sales Records", f"{df_analysis_data.shape[0]:,}")
        else:
            col1.metric("Total Regions Analyzed", 0), col2.metric("Data Available Until", "N/A"), col3.metric("Total Sales Records", 0)

elif mode == "AI Chat Assistant":
    st.header("🤖 AI Chat Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you analyze the EV sales data today?"}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = get_offline_response(prompt, df_analysis_data)
            typewriter(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

elif mode == "Predict Sales":
    st.header("🔮 Predict Future EV Sales")
    if not predictive_artifacts or df_predict_options_data is None:
        st.error("Predictive model artifacts are not loaded.")
    else:
        with st.container(border=True):
            st.subheader("Select Prediction Parameters")
            col1, col2 = st.columns(2)
            with col1:
                regions = sorted(df_predict_options_data["region"].unique())
                region = st.selectbox("Region", regions)
                modes = sorted(df_predict_options_data["mode"].unique())
                mode_in = st.selectbox("Mode", modes)
            with col2:
                powers = sorted(df_predict_options_data["powertrain"].unique())
                power = st.selectbox("Powertrain", powers)
                cats = sorted(df_predict_options_data["category"].unique())
                cat = st.selectbox("Category", cats)
            year = st.slider("Year for Prediction", min_value=2024, max_value=2040, value=2025)
        
        if st.button("Generate Prediction", type="primary", use_container_width=True):
            with st.spinner("Calculating..."):
                pred = perform_prediction(predictive_artifacts, region, mode_in, power, cat, year)
                st.metric(label=f"Predicted Sales for {region} in {year}", value=f"{int(pred):,}")

elif mode == "Visualize Data":
    st.header("📊 Historical Data Visualizations")
    if df_analysis_data is not None:
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True): st.pyplot(plot_global_trend(df_analysis_data))
        with col2:
            with st.container(border=True): st.pyplot(plot_top_regions(df_analysis_data))
    else:
        st.error("Data could not be loaded for visualization.")

elif mode == "Analyze Your CSV":
    st.header("📂 Analyze Your Own EV Sales Data")
    st.info("Upload a CSV file to get an instant analysis. Your file must contain `region`, `year`, `value`, and `parameter` columns.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df_user = pd.read_csv(uploaded_file)
            REQUIRED_COLUMNS = ['region', 'year', 'value', 'parameter']
            if not all(col in df_user.columns for col in REQUIRED_COLUMNS):
                st.error(f"Your CSV is missing required columns.")
            else:
                st.success("File processed successfully!")
                df_user['year'] = pd.to_numeric(df_user['year'], errors='coerce')
                df_user['value'] = pd.to_numeric(df_user['value'], errors='coerce')
                df_user.dropna(subset=['year', 'value'], inplace=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    with st.container(border=True): st.pyplot(plot_global_trend(df_user))
                with col2:
                    with st.container(border=True): st.pyplot(plot_top_regions(df_user))
        except Exception as e:
            st.error(f"An error occurred: {e}.")