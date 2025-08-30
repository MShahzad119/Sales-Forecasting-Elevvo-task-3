# =========================
# Walmart Sales Forecasting Dashboard (Professional Version)
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px

sns.set(style="whitegrid")

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Walmart Sales Forecast Dashboard",
    layout="wide",
    page_icon="📊"
)

st.title("📈 Walmart Sales Forecasting Dashboard")
st.markdown(
    "Interactive dashboard to visualize predicted weekly sales, trends, and multi-store insights. "
    "Developed as **Elevvo Task 3** internship project."
)

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_model.pkl")  # Ensure model file exists in repo
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# =========================
# Load Data
# =========================
@st.cache_data
def load_data(file):
    if file is not None:
        try:
            df = pd.read_csv(file)
        except:
            try:
                df = pd.read_excel(file, engine="openpyxl")
            except:
                st.error("❌ Unsupported file format. Upload CSV or XLSX.")
                return None
        # Convert Date column
        date_col = None
        for col in df.columns:
            if "date" in col.lower():
                date_col = col
                break
        if date_col is None:
            st.error("❌ No 'Date' column found in the file.")
            return None
        
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df.rename(columns={date_col: "Date"}, inplace=True)

        if "Store" not in df.columns:
            st.error("❌ No 'Store' column found in the file.")
            return None

        st.session_state['uploaded_csv'] = df.to_csv(index=False).encode('utf-8')
        stores = sorted(df["Store"].unique())
        return df, stores
    return None, []

# =========================
# Sidebar
# =========================
st.sidebar.header("Filter Options")
uploaded_file = st.sidebar.file_uploader(
    "Upload Forecast CSV/XLSX", type=["csv", "xlsx"]
)
forecast_df, store_list = load_data(uploaded_file)

if forecast_df is not None and len(store_list) > 0:
    selected_store = st.sidebar.selectbox("Select Store", store_list)
    n_weeks = st.sidebar.slider("Weeks to Display", min_value=4, max_value=52, value=12, step=4)

    # Download button
    st.sidebar.markdown("---")
    st.sidebar.markdown("📥 Download Forecast CSV")
    csv_data = st.session_state.get('uploaded_csv')
    if csv_data is not None:
        st.sidebar.download_button(
            "Download Forecast CSV",
            data=csv_data,
            file_name="future_forecast_download.csv",
            mime="text/csv"
        )

    # =========================
    # Tabs
    # =========================
    tab1, tab2, tab3 = st.tabs(["Store Forecast", "Multi-Store Comparison", "Data Table"])

    # ---------- Tab 1: Store Forecast ----------
    with tab1:
        store_data = forecast_df[forecast_df["Store"] == selected_store].sort_values("Date").head(n_weeks)
        store_data["Rolling_4W"] = store_data["Predicted_Weekly_Sales"].rolling(4).mean()

        # KPIs
        st.subheader(f"🏬 Key Metrics - Store {selected_store}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Forecasted Sales", f"${store_data['Predicted_Weekly_Sales'].sum():,.0f}")
        col2.metric("Average Weekly Sales", f"${store_data['Predicted_Weekly_Sales'].mean():,.0f}")
        col3.metric("Peak Weekly Sales", f"${store_data['Predicted_Weekly_Sales'].max():,.0f}")

        # Line plot with rolling average (Plotly)
        fig = px.line(store_data, x="Date", y=["Predicted_Weekly_Sales", "Rolling_4W"],
                      labels={"value":"Weekly Sales", "variable":"Legend"}, title=f"Forecast + Rolling Avg - Store {selected_store}")
        st.plotly_chart(fig, use_container_width=True)

    # ---------- Tab 2: Multi-Store Comparison ----------
    with tab2:
        st.subheader("📊 Multi-Store Forecast Trend (Next Weeks)")
        multi_store_data = forecast_df[forecast_df["Store"].isin(store_list[:3])].sort_values("Date").head(n_weeks)
        fig2 = px.line(multi_store_data, x="Date", y="Predicted_Weekly_Sales", color=multi_store_data["Store"].astype(str),
                       labels={"color":"Store ID", "Predicted_Weekly_Sales":"Weekly Sales"}, title="Forecasted Sales Trend - Stores 1-3")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("🏬 Average Forecasted Sales per Store")
        store_avg = forecast_df.groupby("Store")["Predicted_Weekly_Sales"].mean().sort_values(ascending=False).reset_index()
        fig3 = px.bar(store_avg, x="Store", y="Predicted_Weekly_Sales", color="Predicted_Weekly_Sales",
                      color_continuous_scale="Viridis", title="Average Forecast per Store")
        st.plotly_chart(fig3, use_container_width=True)

    # ---------- Tab 3: Data Table ----------
    with tab3:
        st.subheader("📄 Forecast Data Table")
        st.dataframe(forecast_df.head(200), use_container_width=True)

else:
    st.warning("Please upload a CSV/XLSX file with columns: 'Date', 'Store', 'Dept', 'Predicted_Weekly_Sales'.")
