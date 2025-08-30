# =========================
# Walmart Sales Forecasting Dashboard
# Professional Version - Robust CSV/Excel Handling
# Author: Shahzad
# =========================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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
    "Interactive dashboard to visualize predicted weekly sales with KPIs, trends, and multi-store insights."
)

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_model.pkl")  # Ensure model is in project root
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# =========================
# Load Data (CSV / Excel) + Robust Handling
# =========================
@st.cache_data
def load_data(file_path=None):
    try:
        # -------------------------
        # Load file
        # -------------------------
        if file_path:
            try:
                df = pd.read_csv(file_path)
            except Exception:
                try:
                    df = pd.read_excel(file_path, engine="openpyxl")
                except Exception:
                    try:
                        df = pd.read_excel(file_path, engine="xlrd")
                    except Exception as e:
                        st.error("❌ Unsupported format or corrupt file. Upload CSV, XLS, or XLSX.")
                        return None, []

        else:
            # Default file
            try:
                df = pd.read_csv("future_forecast.csv")
            except FileNotFoundError:
                try:
                    df = pd.read_excel("future_forecast.xlsx", engine="openpyxl")
                except:
                    df = pd.read_excel("future_forecast.xls", engine="xlrd")

        # -------------------------
        # Strip spaces from headers
        # -------------------------
        df.columns = df.columns.str.strip()

        # -------------------------
        # Detect Date column
        # -------------------------
        date_cols = [c for c in df.columns if "date" in c.lower()]
        if not date_cols:
            st.error(f"❌ No 'Date' column found. Columns detected: {df.columns.tolist()}")
            return None, []

        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        dropped = df[df[date_col].isna()].shape[0]
        if dropped > 0:
            st.warning(f"⚠️ Dropped {dropped} rows with invalid dates.")
        df = df.dropna(subset=[date_col])
        df.rename(columns={date_col: "Date"}, inplace=True)

        # -------------------------
        # Check Store column
        # -------------------------
        if "Store" not in df.columns:
            st.error(f"❌ No 'Store' column found. Columns detected: {df.columns.tolist()}")
            return None, []

        stores = sorted(df["Store"].unique())

        # -------------------------
        # Store CSV for download
        # -------------------------
        st.session_state['uploaded_csv'] = df.to_csv(index=False).encode('utf-8')

        return df, stores

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, []

# =========================
# Sidebar Options
# =========================
st.sidebar.header("Filter Options")
uploaded_file = st.sidebar.file_uploader(
    "Upload Forecast (CSV, XLS, or XLSX)", type=["csv", "xls", "xlsx"]
)
forecast_df, store_list = load_data(uploaded_file)

if forecast_df is not None and len(store_list) > 0:

    selected_store = st.sidebar.selectbox("Select Store", store_list)
    n_weeks = st.sidebar.slider("Weeks to Display", min_value=4, max_value=52, value=12, step=4)

    # -------------------------
    # Download Button
    # -------------------------
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
    # Filter data for selected store
    # =========================
    store_data = forecast_df[forecast_df["Store"] == selected_store].sort_values("Date").head(n_weeks)
    store_data["Rolling_4W"] = store_data["Predicted_Weekly_Sales"].rolling(4).mean()

    # =========================
    # KPIs
    # =========================
    total_sales = store_data["Predicted_Weekly_Sales"].sum()
    avg_sales = store_data["Predicted_Weekly_Sales"].mean()
    max_sales = store_data["Predicted_Weekly_Sales"].max()

    st.subheader(f"🏬 Key Metrics - Store {selected_store}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Forecasted Sales", f"${total_sales:,.0f}")
    col2.metric("Average Weekly Sales", f"${avg_sales:,.0f}")
    col3.metric("Peak Weekly Sales", f"${max_sales:,.0f}")

    st.markdown("---")

    # =========================
    # Forecast plot
    # =========================
    st.subheader(f"📈 Forecasted Weekly Sales - Store {selected_store}")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(store_data["Date"], store_data["Predicted_Weekly_Sales"], marker="o", linestyle="-", color="#1f77b4", label="Predicted Sales")
    ax.plot(store_data["Date"], store_data["Rolling_4W"], linestyle="--", color="#ff7f0e", label="4-Week Rolling Avg", linewidth=2)
    ax.fill_between(store_data["Date"], store_data["Rolling_4W"]*0.9, store_data["Rolling_4W"]*1.1, color="#ff7f0e", alpha=0.1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Weekly Sales")
    ax.set_title(f"Forecast + Rolling Average for Store {selected_store}", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # =========================
    # Multi-store forecast
    # =========================
    st.subheader("📊 Multi-Store Forecast Trend (Next Weeks)")
    multi_store_data = forecast_df[forecast_df["Store"].isin(store_list[:3])].sort_values("Date")
    fig2, ax2 = plt.subplots(figsize=(12,5))
    colors = ["#1f77b4","#ff7f0e","#2ca02c"]
    for i, store_id in enumerate(store_list[:3]):
        data = multi_store_data[multi_store_data["Store"] == store_id].head(n_weeks)
        ax2.plot(data["Date"], data["Predicted_Weekly_Sales"], marker="o", linestyle="-", color=colors[i], label=f"Store {store_id}", linewidth=2)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Predicted Weekly Sales")
    ax2.set_title("Forecasted Sales Trend - Stores 1-3", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # =========================
    # Store-wise average
    # =========================
    st.subheader("🏬 Average Forecasted Sales per Store")
    store_avg = forecast_df.groupby("Store")["Predicted_Weekly_Sales"].mean().sort_values().reset_index()
    fig3, ax3 = plt.subplots(figsize=(10,5))
    sns.barplot(x="Predicted_Weekly_Sales", y="Store", data=store_avg, palette="mako", ax=ax3)
    ax3.set_xlabel("Average Predicted Weekly Sales")
    ax3.set_ylabel("Store")
    ax3.set_title("Average Forecast per Store", fontsize=14, fontweight="bold")
    ax3.grid(axis="x", alpha=0.2)
    st.pyplot(fig3)

    st.markdown("---")
    st.markdown("Developed by **Shahzad** | Professional Walmart Sales Forecast Dashboard")

else:
    st.warning("Please upload a CSV or Excel file (.csv, .xls, .xlsx) with 'Date' and 'Store' columns to display the dashboard.")
