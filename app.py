"""
Laptop Price Prediction System
A Streamlit web application that predicts laptop prices based on specifications.
Uses scikit-learn Pipeline with RandomForest and LinearRegression models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="💻 Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# Custom CSS for premium look
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 0.5rem 0;
    }
    .sub-header {
        text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        padding: 1.5rem; border-radius: 16px; text-align: center;
        border: 1px solid #4338ca; margin-bottom: 1rem;
    }
    .metric-card h3 { color: #c4b5fd; font-size: 0.9rem; margin: 0; }
    .metric-card p { color: #f8fafc; font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0 0; }
    .price-box {
        background: linear-gradient(135deg, #065f46, #047857);
        padding: 2rem; border-radius: 20px; text-align: center;
        border: 2px solid #10b981; margin: 1rem 0;
    }
    .price-box h2 { color: #d1fae5; font-size: 1rem; margin: 0; }
    .price-box p { color: #ecfdf5; font-size: 2.5rem; font-weight: 800; margin: 0.5rem 0 0; }
    .stButton > button {
        width: 100%; background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white; border: none; padding: 0.75rem; border-radius: 12px;
        font-size: 1.1rem; font-weight: 600; transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99,102,241,0.4);
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a, #1e1b4b);
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# EUR to INR conversion rate
# ──────────────────────────────────────────────
EUR_TO_INR = 89.5

# ──────────────────────────────────────────────
# Data Loading & Preprocessing (Cached)
# ──────────────────────────────────────────────
@st.cache_data
def load_and_clean_data():
    """Load the laptop dataset and perform all cleaning steps."""
    try:
        df = pd.read_csv("laptop_data.csv", encoding="latin-1")
    except FileNotFoundError:
        return None, "Dataset file 'laptop_data.csv' not found. Place it in the app root directory."
    except Exception as e:
        return None, f"Error reading dataset: {e}"

    # ── Rename price column if needed ──
    if "Price_euros" in df.columns and "Price" not in df.columns:
        df["Price"] = df["Price_euros"]

    # ── Drop ID columns ──
    if "laptop_ID" in df.columns:
        df.drop("laptop_ID", axis=1, inplace=True)
    if "Product" in df.columns:
        df.drop("Product", axis=1, inplace=True)

    # ── Clean Ram: "8GB" → 8 ──
    if "Ram" in df.columns:
        df["Ram"] = df["Ram"].astype(str).str.replace("GB", "", regex=False).str.strip()
        df["Ram"] = pd.to_numeric(df["Ram"], errors="coerce")

    # ── Clean Weight: "1.37kg" → 1.37 ──
    if "Weight" in df.columns:
        df["Weight"] = df["Weight"].astype(str).str.replace("kg", "", regex=False).str.strip()
        df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")

    # ── Extract CPU brand/type ──
    if "Cpu" in df.columns:
        def extract_cpu(cpu_str):
            cpu_str = str(cpu_str).lower()
            if "i7" in cpu_str: return "Intel Core i7"
            if "i5" in cpu_str: return "Intel Core i5"
            if "i3" in cpu_str: return "Intel Core i3"
            if "ryzen" in cpu_str: return "AMD Ryzen"
            if "celeron" in cpu_str: return "Intel Celeron"
            if "pentium" in cpu_str: return "Intel Pentium"
            if "atom" in cpu_str: return "Intel Atom"
            if "xeon" in cpu_str: return "Intel Xeon"
            if "core m" in cpu_str: return "Intel Core M"
            if "amd a" in cpu_str or "a10" in cpu_str or "a12" in cpu_str or "a9" in cpu_str or "a6" in cpu_str or "a4" in cpu_str or "a8" in cpu_str:
                return "AMD A-Series"
            if "amd e" in cpu_str or "e2-" in cpu_str or "e-series" in cpu_str:
                return "AMD E-Series"
            if "amd fx" in cpu_str: return "AMD FX"
            return "Other"
        df["Cpu_Brand"] = df["Cpu"].apply(extract_cpu)

    # ── Extract GPU brand ──
    if "Gpu" in df.columns:
        def extract_gpu(gpu_str):
            gpu_str = str(gpu_str).lower()
            if "nvidia" in gpu_str: return "Nvidia"
            if "intel" in gpu_str: return "Intel"
            if "amd" in gpu_str: return "AMD"
            return "Other"
        df["Gpu_Brand"] = df["Gpu"].apply(extract_gpu)

    # ── Extract touchscreen & IPS from ScreenResolution ──
    if "ScreenResolution" in df.columns:
        df["Touchscreen"] = df["ScreenResolution"].str.lower().str.contains("touchscreen").astype(int)
        df["IPS_Panel"] = df["ScreenResolution"].str.lower().str.contains("ips").astype(int)
        # Extract X & Y resolution
        res = df["ScreenResolution"].str.extract(r"(\d{3,4})x(\d{3,4})")
        df["X_res"] = pd.to_numeric(res[0], errors="coerce").fillna(1366)
        df["Y_res"] = pd.to_numeric(res[1], errors="coerce").fillna(768)
        df["PPI"] = ((df["X_res"]**2 + df["Y_res"]**2)**0.5 / df["Inches"]).round(2)

    # ── Extract storage info from Memory ──
    if "Memory" in df.columns:
        def extract_ssd(mem):
            mem = str(mem)
            ssd = 0
            matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:TB|GB)\s*SSD", mem, re.IGNORECASE)
            for m in matches:
                val = float(m)
                if "TB" in mem.upper().split("SSD")[0]: val *= 1024
                ssd += int(val)
            # Also catch Flash Storage as SSD
            flash = re.findall(r"(\d+(?:\.\d+)?)\s*(?:TB|GB)\s*Flash", mem, re.IGNORECASE)
            for m in flash:
                val = float(m)
                ssd += int(val)
            return ssd

        def extract_hdd(mem):
            mem = str(mem)
            hdd = 0
            # Check for TB HDD first
            tb_matches = re.findall(r"(\d+(?:\.\d+)?)\s*TB\s*HDD", mem, re.IGNORECASE)
            for m in tb_matches:
                hdd += int(float(m) * 1024)
            gb_matches = re.findall(r"(\d+(?:\.\d+)?)\s*GB\s*HDD", mem, re.IGNORECASE)
            for m in gb_matches:
                hdd += int(float(m))
            # Hybrid drives count as HDD
            hyb = re.findall(r"(\d+(?:\.\d+)?)\s*TB\s*Hybrid", mem, re.IGNORECASE)
            for m in hyb:
                hdd += int(float(m) * 1024)
            return hdd

        df["SSD"] = df["Memory"].apply(extract_ssd)
        df["HDD"] = df["Memory"].apply(extract_hdd)

    # ── Convert Price to INR ──
    if "Price" in df.columns:
        df["Price_INR"] = (df["Price"] * EUR_TO_INR).round(0)

    # ── Drop original raw columns ──
    cols_to_drop = ["Cpu", "Gpu", "ScreenResolution", "Memory", "Inches",
                    "Price", "Price_euros", "X_res", "Y_res"]
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # ── Handle missing values ──
    df.dropna(subset=["Price_INR"], inplace=True)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df, None


@st.cache_resource
def train_models(_df):
    """Train LinearRegression and RandomForest models, return the best one."""
    df = _df.copy()
    target = "Price_INR"
    X = df.drop(target, axis=1)
    y = df[target]

    # Identify column types
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Build preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop"
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Model 1: Linear Regression ──
    lr_pipe = Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    lr_r2 = r2_score(y_test, lr_pred)

    # ── Model 2: Random Forest ──
    rf_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1))
    ])
    rf_pipe.fit(X_train, y_train)
    rf_pred = rf_pipe.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)

    # ── Select best model ──
    best_name = "Random Forest" if rf_r2 >= lr_r2 else "Linear Regression"
    best_pipe = rf_pipe if rf_r2 >= lr_r2 else lr_pipe

    # ── Feature importance (for RF) ──
    feat_importance = None
    if best_name == "Random Forest":
        ohe_feat = list(rf_pipe.named_steps["preprocessor"]
                        .named_transformers_["cat"]
                        .get_feature_names_out(cat_cols))
        all_feat = num_cols + ohe_feat
        importances = rf_pipe.named_steps["model"].feature_importances_
        feat_importance = pd.Series(importances, index=all_feat).sort_values(ascending=False).head(15)

    results = {
        "best_name": best_name,
        "best_pipe": best_pipe,
        "lr": {"rmse": lr_rmse, "r2": lr_r2, "pred": lr_pred},
        "rf": {"rmse": rf_rmse, "r2": rf_r2, "pred": rf_pred},
        "y_test": y_test,
        "feat_importance": feat_importance,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "X": X,
    }
    return results


# ──────────────────────────────────────────────
# Main App (top-level execution)
# ──────────────────────────────────────────────

# Header
st.markdown('<h1 class="main-header">💻 Laptop Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict laptop prices using Machine Learning ', unsafe_allow_html=True)

# Load data
df, error = load_and_clean_data()
if error:
    st.error(error)
    st.stop()

# Train models
results = train_models(df)

# ──────────────────────────────────────────
# Sidebar — User Inputs
# ──────────────────────────────────────────
st.sidebar.markdown("## 🔧 Laptop Specifications")
st.sidebar.markdown("Configure the laptop specs below:")

company = st.sidebar.selectbox("🏢 Brand", sorted(df["Company"].unique()))
type_name = st.sidebar.selectbox("📁 Type", sorted(df["TypeName"].unique()))
cpu_brand = st.sidebar.selectbox("🧠 CPU", sorted(df["Cpu_Brand"].unique()))
gpu_brand = st.sidebar.selectbox("🎮 GPU", sorted(df["Gpu_Brand"].unique()))
opsys = st.sidebar.selectbox("🖥️ Operating System", sorted(df["OpSys"].unique()))

ram = st.sidebar.select_slider("💾 RAM (GB)", options=[2, 4, 6, 8, 12, 16, 24, 32, 64], value=8)
weight = st.sidebar.slider("⚖️ Weight (kg)", 0.5, 5.0, 1.8, 0.1)
ssd = st.sidebar.select_slider("💿 SSD (GB)", options=[0, 32, 64, 128, 256, 512, 1024], value=256)
hdd = st.sidebar.select_slider("💽 HDD (GB)", options=[0, 128, 256, 500, 1024, 2048], value=0)
touchscreen = st.sidebar.radio("👆 Touchscreen", ["No", "Yes"], horizontal=True)
ips = st.sidebar.radio("🖼️ IPS Panel", ["No", "Yes"], horizontal=True)
ppi = st.sidebar.slider("📐 PPI (Pixels Per Inch)", 80, 400, 141, 5)

predict_btn = st.sidebar.button("🔮 Predict Price")

# ──────────────────────────────────────────
# Tabs (Prediction + Visualizations only)
# ──────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Visualizations"])

# ── TAB 1: Prediction ──
with tab1:
    if predict_btn:
        try:
            input_data = pd.DataFrame([{
                "Company": company,
                "TypeName": type_name,
                "Ram": ram,
                "Weight": weight,
                "OpSys": opsys,
                "Cpu_Brand": cpu_brand,
                "Gpu_Brand": gpu_brand,
                "Touchscreen": 1 if touchscreen == "Yes" else 0,
                "IPS_Panel": 1 if ips == "Yes" else 0,
                "PPI": ppi,
                "SSD": ssd,
                "HDD": hdd,
            }])

            # Ensure column order matches training data
            for col in results["X"].columns:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[results["X"].columns]

            prediction = results["best_pipe"].predict(input_data)[0]
            prediction = max(prediction, 0)

            st.markdown(f"""
            <div class="price-box">
                <h2>Estimated Laptop Price</h2>
                <p>₹{prediction:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)

            st.info(f"🤖 Predicted using **{results['best_name']}** model (R² = {results['rf']['r2'] if results['best_name'] == 'Random Forest' else results['lr']['r2']:.4f})")

            # Show input summary
            st.markdown("#### 📋 Your Configuration")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Brand:** {company}")
                st.markdown(f"**Type:** {type_name}")
                st.markdown(f"**CPU:** {cpu_brand}")
                st.markdown(f"**GPU:** {gpu_brand}")
            with col2:
                st.markdown(f"**RAM:** {ram} GB")
                st.markdown(f"**SSD:** {ssd} GB")
                st.markdown(f"**HDD:** {hdd} GB")
                st.markdown(f"**Weight:** {weight} kg")
            with col3:
                st.markdown(f"**OS:** {opsys}")
                st.markdown(f"**Touchscreen:** {touchscreen}")
                st.markdown(f"**IPS Panel:** {ips}")
                st.markdown(f"**PPI:** {ppi}")

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
    else:
        st.markdown("### 👈 Configure laptop specs in the sidebar and click **Predict Price**")
        # Quick stats
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><h3>Total Laptops</h3><p>{len(df):,}</p></div>', unsafe_allow_html=True)
        with c2:
            avg_price = df["Price_INR"].mean()
            st.markdown(f'<div class="metric-card"><h3>Avg Price</h3><p>₹{avg_price:,.0f}</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><h3>Brands</h3><p>{df["Company"].nunique()}</p></div>', unsafe_allow_html=True)
        with c4:
            best_r2 = max(results["lr"]["r2"], results["rf"]["r2"])
            st.markdown(f'<div class="metric-card"><h3>Best R² Score</h3><p>{best_r2:.4f}</p></div>', unsafe_allow_html=True)

# ── TAB 2: Visualizations ──
with tab2:
    st.markdown("### 📊 Data Visualizations")

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        # Price Distribution
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        fig1.patch.set_facecolor("#0e1117")
        ax1.set_facecolor("#0e1117")
        ax1.hist(df["Price_INR"], bins=40, color="#8b5cf6", edgecolor="#c4b5fd", alpha=0.85)
        ax1.set_title("Price Distribution (₹)", color="white", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Price (₹)", color="#94a3b8")
        ax1.set_ylabel("Count", color="#94a3b8")
        ax1.tick_params(colors="#94a3b8")
        for spine in ax1.spines.values():
            spine.set_color("#334155")
        st.pyplot(fig1)

    with viz_col2:
        # Brand vs Avg Price
        brand_avg = df.groupby("Company")["Price_INR"].mean().sort_values(ascending=True)
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        fig2.patch.set_facecolor("#0e1117")
        ax2.set_facecolor("#0e1117")
        bars = ax2.barh(brand_avg.index, brand_avg.values, color="#6366f1", edgecolor="#a78bfa")
        ax2.set_title("Avg Price by Brand (₹)", color="white", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Avg Price (₹)", color="#94a3b8")
        ax2.tick_params(colors="#94a3b8")
        for spine in ax2.spines.values():
            spine.set_color("#334155")
        st.pyplot(fig2)

    viz_col3, viz_col4 = st.columns(2)

    with viz_col3:
        # Feature Importance
        if results["feat_importance"] is not None:
            fi = results["feat_importance"].head(10)
            fig3, ax3 = plt.subplots(figsize=(7, 5))
            fig3.patch.set_facecolor("#0e1117")
            ax3.set_facecolor("#0e1117")
            ax3.barh(fi.index[::-1], fi.values[::-1], color="#10b981", edgecolor="#6ee7b7")
            ax3.set_title("Top 10 Feature Importances", color="white", fontsize=14, fontweight="bold")
            ax3.tick_params(colors="#94a3b8")
            for spine in ax3.spines.values():
                spine.set_color("#334155")
            st.pyplot(fig3)
        else:
            st.info("Feature importance is only available for Random Forest.")

    with viz_col4:
        # Model Comparison
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        fig4.patch.set_facecolor("#0e1117")
        ax4.set_facecolor("#0e1117")
        models_names = ["Linear Regression", "Random Forest"]
        r2_scores = [results["lr"]["r2"], results["rf"]["r2"]]
        colors = ["#f59e0b", "#6366f1"]
        bars = ax4.bar(models_names, r2_scores, color=colors, edgecolor=["#fbbf24", "#a78bfa"], width=0.5)
        ax4.set_title("Model Comparison (R² Score)", color="white", fontsize=14, fontweight="bold")
        ax4.set_ylabel("R² Score", color="#94a3b8")
        ax4.set_ylim(0, 1)
        ax4.tick_params(colors="#94a3b8")
        for bar, score in zip(bars, r2_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{score:.4f}", ha="center", color="white", fontweight="bold")
        for spine in ax4.spines.values():
            spine.set_color("#334155")
        st.pyplot(fig4)

