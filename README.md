# 💻 Laptop Price Prediction System

A machine learning-powered web application that predicts laptop prices based on hardware specifications. Built with **Streamlit** and **scikit-learn**.

🔗 **[Live Demo](https://laptop-price-prediction-system-negz8ot4zsvaqgonls7syb.streamlit.app/)**

---

## 📌 Overview

This project uses regression-based ML models to estimate laptop prices in **₹ (INR)** from real-world laptop specifications. The app trains models on-the-fly using a cleaned dataset of **1,300+ laptops** from major brands.

### Key Highlights

- 🤖 **Two ML Models** — Linear Regression & Random Forest (best auto-selected)
- 📊 **R² Score ≈ 0.82** with Random Forest
- 🧹 **Auto Data Cleaning** — handles messy real-world CSV data
- ⚡ **Cached Training** — fast reloads with `@st.cache_resource`
- 🎨 **Premium Dark UI** — gradient cards, styled charts, responsive layout

---

## 🖥️ Features

| Feature | Description |
|---------|-------------|
| **Price Prediction** | Predict laptop price from specs via sidebar inputs |
| **Brand Selection** | 19 brands including Apple, Dell, HP, Lenovo, Asus, MSI |
| **CPU/GPU Options** | Intel i3/i5/i7, AMD Ryzen, Nvidia, Intel, AMD GPUs |
| **Storage Config** | SSD (0–1024 GB) and HDD (0–2048 GB) sliders |
| **Display Features** | Touchscreen, IPS Panel, PPI configuration |
| **Visualizations** | Price distribution, brand comparison, feature importance, model comparison |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core language |
| Streamlit | Web framework & UI |
| scikit-learn | ML pipeline (RandomForest, LinearRegression, ColumnTransformer) |
| pandas / NumPy | Data manipulation |
| matplotlib | Visualizations |

---

## 📂 Project Structure

```
Laptop price prediciton system/
├── app.py              # Complete Streamlit application (single file)
├── laptop_data.csv     # Dataset (1,303 laptops)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/tanishaalone30/laptop-price-prediction-system.git
cd laptop-price-prediction-system

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📊 Dataset

- **Source:** Real-world laptop specifications dataset
- **Size:** 1,303 entries × 13 features
- **Target:** Price in Euros (converted to ₹ INR in-app)

### Features Used

| Feature | Type | Processing |
|---------|------|------------|
| Company | Categorical | OneHotEncoded |
| TypeName | Categorical | OneHotEncoded |
| Ram | Numeric | Extracted from "8GB" → 8 |
| Weight | Numeric | Extracted from "1.37kg" → 1.37 |
| OpSys | Categorical | OneHotEncoded |
| CPU | Categorical | Extracted brand (i3, i5, i7, Ryzen, etc.) |
| GPU | Categorical | Extracted brand (Nvidia, Intel, AMD) |
| Touchscreen | Binary | Extracted from ScreenResolution |
| IPS Panel | Binary | Extracted from ScreenResolution |
| PPI | Numeric | Calculated from resolution & screen size |
| SSD | Numeric | Extracted from Memory column (GB) |
| HDD | Numeric | Extracted from Memory column (GB) |

---

## 🤖 Model Performance

| Model | R² Score | Status |
|-------|----------|--------|
| Linear Regression | ~0.65 | Baseline |
| **Random Forest** | **~0.82** | ✅ **Auto-selected** |

The app automatically compares both models and selects the one with the higher R² score.

---

## 📸 Screenshots

### Prediction Tab
> Configure laptop specs in the sidebar and get instant price predictions in ₹ INR.

### Visualizations Tab
> Explore price distribution, brand-wise comparison, feature importance, and model comparison charts.

---

## 📝 Requirements

```
streamlit
pandas
numpy
scikit-learn
matplotlib
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
