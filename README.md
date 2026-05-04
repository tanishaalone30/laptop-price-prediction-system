# 💻 Laptop Price Prediction System

> **An intelligent ML-powered web application that predicts laptop prices in Indian Rupees (₹) using real-world laptop specifications.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

---

## 🌐 Live Demo

🔗 **[Launch the App →](https://YOUR_USERNAME-laptop-price-predictor.streamlit.app)**

> _Replace `YOUR_USERNAME` with your Streamlit Cloud / GitHub username after deployment._
---

## 📸 Screenshots

| Hero & Dashboard | Prediction Result |
|:---:|:---:|
| Gradient hero banner with 4 live metric cards (Total Laptops, Brands, Avg Price, Price Range) | Predicted price displayed in a vibrant card with full spec summary |

| Price Distribution | Feature Importance |
|:---:|:---:|
| Histogram of laptop prices in ₹ INR with median line | Top-15 features ranked by Random Forest importance |

---

## ✨ Features

- 🎯 **Dual ML Models** — Linear Regression & Random Forest Regressor trained in-app
- 📊 **1,303 Real Laptops** — Dataset covering 19 brands and 6 laptop types
- 🇮🇳 **Prices in ₹ INR** — Automatic Euro → INR conversion
- ⚡ **Instant Predictions** — Cached models for sub-second inference
- 🧠 **Smart Feature Engineering** — Extracts CPU family, GPU brand, SSD/HDD split, screen resolution, IPS & touchscreen flags
- 📈 **Interactive Visualizations** — Price distribution & feature importance charts (matplotlib)
- 🛡️ **Crash-Proof** — Full `try-except` error handling with user-friendly messages
- 🎨 **Modern UI** — Gradient hero, dark sidebar, metric cards with hover effects

---

## 🏗️ Architecture

```
User Input (Sidebar)
       │
       ▼
┌─────────────────────┐
│  Feature Engineering │  ← CPU brand, GPU brand, SSD/HDD, resolution
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  ColumnTransformer   │  ← Numeric passthrough + OneHotEncoder
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌──────────────┐
│ Linear  │ │ Random Forest│
│Regression│ │  Regressor   │
└────┬────┘ └──────┬───────┘
     │             │
     └──────┬──────┘
            ▼
     Predicted Price (₹)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/laptop-price-predictor.git
cd laptop-price-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📁 Project Structure

```
laptop-price-predictor/
├── app.py              # Main Streamlit application (~380 lines)
├── laptop_data.csv     # Dataset — 1,303 laptops, 13 columns
├── requirements.txt    # 5 dependencies only
└── README.md           # This file
```

---

## 📊 Dataset

| Column | Description | Example |
|---|---|---|
| `Company` | Brand name | Apple, Dell, HP |
| `TypeName` | Laptop category | Ultrabook, Gaming, Notebook |
| `Inches` | Screen size | 13.3, 15.6, 17.3 |
| `ScreenResolution` | Display resolution + panel type | IPS Panel Full HD 1920x1080 |
| `Cpu` | Processor details | Intel Core i5 8250U 1.6GHz |
| `Ram` | RAM size | 8GB |
| `Memory` | Storage configuration | 256GB SSD, 128GB SSD + 1TB HDD |
| `Gpu` | Graphics card | Nvidia GeForce MX150 |
| `OpSys` | Operating system | Windows 10, macOS |
| `Weight` | Laptop weight | 1.37kg |
| `Price_euros` | Price in Euros (target) | 1339.69 |

**Derived Features (auto-extracted):**
- `Cpu_Brand` — Intel Core i5, AMD Ryzen, etc.
- `Gpu_Brand` — Nvidia, AMD, Intel
- `SSD_GB` / `HDD_GB` — Storage split in GB
- `X_Res`, `Y_Res`, `PPI_Proxy` — Screen resolution metrics
- `Is_IPS`, `Is_Touchscreen` — Display flags

---

## 🧪 Model Performance

| Model | R² Score | MAE (₹) |
|---|---|---|
| **Random Forest** | **0.8107** | **₹17,104** |
| Linear Regression | 0.7722 | ₹21,935 |

> Random Forest is the recommended model (selected by default in the UI).

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web framework & UI |
| [pandas](https://pandas.pydata.org) | Data loading & manipulation |
| [NumPy](https://numpy.org) | Numerical operations |
| [scikit-learn](https://scikit-learn.org) | ML models & preprocessing |
| [matplotlib](https://matplotlib.org) | Visualizations |

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → branch `main` → file `app.py`
4. Click **Deploy** — your app will be live in ~2 minutes

Your app URL will be: `https://YOUR_USERNAME-laptop-price-predictor.streamlit.app`

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

