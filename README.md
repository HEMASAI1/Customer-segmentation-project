# 🛍️ Customer Segmentation using Machine Learning
> *Stop treating every customer the same. Start marketing smarter.*

Segment retail customers based on real purchasing behavior using **RFM Analysis + HDBSCAN Clustering** — and visualize actionable insights through an interactive business dashboard.

---

## 📸 Demo

### 🖥️ Live Dashboard
![Dashboard Overview](images/dashboard_overview.png)

| 🔍 Customer Lookup & Segment Strategy | 🔮 Churn Risk Prediction |
|---|---|
| ![Customer Details](images/customer_details.png) | ![Churn Prediction](images/churn_prediction.png) |

| 📊 Segment Distribution | 💰 Revenue by Segment |
|---|---|
| ![Segments](images/segment_distribution.png) | ![Revenue](images/revenue_by_segment.png) |

---

## ❗ The Problem

Most businesses send the **same email, same discount, same message** to every customer — whether they spent £20 or £20,000. This results in:

- 💸 Wasted marketing budget on customers who won't convert
- 😤 High-value customers feeling under-valued
- 📉 No early warning system for customers about to leave
- 🤷 Zero understanding of *why* customers churn

**The business impact?** Revenue leaks silently while marketing burns cash.

---

## ✅ The Solution

This project builds a **data-driven customer intelligence system** that:

1. Analyses every customer's purchasing history using **RFM metrics** (Recency, Frequency, Monetary)
2. Automatically groups customers into meaningful segments using **density-based clustering**
3. Predicts which customers are at risk of churning — *before they leave*
4. Delivers a real-time **Streamlit dashboard** so business teams can act on insights instantly

> 📌 Result: 4,308 customers segmented into 7 actionable groups from a dataset of 500K+ transactions.

---

## 🚀 Features

- **End-to-end ML pipeline** — raw CSV to clustered insights in 2 notebooks
- **RFM feature engineering** — proven framework used by top e-commerce companies
- **Outlier removal** — Z-score based cleaning for cleaner cluster boundaries
- **HDBSCAN clustering** — no need to guess the number of clusters; algorithm finds natural groupings
- **Churn scoring engine** — weighted formula combining all 3 RFM dimensions
- **Logistic regression churn predictor** — per-customer churn probability with risk labels
- **Interactive Streamlit dashboard** — lookup any customer, see their segment, risk score & recommended action
- **Plotly visualizations** — scatter plots, bar charts & pie charts, all interactive

---

## 📊 Results & Impact

| Metric | Value |
|---|---|
| Total Customers Analysed | **4,308** |
| Total Revenue in Dataset | **£5,663,347** |
| Avg Revenue per Customer | **£1,314.61** |
| Customers at Churn Risk | **~31.5%** |
| Segments Identified | **7 distinct groups** |

### 🎯 Segment Breakdown

| Segment | Share | Recommended Action |
|---|---|---|
| 🔵 Outlier | 69.5% | Monitor & investigate |
| 🔷 One-Time | 11.4% | Welcome-back offers |
| 🩷 At-Risk | 6.45% | Personalised campaigns |
| 🔴 Regular | 5.59% | Upsell & cross-sell |
| 💚 Loyal | 3.9% | Maintain relationship |
| 🟢 Churned | 1.9% | Re-engagement discounts |
| 🏆 VIP | 1.32% | Rewards & premium offers |

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | `Python 3.12` |
| Data Processing | `Polars`, `Pandas` |
| ML & Clustering | `scikit-learn`, `HDBSCAN` |
| Statistics | `SciPy` (Z-score outlier detection) |
| Visualisation | `Plotly` |
| Dashboard | `Streamlit` |

---

## 📁 Project Structure

```
customer-segmentation/
│
├── data/
│   ├── data.csv                       # Raw UCI Online Retail dataset
│   ├── processed_features.csv         # Cleaned RFM features
│   └── clustered_customers.csv        # Final output with segment labels
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb    # Cleaning, RFM computation, outlier removal
│   └── 02_clustering.ipynb            # Scaling, HDBSCAN clustering, segment mapping
│
├── src/
│   └── app.py                         # Streamlit dashboard (KPIs, churn, charts)
│
├── images/                            # Screenshots for README
├── models/                            # Saved model artifacts
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Usage

```bash
# 1. Clone the repo
git clone https://github.com/your-username/customer-segmentation.git
cd customer-segmentation

# 2. Install dependencies
pip install -r requirements.txt
pip install hdbscan

# 3. Add the dataset
# Download from: https://archive.ics.uci.edu/ml/datasets/online+retail
# Place as: data/data.csv

# 4. Run notebooks in order
jupyter notebook notebooks/01_data_preprocessing.ipynb
jupyter notebook notebooks/02_clustering.ipynb

# 5. Launch dashboard
streamlit run src/app.py
```

---

## 🔮 How the Churn Score Works

Each customer receives a **Churn Score between 0 and 1** based on normalised RFM dimensions:

```
Churn Score = (0.5 × Recency) + (0.3 × Frequency) + (0.2 × Monetary)
```

| Score Range | Risk Level | Action |
|---|---|---|
| > 0.70 | 🔥 High Risk | Immediate intervention |
| 0.40 – 0.70 | ⚠️ Medium Risk | Monitor closely |
| < 0.40 | ✅ Low Risk | Healthy customer |

---

## 🗂️ Dataset

- **Source:** [UCI Machine Learning Repository — Online Retail](https://archive.ics.uci.edu/ml/datasets/online+retail)
- **Period:** December 2010 – December 2011
- **Records:** ~541,000 transactions
- **Customers:** ~4,300 unique (after cleaning)
- **Region:** Primarily United Kingdom

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgements

- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+retail)
- Clustering: [HDBSCAN Documentation](https://hdbscan.readthedocs.io/)
- Dashboard: [Streamlit](https://streamlit.io/)

---

<p align="center">
  <i>Built to turn raw transaction data into business intelligence.</i><br/>
  ⭐ Star this repo if you found it useful!
</p>
