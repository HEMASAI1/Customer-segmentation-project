import streamlit as st
import polars as pl
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.set_page_config(layout="wide")

st.title("📊 Customer Segmentation Dashboard")

# =========================
# 📥 LOAD DATA
# =========================
df = pl.read_csv("data/clustered_customers.csv")
df_pd = df.to_pandas()

# =========================
# 🧠 FEATURE ENGINEERING
# =========================

# Handle negative monetary for visualization
df_pd["Monetary_abs"] = df_pd["Monetary"].abs()

# Normalize features
df_pd["Recency_score"] = df_pd["Recency"] / df_pd["Recency"].max()
df_pd["Frequency_score"] = df_pd["Frequency"] / df_pd["Frequency"].max()
df_pd["Monetary_score"] = df_pd["Monetary"] / df_pd["Monetary"].max()

# Invert good metrics (higher = worse)
df_pd["Frequency_score"] = 1 - df_pd["Frequency_score"]
df_pd["Monetary_score"] = 1 - df_pd["Monetary_score"]

# Churn Score (weighted)
df_pd["Churn_Score"] = (
    0.5 * df_pd["Recency_score"] +
    0.3 * df_pd["Frequency_score"] +
    0.2 * df_pd["Monetary_score"]
)

# Define churn label
df_pd["Churn"] = (df_pd["Churn_Score"] > 0.6).astype(int)

# =========================
# 🤖 TRAIN MODEL (CACHED)
# =========================
@st.cache_resource
def train_model(data):
    X = data[["Recency", "Frequency", "Monetary"]]
    y = data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model

model = train_model(df_pd)

# =========================
# 📊 KPI SECTION
# =========================
st.subheader("📈 Key Insights")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", df_pd["CustomerID"].nunique())
col2.metric("Total Revenue", f"{df_pd['Monetary'].sum():.2f}")
col3.metric("Avg Revenue per Customer", f"{df_pd['Monetary'].mean():.2f}")

# =========================
# 🔍 CUSTOMER SELECTION
# =========================
st.subheader("🔎 Customer Details")

customer_id = st.selectbox("Select Customer ID", df_pd["CustomerID"])

customer_data = df_pd[df_pd["CustomerID"] == customer_id]

col1, col2 = st.columns(2)

with col1:
    st.write("### Customer Info")
    st.dataframe(customer_data)

with col2:
    segment = customer_data["Segment"].values[0]
    st.write("### Segment")
    st.success(segment)

    # Strategy
    if segment == "VIP":
        st.success("🎁 Give rewards and premium offers")
    elif segment == "Churned":
        st.warning("⚠️ Send discounts to re-engage")
    elif segment == "At-Risk":
        st.info("📢 Target with personalized campaigns")
    elif segment == "Loyal":
        st.success("💙 Maintain relationship")
    elif segment == "Regular":
        st.info("📦 Upsell products")
    else:
        st.info("🔍 Monitor behavior")

# =========================
# 🔮 CHURN PREDICTION
# =========================
st.subheader("🔮 Churn Prediction")

if not customer_data.empty:
    customer_features = customer_data[["Recency", "Frequency", "Monetary"]]

    churn_prob = model.predict_proba(customer_features)[0][1]

    # Show churn score
    churn_score = customer_data["Churn_Score"].values[0]
    st.write(f"### Churn Score: {churn_score:.2f}")

    if churn_prob > 0.7:
        st.error(f"🔥 High Risk ({churn_prob:.2f}) - Immediate Action Needed")
    elif churn_prob > 0.4:
        st.warning(f"⚠️ Medium Risk ({churn_prob:.2f}) - Monitor Closely")
    else:
        st.success(f"✅ Low Risk ({churn_prob:.2f}) - Healthy Customer")

# =========================
# 📊 VISUALIZATIONS
# =========================
st.subheader("📊 Customer Insights")

col1, col2 = st.columns(2)

# 🔥 Scatter Plot
with col1:
    fig1 = px.scatter(
        df_pd,
        x="Recency",
        y="Monetary",
        color="Segment",
        title="Recency vs Monetary (Customer Value)",
        hover_data=["Frequency"]
    )
    st.plotly_chart(fig1, use_container_width=True)

# 🔥 Bar Chart
with col2:
    segment_revenue = df_pd.groupby("Segment")["Monetary"].mean().reset_index()

    fig2 = px.bar(
        segment_revenue,
        x="Segment",
        y="Monetary",
        color="Segment",
        title="Average Revenue by Segment"
    )
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# 📉 CHURN DISTRIBUTION
# =========================
st.subheader("📉 Churn Distribution")

fig_churn = px.pie(
    df_pd,
    names="Churn",
    title="Churn vs Active Customers"
)
st.plotly_chart(fig_churn, use_container_width=True)

# =========================
# 📊 EXTRA VISUALS
# =========================
col3, col4 = st.columns(2)

# 🥧 Segment Distribution
with col3:
    fig3 = px.pie(
        df_pd,
        names="Segment",
        title="Customer Segment Distribution"
    )
    st.plotly_chart(fig3, use_container_width=True)

# 📉 Frequency vs Monetary
with col4:
    fig4 = px.scatter(
        df_pd,
        x="Frequency",
        y="Monetary",
        color="Segment",
        size="Monetary_abs",
        title="Frequency vs Monetary",
        hover_data=["Recency"]
    )
    st.plotly_chart(fig4, use_container_width=True)