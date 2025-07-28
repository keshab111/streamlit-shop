import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load data and models
@st.cache_data
def load_data():
    df = pd.read_csv("online_retail_clean.csv")  # Preprocessed CSV
    pivot = pd.read_pickle("customer_product_pivot.pkl")
    similarity_df = pd.read_pickle("product_similarity_df.pkl")
    kmeans_model = joblib.load("rfm_kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return df, pivot, similarity_df, kmeans_model, scaler

df, pivot, similarity_df, kmeans_model, scaler = load_data()

st.title("🛒 Shopper Spectrum App")

# ----------- Tab Layout -------------
tab1, tab2 = st.tabs(["📦 Product Recommender", "👤 Customer Segment Predictor"])

# ----------- Product Recommendation -------------
with tab1:
    st.header("🔍 Find Similar Products")
    product_input = st.text_input("Enter Product Name (case-sensitive):")

    if st.button("Get Recommendations"):
        if product_input not in similarity_df.columns:
            st.error("Product not found. Please enter a valid product name.")
        else:
            recommendations = similarity_df[product_input].sort_values(ascending=False).iloc[1:6].index.tolist()
            st.success("Recommended Products:")
            for i, product in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {product}")

# ----------- Customer Segmentation -------------
with tab2:
    st.header("📊 Predict Customer Segment")

    recency = st.number_input("Recency (days since last purchase)", min_value=0)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0)

    if st.button("Predict Segment"):
        input_data = pd.DataFrame([[recency, frequency, monetary]], columns=["Recency", "Frequency", "Monetary"])
        input_scaled = scaler.transform(input_data)
        cluster = kmeans_model.predict(input_scaled)[0]

        # Define cluster labels (update based on analysis)
        cluster_labels = {0: "Occasional", 1: "High-Value", 2: "At-Risk", 3: "Regular"}
        label = cluster_labels.get(cluster, "Unknown")
        st.success(f"🧠 Predicted Segment: **{label}**")
