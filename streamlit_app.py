# Streamlit core
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from energy_utils import (
    load_and_clean_data,
    prepare_features,
    scale_features,
    train_models,
    compute_feature_importance,
    plot_green_score,
    plot_actual_vs_predicted,
    show_efficiency_confusion_matrix
)
st.title("🌿 Green Energy Prediction Dashboard")

uploaded_file = st.file_uploader("Upload your energy dataset", type=["csv"])

if uploaded_file:
    df = load_and_clean_data(uploaded_file)
    X, y = prepare_features(df)
    X_scaled, scaler = scale_features(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    st.subheader("📊 Model Performance & Green Score")
    results_df = train_models(X_train, X_test, y_train, y_test)
    st.dataframe(results_df.head())

    st.download_button("📁 Download Results CSV", results_df.to_csv(index=False), file_name="green_scores.csv")

    st.subheader("📈 Green Score Distribution")
    fig1 = plt.figure()
    plot_green_score(results_df)
    st.pyplot(fig1)

    st.subheader("📌 Actual vs Predicted (per Model)")
    plot_actual_vs_predicted(results_df)

    st.subheader("🔥 Feature Importance")
    imp_df = compute_feature_importance(X_train, y_train, X.columns)
    fig2 = plt.figure()
    sns.barplot(data=imp_df.head(10), x='Importance', y='Feature', palette='YlGn')
    plt.title('Top 10 Important Features')
    st.pyplot(fig2)

    st.subheader("⚖️ Green Efficiency Classification")
    show_efficiency_confusion_matrix(results_df)
