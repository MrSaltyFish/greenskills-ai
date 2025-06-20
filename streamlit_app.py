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
    st.success("📁 File uploaded successfully!")

    uploaded_file.seek(0)
    df = load_and_clean_data(uploaded_file)
    st.write("✅ Data preview:", df.head())

    # 🛠️ Feature Engineering
    try:
        X, y = prepare_features(df)
        st.write(f"✅ Features and target extracted. (X: {X.shape}, y: {y.shape})")
    except Exception as e:
        st.error(f"❌ Feature extraction failed: {e}")
        st.stop()

    try:
        X_scaled, scaler = scale_features(X)
        st.write("✅ Features scaled successfully.")
    except Exception as e:
        st.error(f"❌ Feature scaling failed: {e}")
        st.stop()

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        st.write("✅ Train-test split done.")
    except Exception as e:
        st.error(f"❌ Data splitting failed: {e}")
        st.stop()


    # 🤖 Train Models
    try:
        with st.spinner("🏗️ Training models..."):
            results_df = train_models(X_train, X_test, y_train, y_test)
        st.success("✅ Models trained successfully.")
        st.subheader("📊 Prediction Sample")
        st.dataframe(results_df.head(10))

        # 💾 Download Button
        st.download_button("📁 Download Results CSV",
                           data=results_df.to_csv(index=False),
                           file_name="green_score_results.csv",
                           mime="text/csv")
    except Exception as e:
        st.error(f"❌ Model training error: {e}")
        st.stop()

    # 📈 Green Score Distribution
    st.subheader("📈 Green Score Distribution")
    try:
        fig1 = plt.figure()
        plot_green_score(results_df)
        st.pyplot(fig1)
    except Exception as e:
        st.error(f"❌ Failed to plot Green Score: {e}")

    # 🎯 Actual vs Predicted per Model
    st.subheader("📌 Actual vs Predicted (per Model)")
    try:
        fig2 = plt.figure()
        # plot_actual_vs_predicted(results_df)
        # st.pyplot(fig2)
        for fig in plot_actual_vs_predicted(results_df):
            st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Failed to plot Actual vs Predicted: {e}")

    # 🔥 Feature Importance
    st.subheader("🔥 Feature Importance (LightGBM)")
    try:
        imp_df = compute_feature_importance(X_train, y_train, X.columns)
        fig3 = plt.figure()
        sns.barplot(data=imp_df.head(10), x='Importance', y='Feature', palette='YlGn')
        plt.title('Top 10 Important Features')
        st.pyplot(fig3)
    except Exception as e:
        st.error(f"❌ Failed to compute or plot feature importances: {e}")

    # ⚖️ Confusion Matrix
    st.subheader("⚖️ Green Efficiency Classification")
    try:
        fig4 = show_efficiency_confusion_matrix(results_df)
        st.pyplot(fig4)
    except Exception as e:
        st.error(f"❌ Failed to plot confusion matrix: {e}")

