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



    # df = pd.read_csv(uploaded_file)
    # st.write("📄 First few rows of uploaded data:")
    # st.dataframe(df.head())
    # uploaded_file.seek(0)

if uploaded_file:
    st.success("📁 File uploaded successfully!")
    uploaded_file.seek(0)
    df = load_and_clean_data(uploaded_file)
    st.write("✅ Data preview:", df.head())

    try:
        X, y = prepare_features(df)
        st.write("✅ Feature matrix and target ready.")
    except Exception as e:
        st.error(f"❌ Error in feature prep: {e}")
        st.stop()

    try:
        X_scaled, scaler = scale_features(X)
        st.write("✅ Features scaled.")
    except Exception as e:
        st.error(f"❌ Scaling error: {e}")
        st.stop()

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        st.write("✅ Data split into train/test.")
    except Exception as e:
        st.error(f"❌ Train/test split error: {e}")
        st.stop()

    try:
        with st.spinner("🏗️ Training models..."):
            results_df = train_models(X_train, X_test, y_train, y_test)
        st.success("✅ Models trained successfully.")
        st.dataframe(results_df.head())
    except Exception as e:
        st.error(f"❌ Model training error: {e}")
        st.stop()

    # 📈 Green Score Distribution
    st.subheader("📈 Green Score Distribution")
    try:
        fig1, ax1 = plt.subplots()
        sns.histplot(results_df['Green_Score'], bins=30, kde=True, color='green', ax=ax1)
        ax1.set_title('Green Score Distribution')
        ax1.set_xlabel('Green Score')
        ax1.set_ylabel('Count')
        st.pyplot(fig1)
    except Exception as e:
        st.error(f"Failed to plot Green Score Distribution: {e}")

    # 📌 Actual vs Predicted for all models
    st.subheader("📌 Actual vs Predicted (per Model)")
    try:
        model_names = results_df['Model'].unique()
        for model_name in model_names:
            sample = results_df[results_df['Model'] == model_name]
            fig2, ax2 = plt.subplots()
            sns.scatterplot(x='Actual', y='Predicted', data=sample, ax=ax2, alpha=0.5)
            ax2.plot([sample['Actual'].min(), sample['Actual'].max()],
                    [sample['Actual'].min(), sample['Actual'].max()],
                    color='red', linestyle='--')
            ax2.set_title(f'Actual vs Predicted - {model_name}')
            ax2.set_xlabel('Actual Energy Usage')
            ax2.set_ylabel('Predicted Energy Usage')
            st.pyplot(fig2)
    except Exception as e:
        st.error(f"Failed to plot Actual vs Predicted: {e}")



# if uploaded_file:
#     df = load_and_clean_data(uploaded_file)
#     X, y = prepare_features(df)
#     st.write("✅ Columns in dataset:", df.columns.tolist())
#     st.write("✅ Shape of X and y:", X.shape, y.shape)

#     X_scaled, scaler = scale_features(X)

#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#     st.subheader("📊 Model Performance & Green Score")
#     try:
#         results_df = train_models(X_train, X_test, y_train, y_test)
#     except Exception as e:
#         st.error(f"🚨 Model training failed: {e}")
#     st.stop()

#     if not results_df.empty:
#         st.success("✅ Models trained successfully!")
#         st.dataframe(results_df.head(10))
#     else:
#         st.warning("⚠️ No results generated. Check model logic or input data.")


#     st.download_button("📁 Download Results CSV", results_df.to_csv(index=False), file_name="green_scores.csv")

#     st.subheader("📈 Green Score Distribution")
#     fig1 = plt.figure()
#     plot_green_score(results_df)
#     st.pyplot(fig1)

#     st.subheader("📌 Actual vs Predicted (per Model)")
#     plot_actual_vs_predicted(results_df)

#     st.subheader("🔥 Feature Importance")
#     imp_df = compute_feature_importance(X_train, y_train, X.columns)
#     fig2 = plt.figure()
#     sns.barplot(data=imp_df.head(10), x='Importance', y='Feature', palette='YlGn')
#     plt.title('Top 10 Important Features')
#     st.pyplot(fig2)

#     st.subheader("⚖️ Green Efficiency Classification")
#     show_efficiency_confusion_matrix(results_df)
