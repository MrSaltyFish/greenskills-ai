
# ✅ Streamlit App: Green Score Predictor
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

import streamlit as st
# Load model and data
@st.cache_data
def load_model():
    model = LGBMRegressor(n_estimators=100)
    return model

@st.cache_data
def load_scaler(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

# Load dataset for feature reference
df = pd.read_csv("energydata_complete.csv")
df.drop(columns=['date'], inplace=True)
df.dropna(inplace=True)
df = df.select_dtypes(include=[np.number])

X = df.drop(columns=['Appliances'])
y = df['Appliances']
scaler = load_scaler(X)
model = load_model()
model.fit(scaler.transform(X), y)

# Streamlit Interface
st.title("🔋 Green Score Predictor for Appliances")
st.markdown("Estimate energy consumption and green score based on building parameters.")

# Sidebar input
st.sidebar.header("Input Sensor Readings")
input_data = {}
for col in X.columns[:10]:
    input_data[col] = st.sidebar.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))

# Optional: Add remaining features as expandable
with st.sidebar.expander("Advanced Features"):
    for col in X.columns[10:]:
        input_data[col] = st.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))

# Prediction
data_df = pd.DataFrame([input_data])
data_scaled = scaler.transform(data_df)
pred = model.predict(data_scaled)[0]

# Green Score Calculation
pred_min = y.min()
pred_max = y.max()
green_score = 100 - ((pred - pred_min) / (pred_max - pred_min)) * 100
green_score = np.clip(green_score, 0, 100)

# Display results
st.metric("🔌 Predicted Appliance Consumption (Wh)", f"{pred:.2f} Wh")
st.metric("🌿 Green Score", f"{green_score:.1f} / 100")

# Plotting
st.subheader("📊 Feature Impact (Top 10)")
model.fit(scaler.transform(X), y)
importances = model.feature_importances_
imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
imp_df = imp_df.sort_values(by='Importance', ascending=False).head(10)
sns.barplot(data=imp_df, x='Importance', y='Feature', palette='Greens_r')
st.pyplot(plt.gcf())