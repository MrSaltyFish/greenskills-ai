# energy_utils.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

# All your function definitions here:
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    if 'date' in df.columns:
        df.drop(columns=['date'], inplace=True)
    df = df.select_dtypes(include=[np.number])
    return df

def prepare_features(df, target='Appliances'):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns), scaler

def train_models(X_train, X_test, y_train, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100),
        'XGBoost': XGBRegressor(n_estimators=100, verbosity=0),
        'LightGBM': LGBMRegressor(n_estimators=100)
    }

    results_df = pd.DataFrame()

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        temp = pd.DataFrame({
            'Model': name,
            'Actual': y_test.reset_index(drop=True),
            'Predicted': preds
        })

        temp['Green_Score'] = 100 - ((temp['Predicted'] - temp['Predicted'].min()) /
                                     (temp['Predicted'].max() - temp['Predicted'].min()) * 100)
        temp['Green_Score'] = temp['Green_Score'].clip(0, 100)
        results_df = pd.concat([results_df, temp], ignore_index=True)

    return results_df

def compute_feature_importance(X_train, y_train, features):
    model = LGBMRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    return imp_df.sort_values(by='Importance', ascending=False)

def plot_green_score(results_df):
    sns.histplot(results_df['Green_Score'], bins=30, kde=True, color='green')
    plt.title('Green Score Distribution')
    plt.xlabel('Green Score')
    plt.ylabel('Count')
    plt.show()

def plot_actual_vs_predicted(results_df):
    for model_name in results_df['Model'].unique():
        sample = results_df[results_df['Model'] == model_name]
        sns.scatterplot(x='Actual', y='Predicted', data=sample, alpha=0.5)
        plt.plot([sample['Actual'].min(), sample['Actual'].max()],
                 [sample['Actual'].min(), sample['Actual'].max()], color='red', linestyle='--')
        plt.title(f'Actual vs Predicted - {model_name}')
        plt.xlabel('Actual Energy Usage')
        plt.ylabel('Predicted Energy Usage')
        plt.show()

def categorize(score):
    if score >= 80:
        return 'High'
    elif score >= 50:
        return 'Medium'
    else:
        return 'Low'

def show_efficiency_confusion_matrix(results_df):
    results_df['Green_Label'] = results_df['Green_Score'].apply(categorize)
    lightgbm_preds = results_df[results_df['Model'] == 'LightGBM'].copy()
    lightgbm_preds['Actual_Label'] = lightgbm_preds['Actual'].apply(
        lambda x: 'High' if x < 50 else 'Medium' if x < 150 else 'Low')
    lightgbm_preds['Predicted_Label'] = lightgbm_preds['Green_Label']

    cm = confusion_matrix(lightgbm_preds['Actual_Label'], lightgbm_preds['Predicted_Label'],
                          labels=['High', 'Medium', 'Low'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['High', 'Medium', 'Low'])
    disp.plot(cmap='Greens')
    plt.title("Green Efficiency Confusion Matrix (LightGBM)")
    plt.show()

