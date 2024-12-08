import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# FutureWarning uyarılarını susturma
warnings.filterwarnings("ignore")

# Veri yükleme (California Housing dataset)
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame

# Features (X) and target variable (y)
X = df.drop(columns='MedHouseVal')  # MedHouseVal is the target variable
y = df['MedHouseVal']

# Handle missing values by replacing them with column mean
X.fillna(X.mean(), inplace=True)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define regression models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1, max_iter=10000),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Evaluate each model
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate RMSE and R²
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results[model_name] = {"RMSE": rmse, "R²": r2}
    print(f"{model_name}: RMSE={rmse:.2f}, R²={r2:.2f}")

# Plotting Feature Importances for Random Forest
rf_model = models["Random Forest"]
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot Feature Importance and save it as an image
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), palette='viridis')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')  # Save the figure as a PNG file
plt.show()

# Residual Analysis for Random Forest and save it as an image
residuals = y_test - rf_model.predict(X_test)
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, color='blue', bins=20)
plt.title('Residuals Distribution (Random Forest)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('residuals_distribution.png')  # Save the figure as a PNG file
plt.show()

# Residual Plot for Random Forest and save it as an image
plt.figure(figsize=(8, 5))
plt.scatter(rf_model.predict(X_test), residuals, alpha=0.7, color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot (Random Forest)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.tight_layout()
plt.savefig('residual_plot.png')  # Save the figure as a PNG file
plt.show()

# Barplot for model comparison (RMSE and R²) and save it as an image
model_names = list(results.keys())
rmse_values = [results[model]["RMSE"] for model in results]
r2_values = [results[model]["R²"] for model in results]

# Plot RMSE Comparison and save it as an image
plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=rmse_values, palette='Blues_d')
plt.title('Model RMSE Comparison')
plt.ylabel('RMSE')
plt.xlabel('Model')
plt.tight_layout()
plt.savefig('rmse_comparison.png')  # Save the figure as a PNG file
plt.show()

# Plot R² Comparison and save it as an image
plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=r2_values, palette='viridis')
plt.title('Model R² Comparison')
plt.ylabel('R²')
plt.xlabel('Model')
plt.tight_layout()
plt.savefig('r2_comparison.png')  # Save the figure as a PNG file
plt.show()
