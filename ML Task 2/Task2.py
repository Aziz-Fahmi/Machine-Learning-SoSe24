import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
from mpl_toolkits.mplot3d import Axes3D

# Reading the CSV file into a DataFrame
file_path = "Heating-data.csv"
df = pd.read_csv(file_path, delimiter='\t')

# Display the first few rows of the DataFrame and column names
print(df.head())
print(df.columns)

# Selecting predictor variables (excluding Date and Gas consumption)
X = df.drop(columns=["Date", "Gas consumption [kWh/day]"])
y = df["Gas consumption [kWh/day]"]

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Displaying the explained variance ratio for each principal component
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance ratio: {explained_variance}")

# Scree plot
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance * 100, alpha=0.7, align='center')
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.xticks(range(1, len(explained_variance) + 1))
plt.grid(True)
plt.show()

# Displaying the loading scores
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(len(pca.components_))], index=X.columns)
print(loadings)

# Heatmap of PC loadings
plt.figure(figsize=(10, 8))
sns.heatmap(loadings, annot=True, cmap='coolwarm')
plt.title('Heatmap of Principal Component Loadings')
plt.show()

# Biplot of the first two PCs
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.colorbar(scatter, label='Gas Consumption')

for i in range(loadings.shape[0]):
    ax.arrow(0, 0, loadings.iloc[i, 0], loadings.iloc[i, 1], color='r', alpha=0.7, width=0.01)
    ax.text(loadings.iloc[i, 0] * 1.05, loadings.iloc[i, 1] * 1.05,
            f"{loadings.index[i]} ({loadings.iloc[i, 0]:.2f}, {loadings.iloc[i, 1]:.2f})",
            color='g', ha='left', va='center')

# Adding explained variance to the axes and title
ax.set_xlabel(f"Principal Component 1 ({explained_variance[0]*100:.2f}%)")
ax.set_ylabel(f"Principal Component 2 ({explained_variance[1]*100:.2f}%)")
plt.title(f"Biplot of the First Two Principal Components\nExplained Variance: {explained_variance[0]*100 + explained_variance[1]*100:.2f}%")
plt.grid(True)
plt.show()

# Choosing the first two principal components
X_pca_2 = X_pca[:, :2]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca_2, y, test_size=0.2, random_state=42)

# Performing Linear Regression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# Predicting on the test set
y_pred_linear = linear_regressor.predict(X_test)

# Evaluating the model
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
print(f"Linear Regression Mean Squared Error: {mse_linear}")
print(f"Linear Regression R-squared: {r2_linear}")

# Performing Random Forest Regression for comparison
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Predicting on the test set
y_pred_rf = rf_regressor.predict(X_test)

# Evaluating the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest Regression Mean Squared Error: {mse_rf}")
print(f"Random Forest Regression R-squared: {r2_rf}")

# Performing Gradient Boosting Regression for comparison
gb_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_regressor.fit(X_train, y_train)

# Predicting on the test set
y_pred_gb = gb_regressor.predict(X_test)

# Evaluating the model
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
print(f"Gradient Boosting Regression Mean Squared Error: {mse_gb}")
print(f"Gradient Boosting Regression R-squared: {r2_gb}")

# Performing Support Vector Regression for comparison
svr_regressor = SVR(kernel='linear')
svr_regressor.fit(X_train, y_train)

# Predicting on the test set
y_pred_svr = svr_regressor.predict(X_test)

# Evaluating the model
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)
print(f"Support Vector Regression Mean Squared Error: {mse_svr}")
print(f"Support Vector Regression R-squared: {r2_svr}")

# Performing XGBoost Regression for comparison
xgb_regressor = XGBRegressor(n_estimators=100, random_state=42)
xgb_regressor.fit(X_train, y_train)

# Predicting on the test set
y_pred_xgb = xgb_regressor.predict(X_test)

# Evaluating the model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"XGBoost Regression Mean Squared Error: {mse_xgb}")
print(f"XGBoost Regression R-squared: {r2_xgb}")

# Comparing results
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'Support Vector Regression', 'XGBoost'],
    'Mean Squared Error': [mse_linear, mse_rf, mse_gb, mse_svr, mse_xgb],
    'R-squared': [r2_linear, r2_rf, r2_gb, r2_svr, r2_xgb]
})

print(results)

# 3D scatter plot with Linear Regression
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='b', label='Actual')

# Creating a grid for plotting the regression plane
xx, yy = np.meshgrid(np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), num=30),
                     np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), num=30))
zz = linear_regressor.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plotting the regression plane
ax.plot_surface(xx, yy, zz, color='r', alpha=0.5)
ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}%)')
ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}%)')
ax.set_zlabel('Gas Consumption')
ax.set_title('3D Scatterplot with Linear Regression')
plt.legend()
plt.show()

# 2D scatter plot with model comparison
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_linear, label='Linear Regression', alpha=0.5)
plt.scatter(y_test, y_pred_rf, label='Random Forest Regression', alpha=0.5)
plt.scatter(y_test, y_pred_gb, label='Gradient Boosting Regression', alpha=0.5)
plt.scatter(y_test, y_pred_svr, label='Support Vector Regression', alpha=0.5)
plt.scatter(y_test, y_pred_xgb, label='XGBoost Regression', alpha=0.5)
plt.xlabel('Actual Gas Consumption')
plt.ylabel('Predicted Gas Consumption')
plt.title('Actual vs Predicted Gas Consumption')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.grid(True)
plt.legend()
plt.show()
