import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR


# Load the CSV file and read only the first n lines
n = 137  # Adjust this value as needed
file_path = r'C:\Users\Aziz\Desktop\FH SWF\SoSe24\Machine Learning SoSe24\ML Task 2\Heating-data.csv'
data = pd.read_csv(file_path, delimiter='\t', nrows=n)

# Convert Date column to datetime and extract useful features
data['Date'] = pd.to_datetime(data['Date'])
data = data.drop(columns=['Date'])

# Explore the data and handle missing values
print(data.head())
print(data.describe())
data = data.dropna()

# Identify and handle non-numeric columns (if any)
categorical_cols = data.select_dtypes(include=['object']).columns
numerical_cols = data.select_dtypes(exclude=['object']).columns

# Encode categorical data (if any)
encoders = {}
encoded_data = data.copy()
for col in categorical_cols:
    encoder = LabelEncoder()
    encoded_data[col] = encoder.fit_transform(data[col])
    encoders[col] = encoder

# Standardize numerical features
scaler = StandardScaler()
encoded_data[numerical_cols] = scaler.fit_transform(encoded_data[numerical_cols])

# Split the data into features (X) and target variable (y)
X = encoded_data.drop('Gas consumption [kWh/day]', axis=1)
y = encoded_data['Gas consumption [kWh/day]']

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X)
explained_variance_ratio = pca.explained_variance_ratio_

# Cumulative explained variance
cumulative_explained_variance = explained_variance_ratio.cumsum()
print("Cumulative Explained Variance:", cumulative_explained_variance)

# Determine the optimal number of principal components
optimal_components = np.argmax(cumulative_explained_variance >= 0.90) + 1
print(f"Optimal number of components to explain at least 90% variance: {optimal_components}")

# Select the first two principal components
X_pca_two_components = X_pca[:, :2]

# Split the data into training and testing sets using the first two principal components
X_train, X_test, y_train, y_test = train_test_split(X_pca_two_components, y, test_size=0.2, random_state=42)

# Fit a linear regression model on PCA-transformed data
lr = LinearRegression()
lr.fit(X_train, y_train)

# Evaluate the model's performance on PCA-transformed data
y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f'Linear Regression on PCA-Transformed Data - Mean Squared Error (First Two Components): {mse_lr:.2f}')
print(f'Linear Regression on PCA-Transformed Data - R-squared (First Two Components): {r2_lr:.2f}')

# Scree plot of the possible five principal components
plt.figure(figsize=(8, 6))
plt.plot(range(1, 6), pca.explained_variance_ratio_[:5], marker='o', linestyle='-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot of the Possible Five Principal Components')
plt.show()

### Additional Analysis ###

# Analyze PCA components to identify essential variables
components_df = pd.DataFrame(pca.components_, columns=X.columns)
print("PCA Components:")
print(components_df)

# Display the explained variance ratio
print("Explained Variance Ratio by Components:")
print(explained_variance_ratio)

# Heatmap of the loads of the first five principal components with respect to the features
# Assuming X is your feature matrix
# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Performing PCA
pca = PCA()
pca.fit(X_scaled)

# Extracting the loadings of the first five principal components
loadings = pca.components_[:5]
loadings_df = pd.DataFrame(loadings.T, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], index=X.columns)

# Printing the loadings
print("Loadings of the first five principal components:\n", loadings_df)

# Plotting the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(loadings_df, annot=True, cmap='coolwarm')
plt.title('Heatmap of the Loads of the First Five Principal Components')
plt.show()

### 3D Plot with Regression Plane ###

# Fit a linear regression model with 3 components for 3D plot
X_pca_three_components = X_pca[:, :3]
X_train_3d, X_test_3d, y_train_3d, y_test_3d = train_test_split(X_pca_three_components, y, test_size=0.9, random_state=42)
lr_3d = LinearRegression()
lr_3d.fit(X_train_3d, y_train_3d)

# Predict and create a meshgrid for the plane
y_pred_3d = lr_3d.predict(X_test_3d)
xx, yy = np.meshgrid(np.linspace(X_test_3d[:, 0].min(), X_test_3d[:, 0].max(), 10),
                     np.linspace(X_test_3d[:, 1].min(), X_test_3d[:, 1].max(), 10))
zz = lr_3d.intercept_ + lr_3d.coef_[0] * xx + lr_3d.coef_[1] * yy

# 3D plot with actual and predicted values and regression plane
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test_3d[:, 0], X_test_3d[:, 1], y_test_3d, color='blue', label='Actual values')
ax.scatter(X_test_3d[:, 0], X_test_3d[:, 1], y_pred_3d, color='red', label='Predicted values')
ax.plot_surface(xx, yy, zz, alpha=0.5, rstride=100, cstride=100)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('Gas consumption [kWh/day]')
plt.title('3D Scatter Plot with Regression Plane')
plt.legend()
plt.show()

### Pipeline ###

# Create a pipeline for PCA and Linear Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=optimal_components)),
    ('regressor', LinearRegression())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the pipeline's performance
y_pred = pipeline.predict(X_test)
mse_pipeline = mean_squared_error(y_test, y_pred)
r2_pipeline = r2_score(y_test, y_pred)
print(f'Pipeline - Mean Squared Error: {mse_pipeline:.2f}')
print(f'Pipeline - R-squared: {r2_pipeline:.2f}')

# Compare linear regression on original data vs. PCA-transformed data
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression on original data
lr_orig = LinearRegression()
lr_orig.fit(X_train_orig, y_train_orig)
y_pred_orig = lr_orig.predict(X_test_orig)

mse_orig = mean_squared_error(y_test_orig, y_pred_orig)
r2_orig = r2_score(y_test_orig, y_pred_orig)
model_score_orig = lr_orig.score(X_test_orig, y_test_orig)

print(f'Original Data - Mean Squared Error: {mse_orig:.2f}, R-squared: {r2_orig:.2f}, Model Score (R²): {model_score_orig:.2f}')

# Compare with PCA-transformed data
print(f'PCA-Transformed Data - Mean Squared Error: {mse_lr:.2f}, R-squared: {r2_lr:.2f}')

# Conclusion
print("Conclusion:")
print(f"The optimal number of principal components considered: {optimal_components}")
print(f"Linear Regression on PCA-Transformed Data - Mean Squared Error: {mse_lr:.2f}, R-squared: {r2_lr:.2f}")
print(f"Linear Regression on Original Data - Mean Squared Error: {mse_orig:.2f}, R-squared: {r2_orig:.2f}, Model Score (R²): {model_score_orig:.2f}")
print(f"Pipeline - Mean Squared Error: {mse_pipeline:.2f}, R-squared: {r2_pipeline:.2f}")






# Random Forest Regression
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f'Random Forest Regression - Mean Squared Error: {mse_rf:.2f}, R-squared: {r2_rf:.2f}')

# Gradient Boosting Regression
gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
print(f'Gradient Boosting Regression - Mean Squared Error: {mse_gb:.2f}, R-squared: {r2_gb:.2f}')

# Support Vector Regression (SVR)
svr = SVR()
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)
print(f'Support Vector Regression (SVR) - Mean Squared Error: {mse_svr:.2f}, R-squared: {r2_svr:.2f}')

# Comparison plot for the models
models = ['Linear Regression (PCA)', 'Random Forest', 'Gradient Boosting', 'SVR']
mse_scores = [mse_lr, mse_rf, mse_gb, mse_svr]
r2_scores = [r2_lr, r2_rf, r2_gb, r2_svr]

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.bar(models, mse_scores, color=['blue', 'green', 'orange', 'red'])
plt.xlabel('Models')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error Comparison')

plt.subplot(1, 2, 2)
plt.bar(models, r2_scores, color=['blue', 'green', 'orange', 'red'])
plt.xlabel('Models')
plt.ylabel('R-squared')
plt.title('R-squared Comparison')

plt.tight_layout()
plt.show()
