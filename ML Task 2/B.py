import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# Load the CSV file and read only the first n lines
n = 137  # Adjust this value as needed
file_path = r'C:\Users\Aziz\Desktop\FH SWF\SoSe24\Machine Learning SoSe24\ML Task 2\Heating-data.csv'
data = pd.read_csv(file_path, delimiter='\t', nrows=n)

# Convert Date column to datetime and extract useful features
data['Date'] = pd.to_datetime(data['Date'])
data = data.drop(columns=['Date'])

# Drop missing values
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

# Split the data into features (X) and target variable (y)
X = encoded_data.drop('Gas consumption [kWh/day]', axis=1)
y = encoded_data['Gas consumption [kWh/day]']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Split the data into training and testing sets using the first two principal components
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Fit a linear regression model on PCA-transformed data
lr = LinearRegression()
lr.fit(X_train, y_train)

# Evaluate the model's performance on PCA-transformed data
y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f'Linear Regression on PCA-Transformed Data - Mean Squared Error (First Two Components): {mse_lr:.2f}')
print(f'Linear Regression on PCA-Transformed Data - R-squared (First Two Components): {r2_lr:.2f}')

# Plot 3D scatter plot with regression plane
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Scatter plot of actual values
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='blue', label='Actual values')

# Scatter plot of predicted values
ax.scatter(X_test[:, 0], X_test[:, 1], y_pred_lr, color='red', label='Predicted values')

# Plot the regression plane
# Meshgrid for regression plane
x0 = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 10)
x1 = np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), 10)
xx0, xx1 = np.meshgrid(x0, x1)
zz = lr.intercept_ + lr.coef_[0] * xx0 + lr.coef_[1] * xx1

# Plotting the regression plane
ax.plot_surface(xx0, xx1, zz, alpha=0.5, rstride=100, cstride=100, color='green', label='Regression plane')

# Set labels and title
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('Gas consumption [kWh/day]')
plt.title('3D Scatter Plot with Regression Plane')

# Legend and display
plt.legend()
plt.show()
