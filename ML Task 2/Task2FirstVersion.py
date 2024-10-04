import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# Import data as a Pandas DataFrame and preprocess them for scikit-learn:
file_path = r'C:\Users\Aziz\Desktop\FH SWF\SoSe24\Machine Learning SoSe24\ML Task 2\Heating-data.csv'
df = pd.read_csv(file_path, delimiter='\t')

# Define features and target
features = ["Sunshine duration [h/day]", "Outdoor temperature [°C]", "Solar yield [kWh/day]", "Solar pump [h/day]", "Valve [h/day]"]
target = "Gas consumption [kWh/day]"

X = df[features].values  # extracts feature values as a matrix
y = df[target].values  # extracts target values as a one-dimensional array

# Choose by random 30 % of data as test data, i.e., 70 % as training data:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Fit and predict with a pipeline of scaling, PCA, and linear regression:
pipe = make_pipeline(StandardScaler(), PCA(2), LinearRegression())
pipe.fit(X_train, y_train)

# Print model score:
print("score (train values): ", f"{pipe.score(X_train, y_train):.2%}")
print("score (test values):", f"{pipe.score(X_test, y_test):.2%}")

# Plot 3D scatter plot:
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Project data onto the principal components:
X_scaled = pipe.steps[0][1].transform(X)
X_trans = pipe.steps[1][1].transform(X_scaled)
y_pred = pipe.predict(X)

# Scatter plot of actual values
ax.scatter(X_trans[:,0], X_trans[:,1], y, marker="o", c='red', label='actual values')
ax.set_xlabel("PC 1"), ax.set_ylabel("PC 2"), ax.set_zlabel(target)
ax.view_init(azim=-60, elev=20)  # position of camera

# Plot regression plane:
x0 = np.linspace(X_trans[:,0].min(), X_trans[:,0].max(), num=2)
x1 = np.linspace(X_trans[:,1].min(), X_trans[:,1].max(), num=2)
xx0, xx1 = np.meshgrid(x0, x1)
X0, X1 = xx0.ravel(), xx1.ravel()
yy = pipe.steps[2][1].predict(np.c_[X0, X1]).ravel()
ax.plot_trisurf(X0, X1, yy, linewidth=0, alpha=0.3)

plt.tight_layout()
plt.show()
