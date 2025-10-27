# KNN-Classification
python program to build Knn classification

# KNN classification model
# Author: Demo program

# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset (you can replace this with your own CSV file)
# Example: data = pd.read_csv("your_dataset.csv")
data = pd.DataFrame({
    'Feature1': [2, 4, 4, 6, 6, 8, 10, 12],
    'Feature2': [4, 2, 4, 2, 6, 6, 8, 10],
    'Label': ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B']
})

# Display first few rows
print("Dataset:\n", data)

# Separate features (X) and labels (y)
X = data[['Feature1', 'Feature2']]
y = data['Label']  # Corrected column name from 'label' to 'Label'

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Make Predictions
y_pred = knn.predict(X_test)

# Evaluate model
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

