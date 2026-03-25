# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 2: Load Dataset (FULL PATH use kar raha hoon)
data = pd.read_csv("C:/Users/HP/OneDrive/Documents/Desktop/house-price-project/data.csv")

# Step 3: Show Data
print("Dataset Preview:")
print(data.head())

# Step 4: Features and Target
X = data[['area', 'bedrooms', 'bathrooms', 'age']]
y = data['price']

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Model Accuracy
y_pred = model.predict(X_test)
print("\nModel Accuracy (R2 Score):", r2_score(y_test, y_pred))

# Step 8: User Input
print("\nEnter house details to predict price:")

area = float(input("Enter area (sq ft): "))
bedrooms = int(input("Enter bedrooms: "))
bathrooms = int(input("Enter bathrooms: "))
age = int(input("Enter age of house: "))

# Step 9: Prediction
prediction = model.predict(np.array([[area, bedrooms, bathrooms, age]]))

print("\n🏠 Predicted House Price:", int(prediction[0]))