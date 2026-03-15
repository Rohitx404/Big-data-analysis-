
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load data
df = pd.read_csv("dataset.csv")

# Basic cleaning
df = df.dropna()
df = df[df['Sales'] > 0]

# Features and target
X = df[['Sales','Discount','Quantity']]
y = df['Profit']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl","wb"))

print("Model trained and saved successfully!")
