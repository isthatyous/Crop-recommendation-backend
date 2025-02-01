# Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

df = pd.read_csv("Crop_recommendation.csv")

print("First few rows of the dataset:")
print(df.head())

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values check:")
print(df.isnull().sum())

# Visualization
plt.figure(figsize=(10, 6))
sns.boxplot(data=df.drop(columns=['label']), orient='h')
plt.title("Boxplot of Numerical Features")
plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='label')
plt.title("Distribution of Labels (Crops)")
plt.show()

df['NPK_ratio'] = df['N'] / (df['P'] + df['K'] + 1)
df['Temp_Rainfall'] = df['temperature'] * df['rainfall']

for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
    df[f'{col}_norm'] = (df[col] - df[col].mean()) / df[col].std()

X = df.drop(columns=['label', 'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Save model
with open('crop_recommendation_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as 'crop_recommendation_model.pkl'")
