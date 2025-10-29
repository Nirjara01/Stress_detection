import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Load your dataset (make sure it exists in the same folder)
df = pd.read_csv("stress_data2.csv")  # Use your dataset name

# Use only the two features for voting validation
X = df[['work_stress', 'overwhelm_freq']]
y = df['Stress_Level']  # Should be 0 (Low), 1 (Moderate), 2 (High)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Gradient Boosting Model Accuracy: {accuracy:.2f}")

# Save the model and scaler
joblib.dump(model, "gradient_boosting_model.pkl")
joblib.dump(scaler, "scalerr.pkl")
print("Saved model as 'gradient_boosting_model.pkl' and scaler as 'scaler.pkl'")
