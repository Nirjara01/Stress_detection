import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, ConfusionMatrixDisplay, f1_score)
from sklearn.inspection import permutation_importance
import joblib
import numpy as np
import matplotlib.pyplot as plt

def train_stress_model():
    try:
        # --------------------------------------------------
        # 📊 Dataset Loading & Validation
        # --------------------------------------------------
        data = pd.read_csv('stress_data1.csv')
        
        # Validate dataset structure
        required_columns = ['work_stress', 'sleep_issues', 'overwhelm_freq',
                           'concentration_diff', 'relaxation_diff', 'Stress_Level']
        if not all(col in data.columns for col in required_columns):
            missing = set(required_columns) - set(data.columns)
            raise ValueError(f"Missing columns in dataset: {missing}")

        # --------------------------------------------------
        # 🔧 Data Preprocessing
        # --------------------------------------------------
        print("\n🔍 Dataset Summary:")
        print(f"Total Samples: {len(data)}")
        print("Class Distribution:")
        print(data['Stress_Level'].value_counts(normalize=True))

        # Handle missing values
        data.dropna(inplace=True)
        
        # Separate features and target
        X = data[required_columns[:-1]]
        y = data['Stress_Level']
        
        # --------------------------------------------------
        # ⚙️ Feature Engineering
        # --------------------------------------------------
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # --------------------------------------------------
        # 🧠 Model Training & Evaluation
        # --------------------------------------------------
        model = GaussianNB()
        print("\n⚙️ Training model with 5-fold cross-validation...")
        cv_scores = cross_val_score(model, X_scaled, y, cv=5)
        print(f"Cross-Validation Accuracy: {cv_scores.mean():.2%} (±{cv_scores.std():.2%})")

        model.fit(X_train, y_train)
        
        # --------------------------------------------------
        # 📈 Performance Metrics
        # --------------------------------------------------
        y_pred = model.predict(X_test)
        
        print("\n🔬 Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
        print(f"Macro F1-Score: {f1_score(y_test, y_pred, average='macro'):.2%}")
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Low', 'Moderate', 'High']))
        

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        labels = ['Low', 'Moderate', 'High']

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix - Stress Level Prediction")
        plt.show()
        # --------------------------------------------------
        # 📊 Feature Importance Analysis
        # --------------------------------------------------
        print("\n📌 Feature Importance:")
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        for idx, score in enumerate(result.importances_mean):
            print(f"{X.columns[idx]}: {score:.2%}")

        # --------------------------------------------------
        # 💾 Save Artifacts
        # --------------------------------------------------
        joblib.dump(model, 'stress_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        print("\n✅ Training artifacts saved successfully!")

    except FileNotFoundError:
        print("\n❌ Error: Dataset file not found")
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")

        
        accuracy = accuracy_score(y_test, y_pred)
        plt.figure(figsize=(4, 4))
        plt.bar(['Model Accuracy'], [accuracy], color='green')
        plt.ylim(0, 1)
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.text(0, accuracy - 0.05, f'{accuracy:.2f}', ha='center', color='white', fontweight='bold')
        plt.show()

if __name__ == "__main__":
    train_stress_model()