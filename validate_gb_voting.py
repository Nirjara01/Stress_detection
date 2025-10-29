import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier

class StressValidationSystem:
    def __init__(self):
        self.model = joblib.load('gradient_boosting_model.pkl')  # Trained GB model
        self.scaler = joblib.load('scalerr.pkl')
        self.response_map = {
            "Never": 0,
            "Rarely": 1,
            "Sometimes": 2,
            "Often": 3
        }
        self.feature_names = ['work_stress',  'overwhelm_freq']

    def validate_with_gb_voting(self, responses):
        q1 = self.response_map.get(responses[0], 0)  # work_stress
        q3 = self.response_map.get(responses[0], 0)  # overwhelm_freq
        q2 = self.response_map.get(responses[0], 0)
        q4 = self.response_map.get(responses[0], 0)
        q5 = self.response_map.get(responses[0], 0)
        X = pd.DataFrame([[q1, q3]], columns=['work_stress', 'overwhelm_freq'])
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)[0]
        return ['Low', 'Moderate', 'High'][pred]

# Example
if __name__ == '__main__':
    system = StressValidationSystem()
    sample_responses = ["Sometimes", "Often", "Often", "Rarely", "Never"]
    result = system.validate_with_gb_voting(sample_responses)
    print(f"Voting Validation Prediction: {result}")
