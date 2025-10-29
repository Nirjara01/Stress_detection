import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier

class StressValidationSystem:
    def __init__(self):
        self.model = joblib.load('gradient_boosting_model.pkl')  # Trained GB model
        self.scaler = joblib.load('scaler.pkl')
        self.response_map = {
            "Never": 0,
            "Rarely": 1,
            "Sometimes": 2,
            "Often": 3
        }
        self.feature_names = ['work_stress', 'sleep_issues', 'overwhelm_freq',
                             'concentration_diff', 'relaxation_diff']

    def validate_with_gb_voting(self, responses):
        q1 = self.response_map.get(responses[0], 0)  # work_stress
        q3 = self.response_map.get(responses[2], 0)  # overwhelm_freq
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
