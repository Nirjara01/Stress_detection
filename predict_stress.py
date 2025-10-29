
import pandas as pd
import joblib

class StressDetectionSystem:
    def __init__(self):
        self.model = joblib.load('stress_model.pkl')
        self.scaler = joblib.load('scaler.pkl')
        self.feature_names = ['work_stress', 'sleep_issues', 'overwhelm_freq',
                             'concentration_diff', 'relaxation_diff']
        
        self.questions = [
            "How often do you feel stressed at work?",
            "Do you experience trouble sleeping due to anxiety?",
            "How frequently do you feel overwhelmed by responsibilities?",
            "How often do you experience difficulty concentrating due to stress?",
            "Do you find it hard to relax even during leisure time?"
        ]
        
        self.response_map = {
            "Never": 0,
            "Rarely": 1,
            "Sometimes": 2,
            "Often": 3
        }
        
        self.suggestion_engine = {
            'work_stress': {
                0: "Maintain good work-life balance",
                1: "Consider time management techniques",
                2: "Try prioritization strategies",
                3: "Seek professional workload management advice"
            },
            'sleep_issues': {
                0: "Keep consistent sleep schedule",
                1: "Limit caffeine intake after noon",
                2: "Try relaxation techniques before bed",
                3: "Consult a sleep specialist"
            },
            'overwhelm_freq': {
                0: "Maintain current task management approach",
                1: "Use task prioritization methods",
                2: "Practice delegation techniques",
                3: "Consider professional counseling"
            },
            'concentration_diff': {
                0: "Maintain focus strategies",
                1: "Take regular short breaks",
                2: "Try mindfulness exercises",
                3: "Consult cognitive behavioral therapist"
            },
            'relaxation_diff': {
                0: "Continue current relaxation practices",
                1: "Add light physical activities",
                2: "Practice deep breathing exercises",
                3: "Try guided meditation programs"
            }
        }
    
    def validate_responses(self, responses):
        valid_choices = list(self.response_map.keys())
        for resp in responses:
            if resp not in valid_choices:
                raise ValueError(f"Invalid response '{resp}'. Valid choices: {valid_choices}")
        return True
    
    def predict(self, responses):
        self.validate_responses(responses)
        
        # Convert to numerical features
        X = pd.DataFrame([list(map(self.response_map.get, responses))], 
                        columns=self.feature_names)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        stress_level = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Generate explanations
        suggestions = self._generate_suggestions(responses)
        feature_impacts = self._analyze_feature_impacts(X_scaled)
        
        return {
            'stress_level': ['Low', 'Moderate', 'High'][stress_level],
            'confidence': round(probabilities.max() * 100, 2),
            'probabilities': {
                'Low': round(probabilities[0] * 100, 2),
                'Moderate': round(probabilities[1] * 100, 2),
                'High': round(probabilities[2] * 100, 2)
            },
            'suggestions': suggestions,
            'feature_impacts': feature_impacts
        }
    
    def _generate_suggestions(self, responses):
        return {
            'per_question': [
                f"{q}: {self.suggestion_engine[col][self.response_map[resp]]}"
                for (q, col, resp) in zip(self.questions, self.feature_names, responses)
            ],
            'general': self._get_general_advice(responses)
        }
    
    def _get_general_advice(self, responses):
        scores = [self.response_map[resp] for resp in responses]
        avg_score = sum(scores) / len(scores)
        
        if avg_score < 1.5:
            return "Maintain current stress management practices"
        elif avg_score < 2.5:
            return "Consider implementing regular stress reduction activities"
        else:
            return "Recommend professional consultation for stress management"
    
    def _analyze_feature_impacts(self, X_scaled):
        contributions = []
        for idx, feature in enumerate(self.feature_names):
            base_prob = self.model.predict_proba(X_scaled)[0].max()
            modified_X = X_scaled.copy()
            modified_X[0][idx] = 0  # Set feature to minimum
            modified_prob = self.model.predict_proba(modified_X)[0].max()
            impact = base_prob - modified_prob
            contributions.append((feature, round(impact * 100, 2)))
        return sorted(contributions, key=lambda x: x[1], reverse=True)

# Example Usage
if __name__ == "__main__":
    system = StressDetectionSystem()
    
    sample_responses = ["Sometimes", "Often", "Rarely", "Sometimes", "Never"]
    
    try:
        result = system.predict(sample_responses)
        print(f"\n🔮 Stress Level Prediction: {result['stress_level']}")
        print(f"🛡️ Confidence Level: {result['confidence']}%")
        
        print("\n📌 Question-Specific Suggestions:")
        for suggestion in result['suggestions']['per_question']:
            print(f"- {suggestion}")
            
        print(f"\n🌟 General Advice: {result['suggestions']['general']}")
        
        print("\n📊 Key Contributing Factors:")
        for feature, impact in result['feature_impacts']:
            print(f"- {feature}: {impact}% impact")
            
    except ValueError as e:
        print(f"❌ Error: {str(e)}")