"""
Task 4: Predictive Machine Learning Insights
Integrates ML to learn data usage patterns and recommend pre-emptive data movements
Uses scikit-learn for real machine learning models
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json
import math
import numpy as np
from component1_data_sorter import DataObject, StorageTier, StorageLocation

# Import scikit-learn models
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not installed. Using statistical fallback.")
    print("   Install with: pip install scikit-learn")


# ===================== DATA MODELS =====================

class AccessPattern:
    """Represents learned access pattern for a file"""

    def __init__(self, file_id: str):
        self.file_id = file_id
        self.hourly_access_profile = [0] * 24  # Access count per hour
        self.daily_access_profile = [0] * 7     # Access count per day of week
        self.trend_coefficient = 0.0            # Positive = increasing, negative = decreasing
        self.seasonality_score = 0.0            # How predictable the pattern is
        self.confidence = 0.0                   # Model confidence (0-1)

    def to_dict(self) -> Dict:
        return {
            'file_id': self.file_id,
            'hourly_profile': self.hourly_access_profile,
            'daily_profile': self.daily_access_profile,
            'trend': self.trend_coefficient,
            'seasonality': self.seasonality_score,
            'confidence': round(self.confidence, 2)
        }


class Prediction:
    """Prediction for future file access"""

    def __init__(self, file_id: str, predicted_accesses: float,
                 recommended_tier: StorageTier, confidence: float, reasoning: str):
        self.file_id = file_id
        self.predicted_accesses = predicted_accesses
        self.recommended_tier = recommended_tier
        self.confidence = confidence
        self.reasoning = reasoning
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict:
        return {
            'file_id': self.file_id,
            'predicted_accesses_next_week': round(self.predicted_accesses, 2),
            'recommended_tier': self.recommended_tier.value,
            'confidence': round(self.confidence, 2),
            'reasoning': self.reasoning,
            'prediction_timestamp': self.timestamp.isoformat()
        }


# ===================== PATTERN LEARNER =====================

class PatternLearner:
    """
    Learns access patterns from historical data
    Uses time-series analysis and pattern recognition
    """

    def __init__(self):
        self.patterns = {}  # file_id -> AccessPattern

    def learn_from_history(self, data_obj: DataObject) -> AccessPattern:
        """Learn access pattern from file's access history"""

        pattern = AccessPattern(data_obj.file_id)

        if len(data_obj.access_history) < 3:
            # Not enough data
            pattern.confidence = 0.1
            return pattern

        # Analyze hourly patterns
        for access_time in data_obj.access_history:
            hour = access_time.hour
            pattern.hourly_access_profile[hour] += 1

        # Analyze daily patterns
        for access_time in data_obj.access_history:
            day = access_time.weekday()
            pattern.daily_access_profile[day] += 1

        # Calculate trend (linear regression simplified)
        pattern.trend_coefficient = self._calculate_trend(data_obj.access_history)

        # Calculate seasonality (how predictable/regular the pattern is)
        pattern.seasonality_score = self._calculate_seasonality(pattern.hourly_access_profile)

        # Calculate confidence based on data quantity and consistency
        pattern.confidence = self._calculate_confidence(data_obj.access_history, pattern)

        self.patterns[data_obj.file_id] = pattern
        return pattern

    def _calculate_trend(self, access_history: List[datetime]) -> float:
        """Calculate trend coefficient (simplified linear regression)"""
        if len(access_history) < 5:
            return 0.0

        # Sort by time
        sorted_accesses = sorted(access_history)

        # Split into two halves and compare
        mid = len(sorted_accesses) // 2
        first_half = sorted_accesses[:mid]
        second_half = sorted_accesses[mid:]

        # Calculate access rate for each half
        if len(first_half) > 0 and len(second_half) > 0:
            first_duration = (first_half[-1] - first_half[0]).total_seconds() / 86400  # days
            second_duration = (second_half[-1] - second_half[0]).total_seconds() / 86400

            if first_duration > 0 and second_duration > 0:
                first_rate = len(first_half) / first_duration
                second_rate = len(second_half) / second_duration

                # Positive = increasing, negative = decreasing
                return (second_rate - first_rate) / (first_rate + 0.001)

        return 0.0

    def _calculate_seasonality(self, hourly_profile: List[int]) -> float:
        """Calculate how predictable the access pattern is (0-1)"""
        if sum(hourly_profile) == 0:
            return 0.0

        # Calculate variance in hourly access
        mean = sum(hourly_profile) / len(hourly_profile)
        variance = sum((x - mean) ** 2 for x in hourly_profile) / len(hourly_profile)

        # High variance = predictable peaks (seasonal)
        # Low variance = random/uniform access
        if mean > 0:
            coefficient_of_variation = math.sqrt(variance) / mean
            # Normalize to 0-1 range
            return min(1.0, coefficient_of_variation / 2)
        return 0.0

    def _calculate_confidence(self, access_history: List[datetime],
                             pattern: AccessPattern) -> float:
        """Calculate model confidence based on data quality"""

        # More data = higher confidence
        data_confidence = min(1.0, len(access_history) / 50)

        # High seasonality = higher confidence (predictable)
        seasonality_confidence = pattern.seasonality_score

        # Combine factors
        confidence = (data_confidence * 0.6) + (seasonality_confidence * 0.4)

        return confidence


# ===================== ML-BASED PREDICTOR =====================

class MLPredictor:
    """
    Uses scikit-learn models for predictions
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.trained = False
        self.model_type = "LinearRegression"  # or "RandomForest"

    def prepare_features(self, data_obj: DataObject) -> np.ndarray:
        """Extract features from data object for ML"""
        features = []

        # Feature 1: Current access rate
        features.append(data_obj.get_access_frequency(days=30))

        # Feature 2: Access count
        features.append(len(data_obj.access_history))

        # Feature 3: Days since last access
        features.append(data_obj.days_since_last_access())

        # Feature 4: Hourly variance (pattern predictability)
        if len(data_obj.access_history) > 0:
            hours = [t.hour for t in data_obj.access_history]
            hour_variance = np.var(hours) if len(hours) > 1 else 0
            features.append(hour_variance)
        else:
            features.append(0)

        # Feature 5: Weekly variance
        if len(data_obj.access_history) > 0:
            days = [t.weekday() for t in data_obj.access_history]
            day_variance = np.var(days) if len(days) > 1 else 0
            features.append(day_variance)
        else:
            features.append(0)

        # Feature 6: Trend (recent vs old access rate)
        if len(data_obj.access_history) >= 10:
            recent_rate = data_obj.get_access_frequency(days=7)
            older_rate = data_obj.get_access_frequency(days=30)
            trend = (recent_rate - older_rate) / (older_rate + 0.001)
            features.append(trend)
        else:
            features.append(0)

        return np.array(features).reshape(1, -1)

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train ML model"""
        if not SKLEARN_AVAILABLE or len(X) < 5:
            return False

        try:
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)

            # Train Linear Regression (fast, interpretable)
            if self.model_type == "LinearRegression":
                self.model = LinearRegression()
                self.model.fit(X_scaled, y)
            # Or Random Forest (more powerful, slower)
            elif self.model_type == "RandomForest":
                self.model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=5,
                    random_state=42
                )
                self.model.fit(X_scaled, y)

            self.trained = True
            return True

        except Exception as e:
            print(f"⚠️  ML training failed: {e}")
            return False

    def predict(self, features: np.ndarray) -> float:
        """Predict using trained model"""
        if not self.trained or self.model is None:
            return None

        try:
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)[0]
            return max(0, prediction)  # Can't have negative accesses
        except Exception as e:
            print(f"⚠️  Prediction failed: {e}")
            return None


# ===================== PREDICTIVE MODEL =====================

class PredictiveModel:
    """
    Predicts future access patterns and recommends pre-emptive actions
    Now enhanced with scikit-learn ML models!
    """

    def __init__(self, use_ml: bool = True):
        self.learner = PatternLearner()
        self.predictions = {}  # file_id -> Prediction
        self.ml_predictor = MLPredictor() if use_ml and SKLEARN_AVAILABLE else None
        self.use_ml = use_ml and SKLEARN_AVAILABLE

    def train(self, data_objects: List[DataObject]):
        """Train model on historical data"""
        if self.use_ml:
            print("Training ML model (scikit-learn)...")
        else:
            print("Training statistical model...")

        # Train pattern learner (always)
        for obj in data_objects:
            pattern = self.learner.learn_from_history(obj)

        # Train ML model if available
        if self.use_ml and len(data_objects) >= 5:
            X_train = []
            y_train = []

            for obj in data_objects:
                features = self.ml_predictor.prepare_features(obj)
                X_train.append(features[0])

                # Target: future access rate (use current as proxy)
                target = obj.get_access_frequency(days=7) * 7  # Weekly accesses
                y_train.append(target)

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            success = self.ml_predictor.train(X_train, y_train)

            if success:
                print(f"✓ ML model trained on {len(data_objects)} files")
                print(f"  Model type: {self.ml_predictor.model_type}")
                print(f"  Features: 6 (access rate, count, recency, patterns, trend)")
            else:
                print(f"✓ Trained on {len(data_objects)} files (statistical fallback)")
                self.use_ml = False
        else:
            print(f"✓ Trained on {len(data_objects)} files")

    def predict_future_access(self, data_obj: DataObject, days_ahead: int = 7) -> float:
        """Predict number of accesses in next N days using ML or statistical fallback"""

        # Try ML prediction first
        if self.use_ml and self.ml_predictor.trained:
            features = self.ml_predictor.prepare_features(data_obj)
            ml_prediction = self.ml_predictor.predict(features)

            if ml_prediction is not None:
                # ML model predicts weekly, scale to requested days
                return ml_prediction * (days_ahead / 7.0)

        # Fallback to statistical method
        pattern = self.learner.patterns.get(data_obj.file_id)

        if not pattern or pattern.confidence < 0.2:
            # Fallback to simple average
            return data_obj.get_access_frequency(days=30) * days_ahead

        # Base prediction on current rate
        current_rate = data_obj.get_access_frequency(days=30)

        # Apply trend adjustment
        trend_multiplier = 1.0 + (pattern.trend_coefficient * 0.5)

        # Predict
        predicted = current_rate * days_ahead * trend_multiplier

        return max(0, predicted)

    def predict_optimal_tier(self, data_obj: DataObject) -> Tuple[StorageTier, float, str]:
        """Predict optimal tier for next week"""

        predicted_accesses = self.predict_future_access(data_obj, days_ahead=7)
        predicted_rate = predicted_accesses / 7

        pattern = self.learner.patterns.get(data_obj.file_id)
        confidence = pattern.confidence if pattern else 0.3

        # Boost confidence if using ML
        if self.use_ml and self.ml_predictor.trained:
            confidence = min(1.0, confidence * 1.2)  # 20% boost for ML

        # Decision logic based on predicted access rate
        reasoning = []

        # Indicate prediction method
        method = "ML" if (self.use_ml and self.ml_predictor.trained) else "Statistical"
        reasoning.append(f"{method} model")

        if predicted_rate > 1.5:
            tier = StorageTier.HOT
            reasoning.append(f"High predicted access: {predicted_rate:.2f}/day")
        elif predicted_rate > 0.3:
            tier = StorageTier.WARM
            reasoning.append(f"Moderate predicted access: {predicted_rate:.2f}/day")
        else:
            tier = StorageTier.COLD
            reasoning.append(f"Low predicted access: {predicted_rate:.2f}/day")

        # Consider trend
        if pattern and pattern.trend_coefficient > 0.3:
            reasoning.append("Increasing trend detected")
            # Upgrade tier proactively
            if tier == StorageTier.COLD:
                tier = StorageTier.WARM
            elif tier == StorageTier.WARM:
                tier = StorageTier.HOT
        elif pattern and pattern.trend_coefficient < -0.3:
            reasoning.append("Decreasing trend detected")
            # Can downgrade tier proactively

        # Consider latency requirement
        if data_obj.latency_requirement.value == 'critical':
            tier = StorageTier.HOT
            reasoning.append("Critical latency requirement")

        return tier, confidence, " | ".join(reasoning)

    def generate_predictions(self, data_objects: List[DataObject]) -> List[Prediction]:
        """Generate predictions for all files"""

        predictions = []

        for obj in data_objects:
            predicted_accesses = self.predict_future_access(obj, days_ahead=7)
            tier, confidence, reasoning = self.predict_optimal_tier(obj)

            prediction = Prediction(
                file_id=obj.file_id,
                predicted_accesses=predicted_accesses,
                recommended_tier=tier,
                confidence=confidence,
                reasoning=reasoning
            )

            predictions.append(prediction)
            self.predictions[obj.file_id] = prediction

        return predictions

    def recommend_preemptive_actions(self, data_objects: List[DataObject]) -> List[Dict]:
        """Recommend pre-emptive migrations before access patterns change"""

        recommendations = []

        for obj in data_objects:
            prediction = self.predictions.get(obj.file_id)

            if not prediction:
                continue

            # If predicted tier differs from current, recommend migration
            if prediction.recommended_tier != obj.tier:
                # Only recommend if confidence is reasonable
                if prediction.confidence > 0.4:
                    recommendations.append({
                        'file_id': obj.file_id,
                        'file_name': obj.name,
                        'current_tier': obj.tier.value,
                        'predicted_tier': prediction.recommended_tier.value,
                        'confidence': prediction.confidence,
                        'reasoning': prediction.reasoning,
                        'urgency': 'high' if prediction.confidence > 0.7 else 'medium',
                        'predicted_accesses_next_week': prediction.predicted_accesses
                    })

        # Sort by confidence (most confident first)
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)

        return recommendations


# ===================== MODEL EVALUATOR =====================

class ModelEvaluator:
    """Evaluates model performance"""

    def __init__(self):
        self.actual_vs_predicted = []

    def evaluate_prediction_accuracy(self, predicted: float, actual: float) -> float:
        """Calculate prediction accuracy (0-1)"""

        if actual == 0 and predicted == 0:
            return 1.0

        if actual == 0:
            return 0.0

        error = abs(predicted - actual) / actual
        accuracy = max(0.0, 1.0 - error)

        return accuracy

    def calculate_metrics(self, predictions: List[Prediction],
                         actual_data: List[DataObject]) -> Dict:
        """Calculate model performance metrics"""

        accuracies = []

        for pred in predictions:
            actual_obj = next((obj for obj in actual_data if obj.file_id == pred.file_id), None)
            if actual_obj:
                actual_rate = actual_obj.get_access_frequency(days=7)
                predicted_rate = pred.predicted_accesses / 7
                accuracy = self.evaluate_prediction_accuracy(predicted_rate, actual_rate)
                accuracies.append(accuracy)

        if not accuracies:
            return {'error': 'No data to evaluate'}

        return {
            'mean_accuracy': sum(accuracies) / len(accuracies),
            'predictions_made': len(predictions),
            'high_confidence_predictions': len([p for p in predictions if p.confidence > 0.7]),
            'medium_confidence_predictions': len([p for p in predictions if 0.4 <= p.confidence <= 0.7]),
            'low_confidence_predictions': len([p for p in predictions if p.confidence < 0.4])
        }


# ===================== PREDICTIVE MANAGER =====================

class PredictiveManager:
    """
    Main manager for predictive ML system
    Integrates with previous tasks
    """

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.model = PredictiveModel()
        self.evaluator = ModelEvaluator()

    def train_model(self):
        """Train ML model on current data"""
        self.model.train(self.data_manager.data_objects)

    def get_predictions(self) -> List[Prediction]:
        """Get predictions for all files"""
        return self.model.generate_predictions(self.data_manager.data_objects)

    def get_preemptive_recommendations(self) -> List[Dict]:
        """Get pre-emptive migration recommendations"""
        return self.model.recommend_preemptive_actions(self.data_manager.data_objects)

    def compare_with_rule_based(self) -> Dict:
        """Compare ML predictions with simple rule-based approach"""

        ml_recommendations = self.get_preemptive_recommendations()

        # Simple rule-based approach (from Task 1)
        from component1_data_sorter import DataClassifier
        classifier = DataClassifier()

        rule_based_changes = []
        for obj in self.data_manager.data_objects:
            rule_tier = classifier.classify(obj)
            if rule_tier != obj.tier:
                rule_based_changes.append(obj.file_id)

        return {
            'ml_recommendations': len(ml_recommendations),
            'rule_based_recommendations': len(rule_based_changes),
            'ml_high_confidence': len([r for r in ml_recommendations if r['confidence'] > 0.7]),
            'advantage': 'ML can predict future changes before they happen'
        }

    def export_predictions(self, filename: str = "ml_predictions.json"):
        """Export predictions for dashboard"""

        predictions = self.get_predictions()
        recommendations = self.get_preemptive_recommendations()

        data = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'trained_files': len(self.data_manager.data_objects),
                'patterns_learned': len(self.model.learner.patterns)
            },
            'predictions': [p.to_dict() for p in predictions],
            'preemptive_recommendations': recommendations,
            'patterns': {
                file_id: pattern.to_dict()
                for file_id, pattern in self.model.learner.patterns.items()
            }
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"ML predictions exported to {filename}")


if __name__ == "__main__":
    print("Task 4: Predictive ML Insights - Ready!")
    print("Import this module to use ML features")
