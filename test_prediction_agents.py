from src.agents import DataAgent, PredictionAgent
from src.utils import DataLoader, MetricsCalculator
from sklearn.preprocessing import LabelEncoder

# 1. Load and prepare data
print("Loading data...")
loader = DataLoader()
X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data()

# 2. Preprocess features
print("Preprocessing...")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Select only numeric columns for now (we'll add encoding later)
numeric_cols = ['Age', 'PlayTimeHours', 'SessionsPerWeek', 
                'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked']
X_train_scaled = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled = scaler.transform(X_test[numeric_cols])

# 3. Train Prediction Agent
print("Training Prediction Agent...")
pred_agent = PredictionAgent()
result = pred_agent.execute({
    'mode': 'train',
    'X_train': X_train_scaled,
    'y_train': y_train
})

print(f"\n✅ Training Success: {result['success']}")
print(f"⏱️  Training Time: {result['execution_time']:.2f}s")
print(f"📊 Accuracy: {result['data']['metrics']['accuracy']:.3f}")
print(f"🤝 Model Agreement: {result['data']['model_agreement']:.3f}")

# 4. Test predictions
print("\nTesting predictions...")
test_result = pred_agent.execute({
    'mode': 'predict',
    'X': X_test_scaled
})

print(f"✅ Prediction Success: {test_result['success']}")
print(f"📊 Predictions made: {len(test_result['data']['predictions'])}")
print(f"🤝 Model Agreement: {test_result['data']['model_agreement']:.3f}")