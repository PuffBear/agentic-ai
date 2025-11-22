"""
Test Prediction Agent
"""

from src.agents import DataAgent, PredictionAgent
from src.utils import DataLoader, MetricsCalculator
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

print("=" * 60)
print("TESTING PREDICTION AGENT")
print("=" * 60)

# 1. Load and prepare data
print("\n1️⃣  Loading data...")
loader = DataLoader()
df = loader.load_data()
print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Split data
print("\n2️⃣  Splitting data...")
X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(
    test_size=0.2,
    validation_size=0.1
)
print(f"✅ Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# 3. Encode TARGET variable (this was missing!)
print("\n3️⃣  Encoding target variable...")
target_encoder = LabelEncoder()
y_train_encoded = target_encoder.fit_transform(y_train)
y_val_encoded = target_encoder.transform(y_val)
y_test_encoded = target_encoder.transform(y_test)
print(f"✅ Target classes: {target_encoder.classes_}")

# 4. Encode categorical features
print("\n4️⃣  Encoding categorical features...")
categorical_cols = ['Gender', 'Location', 'GameGenre', 'GameDifficulty', 'InGamePurchases']
label_encoders = {}

# Make copies to avoid SettingWithCopyWarning
X_train = X_train.copy()
X_val = X_val.copy()
X_test = X_test.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_val[col] = le.transform(X_val[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

print(f"✅ Encoded {len(categorical_cols)} categorical features")

# 5. Remove PlayerID if present
if 'PlayerID' in X_train.columns:
    X_train = X_train.drop('PlayerID', axis=1)
    X_val = X_val.drop('PlayerID', axis=1)
    X_test = X_test.drop('PlayerID', axis=1)

# 6. Scale numerical features
print("\n5️⃣  Scaling features...")
scaler = StandardScaler()
all_features = X_train.columns.tolist()

X_train_scaled = scaler.fit_transform(X_train[all_features])
X_val_scaled = scaler.transform(X_val[all_features])
X_test_scaled = scaler.transform(X_test[all_features])

print(f"✅ Scaled {len(all_features)} features")

# 7. Train Prediction Agent
print("\n6️⃣  Training Prediction Agent...")
print("-" * 60)
pred_agent = PredictionAgent()

result = pred_agent.execute({
    'mode': 'train',
    'X_train': X_train_scaled,
    'y_train': y_train_encoded  # Use encoded target!
})

print("-" * 60)

if result['success']:
    print(f"\n📊 TRAINING RESULTS:")
    print(f"  ✅ Success: {result['success']}")
    print(f"  ⏱️  Time: {result['execution_time']:.2f}s")
    print(f"  🎯 Accuracy: {result['data']['metrics']['accuracy']:.4f}")
    print(f"  📈 Precision: {result['data']['metrics']['precision']:.4f}")
    print(f"  📉 Recall: {result['data']['metrics']['recall']:.4f}")
    print(f"  🎪 F1 Score: {result['data']['metrics']['f1_score']:.4f}")
    print(f"  🤝 Model Agreement: {result['data']['model_agreement']:.4f}")
    
    # 8. Test predictions on validation set
    print("\n7️⃣  Testing on validation set...")
    val_result = pred_agent.execute({
        'mode': 'predict',
        'X': X_val_scaled
    })
    
    if val_result['success']:
        # Calculate validation metrics
        metrics_calc = MetricsCalculator()
        val_metrics = metrics_calc.calculate_classification_metrics(
            y_val_encoded,  # Use encoded target!
            val_result['data']['predictions'],
            val_result['data']['probabilities']
        )
        
        print(f"\n📊 VALIDATION RESULTS:")
        print(f"  ✅ Success: {val_result['success']}")
        print(f"  🎯 Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  📈 Precision: {val_metrics['precision']:.4f}")
        print(f"  📉 Recall: {val_metrics['recall']:.4f}")
        print(f"  🎪 F1 Score: {val_metrics['f1_score']:.4f}")
        print(f"  🤝 Model Agreement: {val_result['data']['model_agreement']:.4f}")
        
        # 9. Hallucination detection stats
        n_hallucinations = val_result['data']['hallucination_mask'].sum()
        pct_hallucinations = (n_hallucinations / len(val_result['data']['hallucination_mask'])) * 100
        
        print(f"\n🛡️  HALLUCINATION DETECTION:")
        print(f"  🚨 Detected: {n_hallucinations} ({pct_hallucinations:.2f}%)")
        print(f"  ✅ Consistent: {len(val_result['data']['hallucination_mask']) - n_hallucinations}")
        
        # 10. Confidence analysis
        mean_confidence = val_result['data']['confidence'].mean()
        high_confidence_count = (val_result['data']['confidence'] > 0.7).sum()
        high_confidence_pct = (high_confidence_count / len(val_result['data']['confidence'])) * 100
        
        print(f"\n💪 CONFIDENCE METRICS:")
        print(f"  📊 Mean Confidence: {mean_confidence:.4f}")
        print(f"  🎯 High Confidence (>0.7): {high_confidence_count} ({high_confidence_pct:.2f}%)")
        print(f"  📉 Low Confidence (<0.7): {len(val_result['data']['confidence']) - high_confidence_count}")
        
        print("\n" + "=" * 60)
        print("✅ PREDICTION AGENT TEST COMPLETE!")
        print("=" * 60)
    else:
        print(f"❌ Validation failed: {val_result.get('error', 'Unknown error')}")
else:
    print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
