"""
Test Agents 4 & 5
"""

from src.agents import ExecutionAgent, MonitoringAgent, PredictionAgent
from src.utils import DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

print("=" * 70)
print("TESTING AGENT 4: EXECUTION & AGENT 5: MONITORING")
print("=" * 70)

# Prepare data
print("\n📊 Loading data...")
loader = DataLoader()
X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data()

target_encoder = LabelEncoder()
y_train_enc = target_encoder.fit_transform(y_train)
y_val_enc = target_encoder.transform(y_val)

categorical_cols = ['Gender', 'Location', 'GameGenre', 'GameDifficulty', 'InGamePurchases']
X_train = X_train.copy()
X_val = X_val.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_val[col] = le.transform(X_val[col])

if 'PlayerID' in X_train.columns:
    X_train = X_train.drop('PlayerID', axis=1)
    X_val = X_val.drop('PlayerID', axis=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("✅ Data ready")

# Train model
print("\n🤖 Training prediction model...")
pred_agent = PredictionAgent()
pred_agent.execute({'mode': 'train', 'X_train': X_train_scaled, 'y_train': y_train_enc})
print("✅ Model trained")

# Test Agent 4: Execution
print("\n" + "=" * 70)
print("AGENT 4: EXECUTION AGENT")
print("=" * 70)

exec_agent = ExecutionAgent()

# Simulate 10 actions
actions = [
    {'name': 'discount_10', 'cost': 5, 'description': '10% discount'},
    {'name': 'notification', 'cost': 0.5, 'description': 'Notification'},
    {'name': 'no_action', 'cost': 0, 'description': 'No action'}
]

print("\n🎯 Simulating 10 action executions...")
for i in range(10):
    action = actions[i % 3]
    prediction = ['Low', 'Medium', 'High'][i % 3]
    
    result = exec_agent.execute({
        'mode': 'simulate',
        'action': action,
        'player_data': {'age': 25, 'level': 10},
        'prediction': prediction,
        'confidence': 0.8
    })
    
    outcome = result['data']['outcome']
    print(f"  {i+1}. {action['name']:20s} | Success: {outcome['success']} | "
          f"Revenue: ${outcome['revenue']:5.2f} | ROI: {outcome['roi']:6.1f}%")

# Get ROI summary
print("\n📊 ROI SUMMARY:")
roi_result = exec_agent.execute({'mode': 'get_roi'})
roi_stats = roi_result['data']['roi_stats']

print(f"  Total Cost: ${roi_stats['total_cost']:.2f}")
print(f"  Total Revenue: ${roi_stats['total_revenue']:.2f}")
print(f"  Net Benefit: ${roi_stats['net_benefit']:.2f}")
print(f"  Overall ROI: {roi_stats['overall_roi']:.1f}%")
print(f"  Success Rate: {roi_stats['success_rate']:.1%}")

# Test Agent 5: Monitoring
print("\n" + "=" * 70)
print("AGENT 5: MONITORING AGENT")
print("=" * 70)

monitor_agent = MonitoringAgent()

# Set baseline
print("\n📊 Setting baseline from training data...")
monitor_agent.execute({
    'mode': 'set_baseline',
    'X': X_train_scaled,
    'y': y_train_enc
})
print("✅ Baseline set")

# Check drift
print("\n🔍 Checking for data drift...")
drift_result = monitor_agent.execute({
    'mode': 'check_drift',
    'X': X_val_scaled,
    'X_baseline': X_train_scaled
})

drift_data = drift_result['data']
print(f"  Drift Detected: {drift_data['drift_detected']}")
print(f"  Drifted Features: {drift_data['n_drifted_features']}/{X_val_scaled.shape[1]}")
print(f"  Max Drift Score: {drift_data['max_drift_score']:.4f}")

# Monitor performance
print("\n📈 Monitoring model performance...")
pred_result = pred_agent.execute({'mode': 'predict', 'X': X_val_scaled})
y_pred = pred_result['data']['predictions']

perf_result = monitor_agent.execute({
    'mode': 'monitor_performance',
    'y_true': y_val_enc,
    'y_pred': y_pred,
    'confidence': pred_result['data']['confidence']
})

perf_data = perf_result['data']['performance']
print(f"  Accuracy: {perf_data['accuracy']:.2%}")
print(f"  Avg Confidence: {perf_data['avg_confidence']:.2%}")
print(f"  Recommend Retrain: {perf_result['data']['recommend_retrain']}")

# Get alerts
print("\n🚨 System Alerts:")
alerts_result = monitor_agent.execute({'mode': 'get_alerts'})
alerts = alerts_result['data']['alerts']

if alerts:
    for alert in alerts:
        print(f"  [{alert['type']}] {alert['message']}")
else:
    print("  No alerts - system healthy ✅")

print("\n" + "=" * 70)
print("✅ AGENTS 4 & 5 COMPLETE!")
print("=" * 70)
