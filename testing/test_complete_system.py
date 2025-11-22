"""
Test Complete 5-Agent System with Orchestrator
"""

from src.orchestrator import AgenticOrchestrator
from src.utils import DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

print("=" * 80)
print("🎮 COMPLETE AGENTIC SYSTEM TEST - 5 AGENTS + GUARDRAILS")
print("=" * 80)

# Load and prepare data
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

# Initialize orchestrator
print("\n🎯 Initializing Orchestrator...")
orchestrator = AgenticOrchestrator()

# Train pipeline
print("\n🤖 Training pipeline...")
train_result = orchestrator.train_pipeline(X_train_scaled, y_train_enc)
print(f"✅ Training Accuracy: {train_result['training_metrics']['accuracy']:.2%}")
print(f"✅ Model Agreement: {train_result['model_agreement']:.2%}")

# Process sample players
print("\n🎬 Processing 10 sample players through complete pipeline...")
print("=" * 80)

n_samples = 10
sample_indices = np.random.choice(len(X_val), n_samples, replace=False)

for i, idx in enumerate(sample_indices, 1):
    print(f"\n{'─' * 80}")
    print(f"PLAYER {i}/{n_samples}")
    print(f"{'─' * 80}")
    
    player_features_val = X_val.iloc[idx]
    player_scaled = X_val_scaled[idx]
    true_label_enc = y_val_enc[idx]
    true_label = target_encoder.inverse_transform([true_label_enc])[0]
    
    player_data = {
        'age': int(player_features_val['Age']),
        'playtime_hours': float(player_features_val['PlayTimeHours']),
        'sessions_per_week': int(player_features_val['SessionsPerWeek']),
        'player_level': int(player_features_val['PlayerLevel']),
        'has_purchases': bool(player_features_val['InGamePurchases'] == 1)
    }
    
    # Run through pipeline
    result = orchestrator.process_player(player_scaled, player_data, execute_action=True)
    
    # Display results
    print(f"True Label: {true_label}")
    print(f"Status: {result['status']}")
    
    if result['status'] != 'BLOCKED_INPUT':
        pred = result['prediction']
        print(f"Prediction: {pred['label']} (Confidence: {pred['confidence']:.1%}, Agreement: {pred['model_agreement']:.1%})")
        
        if 'recommended_action' in result:
            action = result['recommended_action']
            print(f"Action: {action['name']} (${action['cost']})")
        
        validation = result['validation']
        print(f"Validation: {'✅ APPROVED' if validation['approved'] else '❌ BLOCKED'} (Risk: {validation['overall_risk']})")
        
        if validation['all_concerns']:
            print(f"Concerns: {', '.join(validation['all_concerns'][:2])}")
        
        if result['execution']:
            exec_data = result['execution']
            outcome = exec_data['outcome']
            print(f"Execution: Success={outcome['success']}, Revenue=${outcome['revenue']:.2f}, ROI={outcome['roi']:.1f}%")

# System health check
print("\n" + "=" * 80)
print("🏥 SYSTEM HEALTH CHECK")
print("=" * 80)

health = orchestrator.monitor_system_health(X_val_scaled, y_val_enc, X_train_scaled)

print(f"\n📊 Data Drift:")
print(f"  Drift Detected: {health['drift']['drift_detected']}")
print(f"  Drifted Features: {health['drift']['n_drifted_features']}")

print(f"\n📈 Performance:")
print(f"  Accuracy: {health['performance']['performance']['accuracy']:.2%}")
print(f"  Recommend Retrain: {health['performance']['recommend_retrain']}")

print(f"\n🚨 Alerts: {len(health['alerts'])}")
for alert in health['alerts'][:3]:
    print(f"  - [{alert['type']}] {alert['message']}")

# Pipeline statistics
print("\n" + "=" * 80)
print("📊 PIPELINE STATISTICS")
print("=" * 80)

stats = orchestrator.get_pipeline_stats()

print(f"\nTotal Runs: {stats['total_runs']}")
print(f"\nStatus Distribution:")
for status, count in stats['status_distribution'].items():
    print(f"  {status}: {count}")

print(f"\nExecution Stats:")
exec_stats = stats['execution_stats']
print(f"  Total Cost: ${exec_stats['total_cost']:.2f}")
print(f"  Total Revenue: ${exec_stats['total_revenue']:.2f}")
print(f"  Net Benefit: ${exec_stats['net_benefit']:.2f}")
print(f"  ROI: {exec_stats['roi']:.1f}%")

print(f"\nGuardrail Stats:")
guard_stats = stats['guardrail_stats']
print(f"  Approval Rate: {guard_stats['approval_rate']:.1%}")
print(f"  Risk Distribution: {guard_stats['risk_distribution']}")

print("\n" + "=" * 80)
print("✅ COMPLETE SYSTEM TEST FINISHED!")
print("=" * 80)
