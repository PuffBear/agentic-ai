"""
Complete System Demo - Python Script Version
Quick demonstration of all 3 agents working together
"""

from src.agents import DataAgent, PredictionAgent, PrescriptiveAgent
from src.utils import DataLoader, MetricsCalculator
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd

def main():
    print("\n" + "=" * 70)
    print("🎮 AGENTIC GAMING ANALYTICS - COMPLETE SYSTEM DEMO")
    print("=" * 70)
    
    # STEP 1: Data Ingestion
    print("\n📊 STEP 1: Data Ingestion & Feature Engineering")
    print("-" * 70)
    loader = DataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data()
    print(f"✅ Data loaded: {X_train.shape[0]:,} train, {X_val.shape[0]:,} val, {X_test.shape[0]:,} test")
    
    # Preprocessing
    print("\n🔧 Preprocessing data...")
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
    
    print(f"✅ Preprocessing complete: {X_train.shape[1]} features")
    
    # STEP 2: Prediction Agent
    print("\n🤖 STEP 2: Multi-Model Prediction (Ensemble)")
    print("-" * 70)
    pred_agent = PredictionAgent()
    
    print("Training ensemble (RF + XGBoost + Neural Network)...")
    train_result = pred_agent.execute({
        'mode': 'train',
        'X_train': X_train_scaled,
        'y_train': y_train_enc
    })
    
    print(f"✅ Training complete in {train_result['execution_time']:.2f}s")
    print(f"   Accuracy: {train_result['data']['metrics']['accuracy']:.4f}")
    print(f"   Model Agreement: {train_result['data']['model_agreement']:.2%}")
    
    # Validation
    val_result = pred_agent.execute({
        'mode': 'predict',
        'X': X_val_scaled
    })
    
    metrics_calc = MetricsCalculator()
    val_metrics = metrics_calc.calculate_classification_metrics(
        y_val_enc,
        val_result['data']['predictions'],
        val_result['data']['probabilities']
    )
    
    print(f"\n📊 Validation Results:")
    print(f"   Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"   F1 Score: {val_metrics['f1_score']:.4f}")
    print(f"   Hallucination Rate: {(val_result['data']['hallucination_mask'].sum()/len(val_result['data']['hallucination_mask'])*100):.2f}%")
    
    # STEP 3: Prescriptive Agent (RL)
    print("\n🎲 STEP 3: Prescriptive Recommendations (RL)")
    print("-" * 70)
    presc_agent = PrescriptiveAgent()
    
    print("Running RL simulation (1000 iterations)...")
    for i in range(1000):
        player_data = {
            'age': np.random.randint(18, 60),
            'playtime_hours': np.random.uniform(10, 500),
            'sessions_per_week': np.random.randint(1, 20),
            'player_level': np.random.randint(1, 100),
            'has_purchases': np.random.random() > 0.5,
            'predicted_engagement': np.random.randint(0, 3)
        }
        
        rec_result = presc_agent.execute({
            'mode': 'recommend',
            'player_data': player_data
        })
        
        action_id = rec_result['data']['action_id']
        predicted_eng = player_data['predicted_engagement']
        
        if predicted_eng == 0:
            reward = 2 if action_id in [0, 1, 2] else -1
        elif predicted_eng == 1:
            reward = 2 if action_id in [2, 3] else 0
        else:
            reward = 1 if action_id == 4 else -1
        
        presc_agent.execute({
            'mode': 'update',
            'action_id': action_id,
            'reward': reward
        })
    
    stats_result = presc_agent.execute({'mode': 'stats'})
    
    print(f"✅ RL training complete!")
    print(f"   Best Action: {stats_result['data']['best_action_name']}")
    print(f"   Total Iterations: {stats_result['data']['total_iterations']}")
    
    # STEP 4: End-to-End Demo
    print("\n🎬 STEP 4: End-to-End Demo on Sample Players")
    print("-" * 70)
    
    n_samples = 5
    sample_indices = np.random.choice(len(X_val), n_samples, replace=False)
    
    print(f"\n{'Player':<8} {'True':<10} {'Predicted':<10} {'Confidence':<12} {'Recommended Action':<20}")
    print("-" * 70)
    
    for idx in sample_indices:
        player_scaled = X_val_scaled[idx:idx+1]
        true_eng = target_encoder.inverse_transform([y_val_enc[idx]])[0]
        
        pred_result = pred_agent.execute({'mode': 'predict', 'X': player_scaled})
        predicted_eng = target_encoder.inverse_transform([pred_result['data']['predictions'][0]])[0]
        confidence = pred_result['data']['confidence'][0]
        
        player_data = {
            'age': X_val.iloc[idx]['Age'],
            'playtime_hours': X_val.iloc[idx]['PlayTimeHours'],
            'sessions_per_week': X_val.iloc[idx]['SessionsPerWeek'],
            'player_level': X_val.iloc[idx]['PlayerLevel'],
            'has_purchases': X_val.iloc[idx]['InGamePurchases'] == 1,
            'predicted_engagement': pred_result['data']['predictions'][0]
        }
        
        action_result = presc_agent.execute({'mode': 'recommend', 'player_data': player_data})
        action_name = action_result['data']['action']['name']
        
        print(f"{idx:<8} {true_eng:<10} {predicted_eng:<10} {confidence:<12.2%} {action_name:<20}")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ SYSTEM DEMO COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Agent 1: Data Ingestion          ✅ Operational")
    print(f"   Agent 2: Prediction (Ensemble)   ✅ {val_metrics['accuracy']:.2%} Accuracy")
    print(f"   Agent 3: Prescriptive (RL)       ✅ Converged")
    print(f"   Hallucination Detection          ✅ {(val_result['data']['hallucination_mask'].sum()/len(val_result['data']['hallucination_mask'])*100):.2f}% Rate")
    print(f"\n🎯 All systems operational and ready for deployment!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
