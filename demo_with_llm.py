"""
Complete System Demo with LLM Integration
Shows ML agents + LLM orchestration + Guardrails
"""

from src.agents import DataAgent, PredictionAgent, PrescriptiveAgent
from src.orchestrator import LLMOrchestrator
from src.guardrails.llm_guardrails import LLMGuardrail
from src.utils import DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

def main():
    print("\n" + "=" * 80)
    print("🎮 AGENTIC GAMING ANALYTICS - WITH LLM ORCHESTRATION")
    print("=" * 80)
    
    # Step 1: Prepare data (same as before)
    print("\n📊 STEP 1: Data Preparation")
    print("-" * 80)
    loader = DataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data()
    
    # Preprocessing
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
    
    print(f"✅ Data prepared: {X_train.shape[0]:,} train samples")
    
    # Step 2: Train ML agents
    print("\n🤖 STEP 2: Training ML Agents")
    print("-" * 80)
    pred_agent = PredictionAgent()
    pred_agent.execute({'mode': 'train', 'X_train': X_train_scaled, 'y_train': y_train_enc})
    
    presc_agent = PrescriptiveAgent()
    
    print("✅ Agents trained")
    
    # Step 3: Initialize LLM Orchestrator
    print("\n🧠 STEP 3: Initializing LLM Orchestrator (LangChain + Groq)")
    print("-" * 80)
    orchestrator = LLMOrchestrator({
        'prediction': pred_agent,
        'prescriptive': presc_agent
    })
    
    # Initialize guardrails
    input_guardrail = LLMGuardrail(layer="input")
    prediction_guardrail = LLMGuardrail(layer="prediction")
    action_guardrail = LLMGuardrail(layer="action")
    
    print("✅ LLM Orchestrator and Guardrails initialized")
    
    # Step 4: Demo with sample players
    print("\n🎬 STEP 4: End-to-End Demo with LLM Reasoning")
    print("=" * 80)
    
    n_samples = 3
    sample_indices = np.random.choice(len(X_val), n_samples, replace=False)
    
    for i, idx in enumerate(sample_indices, 1):
        print(f"\n{'─' * 80}")
        print(f"PLAYER {i} OF {n_samples}")
        print(f"{'─' * 80}")
        
        # Get player data
        player_features = X_val.iloc[idx]
        player_scaled = X_val_scaled[idx:idx+1]
        true_engagement = target_encoder.inverse_transform([y_val_enc[idx]])[0]
        
        player_data = {
            'age': int(player_features['Age']),
            'playtime_hours': float(player_features['PlayTimeHours']),
            'sessions_per_week': int(player_features['SessionsPerWeek']),
            'player_level': int(player_features['PlayerLevel']),
            'has_purchases': bool(player_features['InGamePurchases'] == 1)
        }
        
        print(f"\n👤 Player Profile:")
        for key, value in player_data.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
        
        # Layer 1: Input Guardrail
        print(f"\n🛡️  Layer 1: Input Validation")
        input_validation = input_guardrail.check_input_quality(player_data)
        print(f"   Status: {'✅ PASSED' if input_validation['passed'] else '❌ FAILED'}")
        if input_validation['issues']:
            print(f"   Issues: {input_validation['issues'][0]}")
        
        if not input_validation['passed']:
            print(f"   ⚠️  Skipping this player due to input validation failure")
            continue
        
        # ML Prediction
        print(f"\n�� ML Prediction (Ensemble):")
        pred_result = pred_agent.execute({'mode': 'predict', 'X': player_scaled})
        predicted_eng_encoded = pred_result['data']['predictions'][0]
        predicted_eng = target_encoder.inverse_transform([predicted_eng_encoded])[0]
        confidence = pred_result['data']['confidence'][0]
        model_agreement = pred_result['data']['model_agreement']
        
        print(f"   True Engagement: {true_engagement}")
        print(f"   Predicted: {predicted_eng}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Model Agreement: {model_agreement:.1%}")
        
        # Layer 2: Prediction Guardrail
        print(f"\n🛡️  Layer 2: Prediction Validation (Hallucination Check)")
        pred_validation = prediction_guardrail.check_prediction_quality(
            predicted_eng,
            confidence,
            model_agreement,
            player_data
        )
        print(f"   Status: {'✅ PASSED' if pred_validation['passed'] else '❌ FAILED'}")
        print(f"   Risk Level: {pred_validation['risk']}")
        if pred_validation['concerns']:
            print(f"   Concerns: {pred_validation['concerns'][0][:100]}...")
        
        # LLM Explanation
        print(f"\n🧠 LLM Explanation:")
        player_data['prediction'] = predicted_eng
        player_data['confidence'] = confidence
        player_data['model_agreement'] = model_agreement
        player_data['predicted_engagement'] = predicted_eng_encoded
        
        explanation = orchestrator.explain_prediction(
            player_data,
            predicted_eng,
            confidence,
            model_agreement
        )
        print(f"   {explanation}")
        
        # Action Recommendation
        print(f"\n🎯 Action Recommendation:")
        action_rec = orchestrator.recommend_action_with_reasoning(
            player_data,
            predicted_eng,
            presc_agent.actions
        )
        
        print(f"   Recommended: {action_rec['action']['name']} (${action_rec['action']['cost']})")
        print(f"   LLM Reasoning: {action_rec['reasoning'][:150]}...")
        
        # Layer 3: Action Guardrail
        print(f"\n🛡️  Layer 3: Action Validation")
        action_validation = action_guardrail.check_action_safety(
            action_rec['action'],
            player_data
        )
        print(f"   Status: {'✅ APPROVED' if action_validation['approved'] else '❌ REJECTED'}")
        print(f"   Risk Level: {action_validation['risk']}")
        if action_validation['concerns']:
            print(f"   Concerns: {action_validation['concerns'][0][:100]}...")
        
        # Final Decision
        print(f"\n✅ Final Decision:")
        if input_validation['passed'] and pred_validation['passed'] and action_validation['approved']:
            print(f"   ALL GUARDRAILS PASSED - Action APPROVED")
            print(f"   Execute: {action_rec['action']['name']}")
        else:
            print(f"   ⚠️  GUARDRAIL FAILURE - Action BLOCKED")
            print(f"   Reason: Validation failed at one or more layers")
    
    # Summary
    print(f"\n" + "=" * 80)
    print("✅ DEMO COMPLETE - LLM-ORCHESTRATED AGENTIC SYSTEM")
    print("=" * 80)
    print(f"\n🎯 System Components:")
    print(f"   ✅ ML Agents: Prediction (Ensemble) + Prescriptive (RL)")
    print(f"   ✅ LLM Orchestrator: LangChain + Groq (Llama 3.2 90B)")
    print(f"   ✅ 3-Layer Guardrails: Input + Prediction + Action")
    print(f"   ✅ Natural Language: Explanations + Reasoning + Validation")
    print(f"\n🚀 This is a TRUE agentic AI system!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
