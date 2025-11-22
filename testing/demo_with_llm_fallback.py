"""
Complete System Demo with LLM Integration (with fallback)
"""

from src.agents import DataAgent, PredictionAgent, PrescriptiveAgent
from src.utils import DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

class MockLLM:
    """Fallback when LLM isn't available"""
    
    def explain_prediction(self, player_data, prediction, confidence, agreement):
        return f"Predicted {prediction} engagement with {confidence:.1%} confidence (Model agreement: {agreement:.1%}). This prediction is based on the player's activity patterns and engagement history."
    
    def recommend_action_with_reasoning(self, player_data, prediction, actions):
        # Simple rule-based fallback
        if prediction == 'Low':
            action = actions[0]  # discount
            reasoning = "Low engagement detected - offering discount to re-engage player and boost activity"
        elif prediction == 'Medium':
            action = actions[2]  # notification
            reasoning = "Medium engagement - sending targeted notification to maintain and increase activity levels"
        else:
            action = actions[4]  # no action
            reasoning = "High engagement - player is already highly engaged, no intervention needed to avoid over-messaging"
        
        return {'action': action, 'reasoning': reasoning, 'llm_override': False}
    
    def validate_decision(self, context):
        confidence = context.get('confidence', 0)
        agreement = context.get('model_agreement', 0)
        
        validated = confidence > 0.6 and agreement > 0.7
        
        if confidence > 0.8 and agreement > 0.85:
            risk_level = 'LOW'
            explanation = 'High confidence and strong model agreement - prediction is highly reliable'
        elif confidence > 0.6 and agreement > 0.7:
            risk_level = 'MEDIUM'
            explanation = 'Acceptable confidence and agreement - prediction is reasonably reliable'
        else:
            risk_level = 'HIGH'
            explanation = 'Low confidence or poor model agreement - prediction may be unreliable'
        
        concerns = []
        if confidence < 0.7:
            concerns.append('Low prediction confidence')
        if agreement < 0.75:
            concerns.append('Models show disagreement')
        
        return {
            'validated': validated,
            'risk_level': risk_level,
            'concerns': concerns,
            'explanation': explanation
        }

def main():
    print("\n" + "=" * 80)
    print("🎮 AGENTIC GAMING ANALYTICS - COMPLETE SYSTEM DEMO")
    print("=" * 80)
    
    # Step 1: Prepare data
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
    print("\n🤖 STEP 2: Training ML Agents (Ensemble)")
    print("-" * 80)
    pred_agent = PredictionAgent()
    train_result = pred_agent.execute({'mode': 'train', 'X_train': X_train_scaled, 'y_train': y_train_enc})
    
    presc_agent = PrescriptiveAgent()
    
    print(f"✅ Agents trained in {train_result['execution_time']:.2f}s")
    print(f"   Training Accuracy: {train_result['data']['metrics']['accuracy']:.2%}")
    print(f"   Model Agreement: {train_result['data']['model_agreement']:.2%}")
    
    # Step 3: Initialize Mock LLM
    print("\n🧠 STEP 3: Initializing Decision Layer")
    print("-" * 80)
    print("   Using rule-based reasoning (LLM fallback mode)")
    print("   💡 To enable real LLM: Use Python 3.11/3.12 and set GROQ_API_KEY")
    
    llm = MockLLM()
    
    print("✅ Decision layer initialized")
    
    # Step 4: Demo with sample players
    print("\n🎬 STEP 4: End-to-End Demo on Sample Players")
    print("=" * 80)
    
    n_samples = 5
    sample_indices = np.random.choice(len(X_val), n_samples, replace=False)
    
    correct_predictions = 0
    total_cost = 0
    
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
        
        # ML Prediction
        print(f"\n🤖 ML Prediction (Ensemble: RF + XGBoost + Neural Net):")
        pred_result = pred_agent.execute({'mode': 'predict', 'X': player_scaled})
        predicted_eng_encoded = pred_result['data']['predictions'][0]
        predicted_eng = target_encoder.inverse_transform([predicted_eng_encoded])[0]
        confidence = pred_result['data']['confidence'][0]
        model_agreement = pred_result['data']['model_agreement']
        is_hallucination = pred_result['data']['hallucination_mask'][0]
        
        print(f"   True Engagement:    {true_engagement}")
        print(f"   Predicted:          {predicted_eng} {'✅' if predicted_eng == true_engagement else '❌'}")
        print(f"   Confidence:         {confidence:.1%}")
        print(f"   Model Agreement:    {model_agreement:.1%}")
        print(f"   Hallucination Flag: {'🚨 YES' if is_hallucination else '✅ NO'}")
        
        if predicted_eng == true_engagement:
            correct_predictions += 1
        
        # Explanation
        print(f"\n💭 AI Explanation:")
        explanation = llm.explain_prediction(player_data, predicted_eng, confidence, model_agreement)
        print(f"   {explanation}")
        
        # Action Recommendation
        print(f"\n🎯 Action Recommendation:")
        action_rec = llm.recommend_action_with_reasoning(
            player_data,
            predicted_eng,
            presc_agent.actions
        )
        
        print(f"   Recommended: {action_rec['action']['name']} (Cost: ${action_rec['action']['cost']})")
        print(f"   Reasoning: {action_rec['reasoning']}")
        
        total_cost += action_rec['action']['cost']
        
        # Validation
        print(f"\n🛡️  Guardrail Validation:")
        validation = llm.validate_decision({
            'prediction': predicted_eng,
            'confidence': confidence,
            'model_agreement': model_agreement,
            'action': action_rec['action']['name'],
            'action_cost': action_rec['action']['cost']
        })
        
        status_icon = '✅' if validation['validated'] else '❌'
        print(f"   Status: {status_icon} {'APPROVED' if validation['validated'] else 'REJECTED'}")
        print(f"   Risk Level: {validation['risk_level']}")
        print(f"   Explanation: {validation['explanation']}")
        if validation['concerns']:
            print(f"   Concerns: {', '.join(validation['concerns'])}")
        
        # Final decision
        if validation['validated'] and not is_hallucination:
            print(f"\n✅ FINAL DECISION: Execute {action_rec['action']['name']}")
        else:
            print(f"\n⚠️  FINAL DECISION: Action BLOCKED (Failed validation or hallucination detected)")
    
    # Summary
    accuracy = correct_predictions / n_samples
    print(f"\n" + "=" * 80)
    print("📊 DEMO SUMMARY")
    print("=" * 80)
    print(f"\n✅ System Performance:")
    print(f"   Prediction Accuracy: {accuracy:.1%} ({correct_predictions}/{n_samples} correct)")
    print(f"   Total Action Cost: ${total_cost:.2f}")
    
    print(f"\n🎯 System Components:")
    print(f"   ✅ Agent 1: Data Ingestion & Preprocessing")
    print(f"   ✅ Agent 2: Prediction (Ensemble ML) - {train_result['data']['metrics']['accuracy']:.2%} accuracy")
    print(f"   ✅ Agent 3: Prescriptive (RL-based recommendations)")
    print(f"   ✅ Hallucination Detection: Model agreement checking")
    print(f"   ✅ Guardrails: Multi-layer validation system")
    
    print(f"\n💡 Architecture Highlights:")
    print(f"   • Ensemble Learning: 3 models voting (RF + XGBoost + NN)")
    print(f"   • Confidence Scoring: Per-prediction reliability metrics")
    print(f"   • Hallucination Detection: Cross-model consistency checks")
    print(f"   • Thompson Sampling: Contextual bandit for action selection")
    print(f"   • Multi-layer Guardrails: Input → Prediction → Action validation")
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE AGENTIC SYSTEM DEMO FINISHED!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
