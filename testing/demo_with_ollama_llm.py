"""
Complete System Demo with Ollama + LangChain
Uses local Llama3.2 model - no API keys needed!
"""

from src.agents import PredictionAgent, PrescriptiveAgent
from src.utils import DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from langchain_ollama import ChatOllama
# Fix: Use langchain_core instead of langchain.schema
from langchain_core.messages import HumanMessage, SystemMessage

class OllamaLLM:
    """LLM using Ollama + LangChain (local, free)"""
    
    def __init__(self, model="llama3.2"):
        """
        Initialize Ollama LLM
        
        Args:
            model: Ollama model name (llama3.2, llama2, etc.)
        """
        try:
            self.llm = ChatOllama(
                model=model,
                temperature=0.3,
            )
            # Test it works
            test = self.llm.invoke([HumanMessage(content="Say 'OK' if you're working")])
            print(f"✅ Ollama LLM initialized ({model})")
            print(f"   Test response: {test.content}")
        except Exception as e:
            print(f"❌ Ollama initialization failed: {e}")
            print("\n💡 Make sure Ollama is running:")
            print("   1. Run: ollama serve")
            print("   2. Run: ollama pull llama3.2")
            raise
    
    def ask(self, prompt: str, system_message: str = "You are a helpful AI assistant.") -> str:
        """Ask the LLM anything"""
        try:
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"LLM Error: {str(e)}"
    
    def explain_prediction(self, player_data, prediction, confidence, agreement):
        prompt = f"""Explain this gaming analytics prediction in 2-3 sentences:

Player Profile:
- Age: {player_data['age']}
- Playtime: {player_data['playtime_hours']:.1f} hours
- Sessions/Week: {player_data['sessions_per_week']}
- Level: {player_data['player_level']}
- Has Purchases: {player_data['has_purchases']}

Prediction Results:
- Predicted Engagement: {prediction}
- Confidence: {confidence:.1%}
- Model Agreement: {agreement:.1%}

Why does this prediction make sense? What are the risks?"""

        return self.ask(prompt, "You are an expert gaming analytics AI. Be concise and insightful.")
    
    def recommend_action_with_reasoning(self, player_data, prediction, actions):
        actions_text = "\n".join([
            f"{i}. {a['name']} (${a['cost']}) - {a['description']}" 
            for i, a in enumerate(actions)
        ])
        
        prompt = f"""Recommend the BEST action for this player:

Player Context:
- Predicted Engagement: {prediction}
- Age: {player_data['age']}
- Playtime: {player_data['playtime_hours']:.1f} hours
- Level: {player_data['player_level']}
- Has Purchases: {player_data['has_purchases']}

Available Actions:
{actions_text}

Which action is most effective? Format EXACTLY as:
ACTION: [action name]
REASONING: [1-2 sentences explaining why]"""

        response = self.ask(prompt, "You are an expert in player retention strategies.")
        
        # Parse response
        try:
            lines = response.split('\n')
            action_line = [l for l in lines if 'ACTION:' in l.upper()][0]
            reasoning_line = [l for l in lines if 'REASONING:' in l.upper()][0]
            
            action_name = action_line.split(':', 1)[1].strip().lower()
            reasoning = reasoning_line.split(':', 1)[1].strip()
            
            # Find matching action
            recommended = next((a for a in actions if action_name in a['name'].lower()), actions[2])
            
            return {
                'action': recommended,
                'reasoning': reasoning,
                'llm_override': True
            }
        except Exception as e:
            print(f"   (Parse error, using default: {e})")
            return {
                'action': actions[2],
                'reasoning': response[:200],
                'llm_override': False
            }
    
    def validate_decision(self, context):
        prompt = f"""Validate this AI decision for safety:

Decision Context:
- Prediction: {context['prediction']} engagement
- Confidence: {context['confidence']:.1%}
- Model Agreement: {context['model_agreement']:.1%}
- Proposed Action: {context['action']}
- Cost: ${context['action_cost']}

Is this decision safe to execute? Format EXACTLY as:
APPROVED: YES or NO
RISK: LOW, MEDIUM, or HIGH
REASON: [why in 1 sentence]"""

        response = self.ask(prompt, "You are an AI safety validator. Be strict and cautious.")
        
        # Parse response
        try:
            approved = 'YES' in response.split('APPROVED:')[1].split('\n')[0].upper()
            risk_line = response.split('RISK:')[1].split('\n')[0].strip()
            risk = risk_line.split()[0].upper()  # Take first word
            reason = response.split('REASON:')[1].strip().split('\n')[0]
            
            return {
                'validated': approved,
                'risk_level': risk if risk in ['LOW', 'MEDIUM', 'HIGH'] else 'MEDIUM',
                'concerns': [] if approved else ['LLM flagged as risky'],
                'explanation': reason
            }
        except Exception as e:
            print(f"   (Parse error: {e})")
            return {
                'validated': False,
                'risk_level': 'HIGH',
                'concerns': ['Failed to parse LLM response'],
                'explanation': response[:200]
            }

def main():
    print("\n" + "=" * 80)
    print("🎮 AGENTIC GAMING ANALYTICS - WITH OLLAMA + LANGCHAIN")
    print("=" * 80)
    
    # Initialize Ollama LLM
    print("\n🧠 Initializing Ollama LLM (Local Llama3.2)...")
    print("-" * 80)
    try:
        llm = OllamaLLM(model="llama3.2")  # or "llama2"
    except Exception:
        print("\n❌ Failed to initialize Ollama")
        return
    
    # Test LLM with a simple question
    print("\n💬 Testing LLM...")
    test_response = llm.ask("In one sentence, what makes a good gaming analytics system?")
    print(f"   LLM: {test_response[:150]}...")
    
    # Prepare data
    print("\n📊 STEP 1: Data Preparation")
    print("-" * 80)
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
    
    print(f"✅ Data prepared")
    
    # Train ML agents
    print("\n🤖 STEP 2: Training ML Agents")
    print("-" * 80)
    pred_agent = PredictionAgent()
    train_result = pred_agent.execute({'mode': 'train', 'X_train': X_train_scaled, 'y_train': y_train_enc})
    presc_agent = PrescriptiveAgent()
    print(f"✅ Agents trained ({train_result['data']['metrics']['accuracy']:.2%} accuracy)")
    
    # Demo with sample players
    print("\n🎬 STEP 3: End-to-End Demo with Ollama LLM")
    print("=" * 80)
    
    n_samples = 2  # Just 2 for demo
    sample_indices = np.random.choice(len(X_val), n_samples, replace=False)
    
    for i, idx in enumerate(sample_indices, 1):
        print(f"\n{'─' * 80}")
        print(f"PLAYER {i} OF {n_samples}")
        print(f"{'─' * 80}")
        
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
        print(f"\n�� ML Prediction (Ensemble):")
        pred_result = pred_agent.execute({'mode': 'predict', 'X': player_scaled})
        predicted_eng_encoded = pred_result['data']['predictions'][0]
        predicted_eng = target_encoder.inverse_transform([predicted_eng_encoded])[0]
        confidence = pred_result['data']['confidence'][0]
        model_agreement = pred_result['data']['model_agreement']
        
        correct = "✅" if predicted_eng == true_engagement else "❌"
        print(f"   True: {true_engagement} | Predicted: {predicted_eng} {correct}")
        print(f"   Confidence: {confidence:.1%} | Agreement: {model_agreement:.1%}")
        
        # LLM Explanation (Ollama + LangChain)
        print(f"\n🧠 LLM Explanation (Ollama - Llama3.2):")
        explanation = llm.explain_prediction(player_data, predicted_eng, confidence, model_agreement)
        print(f"   {explanation}")
        
        # LLM Action Recommendation
        print(f"\n🎯 LLM Action Recommendation:")
        action_rec = llm.recommend_action_with_reasoning(player_data, predicted_eng, presc_agent.actions)
        print(f"   Recommended: {action_rec['action']['name']} (${action_rec['action']['cost']})")
        print(f"   Reasoning: {action_rec['reasoning']}")
        
        # LLM Validation (Guardrail)
        print(f"\n🛡️  LLM Validation (Guardrail):")
        validation = llm.validate_decision({
            'prediction': predicted_eng,
            'confidence': confidence,
            'model_agreement': model_agreement,
            'action': action_rec['action']['name'],
            'action_cost': action_rec['action']['cost']
        })
        print(f"   Status: {'✅ APPROVED' if validation['validated'] else '❌ REJECTED'}")
        print(f"   Risk: {validation['risk_level']}")
        print(f"   Reason: {validation['explanation']}")
    
    # Interactive mode
    print(f"\n{'=' * 80}")
    print("💬 INTERACTIVE MODE - Ask Ollama LLM anything!")
    print("=" * 80)
    print("Type your questions (or 'quit' to exit):\n")
    
    while True:
        try:
            question = input("🤔 You: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if not question:
                continue
            
            print("🧠 LLM: ", end="", flush=True)
            response = llm.ask(question, "You are an expert in gaming analytics and AI systems.")
            print(response)
            print()
        except KeyboardInterrupt:
            break
    
    print("\n✅ Demo complete!")

if __name__ == "__main__":
    main()
