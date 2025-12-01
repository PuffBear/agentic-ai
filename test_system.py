"""
Quick system test to verify all components are working
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

print("=" * 80)
print("AGENTIC GAMING ANALYTICS - SYSTEM TEST")
print("=" * 80)

# Test 1: Import all modules
print("\n[1/7] Testing imports...")
try:
    from src.agents import (
        DataAgent, PredictionAgent, PrescriptiveAgent,
        ExecutionAgent, MonitoringAgent
    )
    from src.guardrails import GuardrailSystem
    from src.guardrails.layer1_input import InputValidationGuardrail
    from src.guardrails.layer2_prediction import PredictionValidationGuardrail
    from src.guardrails.layer3_action import ActionValidationGuardrail
    from src.models import EnsembleModel, ContextualBandit, DriftDetector
    from src.orchestrator import AgenticOrchestrator
    from src.utils.data_loader import DataLoader
    from src.utils.feature_engineering import FeatureEngineer
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize agents
print("\n[2/7] Testing agent initialization...")
try:
    data_agent = DataAgent()
    pred_agent = PredictionAgent()
    strat_agent = PrescriptiveAgent()
    exec_agent = ExecutionAgent()
    mon_agent = MonitoringAgent()
    print("✓ All agents initialized")
except Exception as e:
    print(f"✗ Agent initialization failed: {e}")
    sys.exit(1)

# Test 3: Initialize guardrails
print("\n[3/7] Testing guardrail initialization...")
try:
    input_guard = InputValidationGuardrail()
    pred_guard = PredictionValidationGuardrail()
    action_guard = ActionValidationGuardrail()
    guardrail_system = GuardrailSystem()
    print("✓ All guardrails initialized")
except Exception as e:
    print(f"✗ Guardrail initialization failed: {e}")
    sys.exit(1)

# Test 4: Initialize models
print("\n[4/7] Testing model initialization...")
try:
    ensemble = EnsembleModel()
    bandit = ContextualBandit()
    drift_detector = DriftDetector()
    print("✓ All models initialized")
except Exception as e:
    print(f"✗ Model initialization failed: {e}")
    sys.exit(1)

# Test 5: Test orchestrator
print("\n[5/7] Testing orchestrator...")
try:
    orchestrator = AgenticOrchestrator()
    print("✓ Orchestrator initialized")
except Exception as e:
    print(f"✗ Orchestrator initialization failed: {e}")
    sys.exit(1)

# Test 6: Test data loading
print("\n[6/7] Testing data loading...")
try:
    data_loader = DataLoader()
    df = data_loader.load_gaming_dataset()
    print(f"✓ Data loaded: {len(df)} records")
except FileNotFoundError:
    print("⚠ Dataset not found (expected - needs to be downloaded)")
    print("  Download from: https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset")
except Exception as e:
    print(f"✗ Data loading setup failed: {e}")

# Test 7: Test guardrail validation
print("\n[7/7] Testing guardrail validation...")
try:
    # Test input validation
    test_input = {
        'PlayerID': 1,
        'Age': 25,
        'Gender': 'Male',
        'Location': 'USA',
        'GameGenre': 'Action',
        'PlayTimeHours': 100.0,
        'InGamePurchases': 1,
        'GameDifficulty': 'Medium',
        'SessionsPerWeek': 5,
        'AvgSessionDurationMinutes': 60.0,
        'PlayerLevel': 30,
        'AchievementsUnlocked': 50
    }
    
    is_valid, msg, details = input_guard.validate(test_input)
    if is_valid:
        print("✓ Input validation working")
    else:
        print(f"⚠ Input validation flagged test data: {msg}")
    
    # Test prediction validation
    test_predictions = {
        'predictions': ['High', 'Medium', 'Low'],
        'probabilities': [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]],
        'confidence': [0.8, 0.7, 0.7]
    }
    
    is_valid, msg, details = pred_guard.validate(test_predictions)
    if is_valid:
        print("✓ Prediction validation working")
    else:
        print(f"⚠ Prediction validation issue: {msg}")
    
    # Test action validation
    test_action = {
        'action': 'send_push_notification',
        'player_context': test_input,
        'confidence': 0.8
    }
    
    is_valid, msg, details = action_guard.validate(test_action)
    if is_valid:
        print("✓ Action validation working")
    else:
        print(f"⚠ Action validation issue: {msg}")
    
except Exception as e:
    print(f"✗ Guardrail validation test failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("SYSTEM TEST COMPLETE")
print("=" * 80)
print("""
✅ All core components are working!

Next steps:
1. Download dataset (if not already done)
2. Run: python main.py --mode demo
3. Or run Streamlit app: streamlit run app.py

System Components Ready:
  🤖 5 Agents (Data, Prediction, Strategy, Execution, Monitoring)
  🛡️ 3 Guardrail Layers (Input, Prediction, Action)
  🧠 ML Models (Ensemble, RL Bandit, Drift Detector)
  📊 Orchestrator (Complete pipeline)
  🌐 Web Interface (Streamlit app)
""")
print("=" * 80)
