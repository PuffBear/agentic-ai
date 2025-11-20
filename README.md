# Agentic Gaming Analytics Platform

A multi-agent AI system for predictive gaming behavior analytics with reinforcement learning and multi-layered guardrails.

## Overview

This project implements a **5-agent agentic AI framework** that addresses predictive analytics with autonomous decision-making, inspired by Tredence's vision of next-generation analytics. The system includes:

- **Multi-Agent Collaboration**: 5 specialized agents working in coordination
- **Real-Time Adaptive Learning**: Reinforcement learning with contextual bandits
- **Multi-Layered Guardrails**: 3-layer defense against hallucinations and risks
- **Autonomous Decision-Making**: Complete Predict → Prescribe → Act → Learn loop
- **Model Drift Detection**: Automated monitoring and retraining triggers

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│            MULTI-LAYER GUARDRAIL PIPELINE               │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Input Validation                              │
│  Layer 2: Prediction Validation                         │
│  Layer 3: Action Validation                             │
└─────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    5-AGENT SYSTEM                        │
├─────────────────────────────────────────────────────────┤
│  Agent 1: Data Ingestion & Preprocessing                │
│  Agent 2: Multi-Model Prediction                        │
│  Agent 3: Prescriptive Strategy                         │
│  Agent 4: Execution & Simulation                        │
│  Agent 5: Monitoring & Adaptive Learning                │
└─────────────────────────────────────────────────────────┘
```

## Dataset

**Source**: [Kaggle - Predict Online Gaming Behavior Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset)

**Features**:
- Demographics: Age, Gender, Location
- Gameplay: PlaytimeHours, SessionsPerWeek, AvgSessionDuration
- Progression: PlayerLevel, AchievementsUnlocked
- Economics: InGamePurchases
- **Target**: EngagementLevel (High/Medium/Low)

## Quick Start

### Prerequisites

```bash
# Python 3.10+
python --version

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Add your API keys (for free LLMs)
# GROQ_API_KEY=your_key_here
```

### Download Dataset

```bash
# Using Kaggle API
kaggle datasets download -d rabieelkharoua/predict-online-gaming-behavior-dataset -p data/raw/
unzip data/raw/predict-online-gaming-behavior-dataset.zip -d data/raw/
```

### Run the System

```bash
# Run full agentic pipeline
python main.py

# Or explore in notebooks
jupyter notebook notebooks/04_full_system_demo.ipynb
```

## Project Structure

```
agentic-gaming-analytics/
├── src/
│   ├── agents/              # 5 specialized agents
│   ├── guardrails/          # 3-layer validation
│   ├── models/              # ML models & RL
│   └── utils/               # Helper functions
├── notebooks/               # Jupyter notebooks for exploration
├── tests/                   # Unit tests
├── docs/                    # Detailed documentation
└── experiments/             # Experiment tracking
```

## Agent Details

### Agent 1: Data Ingestion
- Autonomous data loading and validation
- Real-time feature engineering
- Anomaly detection in input streams

### Agent 2: Multi-Model Prediction
- Ensemble predictions (Random Forest, XGBoost, Neural Network)
- Confidence scoring and uncertainty quantification
- Cross-model consistency checks

### Agent 3: Prescriptive Strategy
- Action recommendation (retention offers, notifications, content suggestions)
- Contextual bandit for optimal action selection
- Risk-reward optimization

### Agent 4: Execution & Simulation
- Simulates actions on test data
- Tracks outcomes and calculates rewards
- Maintains audit trail of all decisions

### Agent 5: Monitoring & Learning
- Model drift detection (KS test, PSI)
- Reinforcement learning policy updates
- Auto-triggers retraining when needed

## Guardrail System

### Layer 1: Input Validation
- Schema validation with Pydantic
- Range checks and data type enforcement
- Adversarial input detection

### Layer 2: Prediction Validation
- Cross-model consistency (hallucination detection)
- Confidence threshold filtering
- Anomaly detection in predictions

### Layer 3: Action Validation
- Rule-based safety constraints
- High-risk decision flagging
- Output monitoring and logging

## Reinforcement Learning

**Approach**: Contextual Multi-Armed Bandit

**Problem**:
- **Context**: Player features
- **Actions**: Discount offers, notifications, content recommendations, no action
- **Reward**: Change in engagement level

**Algorithm**: Thompson Sampling / Upper Confidence Bound (UCB)

**Implementation**: Custom contextual bandit with online learning

## Evaluation Metrics

### Agentic Capabilities
- Agent response time
- Decision accuracy
- Multi-agent coordination efficiency
- Autonomous action success rate

### Guardrails
- False positive/negative rates
- Layer-by-layer effectiveness
- Hallucination detection accuracy

### RL Performance
- Cumulative regret
- Convergence rate
- Reward improvement over time

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md)
- [Guardrails Implementation](docs/GUARDRAILS.md)
- [RL Approach](docs/RL_APPROACH.md)
- [API Reference](docs/API.md)

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## References

1. **Tredence**: [The Next Evolution of Predictive Analytics with Agentic AI](https://www.tredence.com/blog/predictive-analytics-with-agentic-ai)
2. **Medium**: Building a Multi-Layered Agentic Guardrail Pipeline to Reduce Hallucinations and Mitigate Risk
3. **Dataset**: [Kaggle - Predict Online Gaming Behavior](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset)

## License

MIT License

## Author

Agriya - Computer Science & Mathematics @ Ashoka University

---

**Built with**: LangChain, Groq, scikit-learn, XGBoost, VowpalWabbit, MLflow