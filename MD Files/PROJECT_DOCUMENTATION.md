# Agentic Gaming Analytics - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Directory Structure](#directory-structure)
4. [Setup & Installation](#setup--installation)
5. [Core Components](#core-components)
6. [Usage Guide](#usage-guide)
7. [API Reference](#api-reference)
8. [Development Guide](#development-guide)

---

## Project Overview

**Agentic Gaming Analytics** is a multi-agent AI system designed for autonomous predictive analytics of online gaming behavior. The system combines:

- **5 Specialized Agents**: Data ingestion, multi-model prediction, prescriptive strategy, execution & simulation, and monitoring & adaptive learning
- **3-Layer Guardrail System**: Input validation, prediction validation, and action validation
- **LLM-Powered Conversational Interface**: Natural language interaction using Groq/Claude
- **Reinforcement Learning**: Contextual bandit for optimal action selection
- **Complete Autonomy**: Predict → Prescribe → Act → Learn loop

### Key Features

✅ **Conversational AI Interface** - Chat naturally with your analytics system  
✅ **Multi-Agent Collaboration** - 5 specialized agents working in coordination  
✅ **Real-Time Adaptive Learning** - RL-powered continuous improvement  
✅ **Multi-Layered Safety** - 3-tier defense against hallucinations and risks  
✅ **Autonomous Decision-Making** - From prediction to action without human intervention  
✅ **Model Drift Detection** - Automated monitoring and retraining triggers

### Dataset

**Source**: [Kaggle - Predict Online Gaming Behavior](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset)

**Features**:
- Demographics: Age, Gender, Location
- Gameplay: PlayTimeHours, SessionsPerWeek, AvgSessionDuration
- Progression: PlayerLevel, AchievementsUnlocked
- Economics: InGamePurchases
- Target: EngagementLevel (High/Medium/Low)

---

## System Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────┐
│                  User Interface                       │
│            (Streamlit Chat / CLI / API)               │
└───────────────────┬──────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────┐
│              LLM Orchestrator Layer                   │
│  (Claude/GPT-4 via Groq - Natural Language ↔ Tools)  │
│  - Intent Recognition                                 │
│  - Planning & Reasoning                               │
│  - Tool Selection & Orchestration                     │
│  - Response Generation                                │
└───────────────────┬──────────────────────────────────┘
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
┌─────────────────┐  ┌─────────────────┐
│   Guardrails    │  │  Agent Tools    │
│   (3 Layers)    │  │  (5 Agents)     │
└─────────────────┘  └─────────────────┘
          │                   │
          └─────────┬─────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   Data & Models     │
         │  - ML Models        │
         │  - Vector Memory    │
         │  - Experiment DB    │
         └─────────────────────┘
```

### Multi-Agent System

```
┌─────────────────────────────────────────────────────────┐
│                MULTI-LAYER GUARDRAIL PIPELINE            │
├─────────────────────────────────────────────────────────┤
│ Layer 1: Input Validation (Schema, Range, Adversarial) │
│ Layer 2: Prediction Validation (Consistency, Confidence)│
│ Layer 3: Action Validation (Safety, Risk Assessment)   │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                    5-AGENT SYSTEM                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Agent 1: Data Ingestion & Preprocessing                │
│  ├─ Load & validate data                                │
│  ├─ Feature engineering                                 │
│  ├─ Anomaly detection                                   │
│  └─ Data quality checks                                 │
│                                                          │
│  Agent 2: Multi-Model Prediction                        │
│  ├─ Ensemble predictions (RF, XGBoost, NN)             │
│  ├─ Confidence scoring                                  │
│  ├─ Cross-model consistency                             │
│  └─ Explainability (SHAP)                               │
│                                                          │
│  Agent 3: Prescriptive Strategy                         │
│  ├─ Action recommendation                               │
│  ├─ Contextual bandit (Thompson Sampling/UCB)          │
│  ├─ Risk-reward optimization                            │
│  └─ Strategy selection                                  │
│                                                          │
│  Agent 4: Execution & Simulation                        │
│  ├─ Action simulation                                   │
│  ├─ Outcome prediction                                  │
│  ├─ Reward calculation                                  │
│  └─ Audit trail                                         │
│                                                          │
│  Agent 5: Monitoring & Adaptive Learning                │
│  ├─ Model drift detection (KS test, PSI)               │
│  ├─ Performance monitoring                              │
│  ├─ RL policy updates                                   │
│  └─ Auto-retraining triggers                            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Conversational Flow

```
User: "Why are players disengaging?"
         ↓
    [LLM Interprets]
         ↓
    [Plans Analysis]
         ↓
┌────────────────────────┐
│ 1. Segment by          │
│    engagement level    │ → Agent 1: Data Ingestion
└────────────────────────┘
         ↓
┌────────────────────────┐
│ 2. Train classifier    │
│    to predict churn    │ → Agent 2: Prediction
└────────────────────────┘
         ↓
┌────────────────────────┐
│ 3. Identify risk       │
│    factors             │ → Agent 2: Feature Importance
└────────────────────────┘
         ↓
┌────────────────────────┐
│ 4. Recommend           │
│    interventions       │ → Agent 3: Strategy
└────────────────────────┘
         ↓
┌────────────────────────┐
│ 5. Simulate impact     │ → Agent 4: Simulation
└────────────────────────┘
         ↓
    [LLM Synthesizes]
         ↓
Agent: "I found that SessionsPerWeek < 3 is the strongest 
       predictor (67% churn risk). I recommend weekly 
       re-engagement campaigns for this segment..."
```

---

## Directory Structure

```
agentic-gaming-analytics/
│
├── src/                          # Source code
│   ├── agents/                   # 5 Specialized agents
│   │   ├── __init__.py
│   │   ├── base_agent.py        # Abstract base class for all agents
│   │   ├── data_ingestion.py   # Agent 1: Data loading & preprocessing
│   │   ├── prediction.py        # Agent 2: Multi-model predictions
│   │   ├── strategy.py          # Agent 3: Prescriptive recommendations
│   │   ├── simulation.py        # Agent 4: Action simulation & execution
│   │   └── monitoring.py        # Agent 5: Drift detection & learning
│   │
│   ├── guardrails/               # 3-Layer validation system
│   │   ├── __init__.py
│   │   ├── base_guardrail.py   # Abstract guardrail class
│   │   ├── input_validator.py  # Layer 1: Input validation
│   │   ├── prediction_validator.py # Layer 2: Prediction checks
│   │   └── action_validator.py # Layer 3: Action safety
│   │
│   ├── models/                   # ML models & RL components
│   │   ├── __init__.py
│   │   ├── ensemble.py          # Ensemble model (RF, XGBoost, NN)
│   │   ├── bandit.py            # Contextual bandit implementation
│   │   ├── drift_detector.py   # Model drift detection
│   │   └── explainer.py         # SHAP/LIME explainability
│   │
│   ├── chat/                     # LLM conversational interface
│   │   ├── __init__.py
│   │   ├── llm_interface.py    # Main LLM orchestrator
│   │   ├── tools.py             # LangChain tool definitions
│   │   ├── memory.py            # Conversation memory management
│   │   └── prompts.py           # System & tool prompts
│   │
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── data_loader.py      # Data loading utilities
│       ├── logger.py            # Logging configuration
│       ├── metrics.py           # Evaluation metrics
│       └── visualization.py     # Plotting utilities
│
├── configs/                      # Configuration files
│   ├── agents.yaml              # Agent configurations
│   ├── models.yaml              # Model hyperparameters
│   ├── guardrails.yaml          # Guardrail thresholds
│   └── llm.yaml                 # LLM settings (model, temperature, etc.)
│
├── data/                         # Data storage
│   ├── raw/                     # Original dataset
│   ├── processed/               # Cleaned & engineered features
│   ├── external/                # External data sources
│   └── embeddings/              # Vector embeddings for memory
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_agent_development.ipynb
│   ├── 03_guardrail_testing.ipynb
│   ├── 04_chat_interface_demo.ipynb
│   └── 05_full_system_demo.ipynb
│
├── tests/                        # Unit & integration tests
│   ├── __init__.py
│   ├── test_agents.py
│   ├── test_guardrails.py
│   ├── test_models.py
│   ├── test_chat_interface.py
│   └── test_integration.py
│
├── logs/                         # Application logs
│   ├── agent_decisions/         # Agent decision logs
│   ├── guardrail_alerts/        # Safety alerts
│   ├── chat_history/            # Conversation logs
│   └── model_performance/       # Model metrics
│
├── experiments/                  # MLflow experiment tracking
│   └── mlruns/                  # MLflow artifacts
│
├── docs/                         # Documentation
│   ├── API_REFERENCE.md         # API documentation
│   ├── AGENT_GUIDE.md           # Agent development guide
│   ├── GUARDRAIL_GUIDE.md       # Guardrail implementation
│   ├── LLM_INTERFACE.md         # Chat interface documentation
│   └── DEPLOYMENT.md            # Deployment instructions
│
├── scripts/                      # Utility scripts
│   ├── download_data.sh         # Download Kaggle dataset
│   ├── setup_env.sh             # Environment setup
│   └── run_experiments.py       # Run experiment suite
│
├── app.py                        # Streamlit web interface
├── main.py                       # CLI entry point
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker compose for services
└── README.md                     # Project README
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- pip or conda
- (Optional) Docker for containerized deployment
- API Keys: Groq API, Anthropic API (for Claude)

### Installation Steps

#### 1. Clone Repository

```bash
git clone https://github.com/PuffBear/agentic-ai.git
cd agentic-ai
```

#### 2. Create Virtual Environment

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n agentic-ai python=3.10
conda activate agentic-ai
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
```
# LLM & Agent Frameworks
langchain==0.1.0
langchain-groq==0.0.1
langchain-anthropic==0.1.0
langgraph==0.0.20

# ML & Data Science
scikit-learn==1.4.0
xgboost==2.0.3
tensorflow==2.15.0
pandas==2.2.0
numpy==1.26.3

# RL & Bandits
vowpalwabbit==9.9.0

# Explainability
shap==0.44.1
lime==0.2.0.1

# Vector Database & Memory
chromadb==0.4.22
faiss-cpu==1.7.4

# Validation & Quality
pydantic==2.5.3
great-expectations==0.18.8

# Experiment Tracking
mlflow==2.10.0

# Web Interface
streamlit==1.30.0
plotly==5.18.0

# Utilities
python-dotenv==1.0.0
pyyaml==6.0.1
tqdm==4.66.1
```

#### 4. Setup Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your API keys:
```env
# LLM API Keys
GROQ_API_KEY=your_groq_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Model Configuration
DEFAULT_LLM_MODEL=llama3-70b-8192
LLM_TEMPERATURE=0.7
MAX_TOKENS=4096

# Paths
DATA_DIR=./data
LOGS_DIR=./logs
EXPERIMENTS_DIR=./experiments

# MLflow
MLFLOW_TRACKING_URI=./experiments/mlruns

# Vector Database
CHROMA_PERSIST_DIR=./data/embeddings

# Guardrail Thresholds
CONFIDENCE_THRESHOLD=0.75
RISK_THRESHOLD=0.8
```

#### 5. Download Dataset

**Option A: Using Kaggle API**
```bash
# Install kaggle CLI
pip install kaggle

# Configure Kaggle credentials (get from kaggle.com/account)
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
bash scripts/download_data.sh
```

**Option B: Manual Download**
1. Go to [Kaggle Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset)
2. Download `online_gaming_behavior_dataset.csv`
3. Place in `data/raw/`

#### 6. Initialize Vector Database

```bash
python -c "from src.chat.memory import initialize_vector_db; initialize_vector_db()"
```

#### 7. Run Tests

```bash
pytest tests/ -v --cov=src --cov-report=html
```

#### 8. Launch Application

**Option A: Streamlit Web UI**
```bash
streamlit run app.py
```

**Option B: CLI**
```bash
python main.py
```

**Option C: Docker**
```bash
docker-compose up --build
```

---

## Core Components

### 1. Agents (`src/agents/`)

Each agent inherits from `BaseAgent` and implements specific functionality:

#### Agent 1: Data Ingestion & Preprocessing
```python
# src/agents/data_ingestion.py
class DataIngestionAgent(BaseAgent):
    """
    Responsible for:
    - Loading and validating data
    - Feature engineering
    - Data quality checks
    - Anomaly detection in inputs
    """
    
    def load_data(self, source: str) -> pd.DataFrame
    def validate_schema(self, df: pd.DataFrame) -> bool
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame
    def detect_anomalies(self, df: pd.DataFrame) -> List[Dict]
```

#### Agent 2: Multi-Model Prediction
```python
# src/agents/prediction.py
class PredictionAgent(BaseAgent):
    """
    Responsible for:
    - Ensemble predictions (RF, XGBoost, NN)
    - Confidence scoring
    - Cross-model consistency checks
    - Explainability (SHAP values)
    """
    
    def predict(self, X: pd.DataFrame) -> Dict
    def get_confidence(self, predictions: np.ndarray) -> float
    def explain_prediction(self, X: pd.DataFrame) -> Dict
    def check_consistency(self, predictions: List) -> bool
```

#### Agent 3: Prescriptive Strategy
```python
# src/agents/strategy.py
class StrategyAgent(BaseAgent):
    """
    Responsible for:
    - Action recommendation
    - Contextual bandit optimization
    - Risk-reward analysis
    - Strategy selection
    """
    
    def recommend_action(self, context: Dict) -> str
    def update_policy(self, context: Dict, action: str, reward: float)
    def optimize_strategy(self, player_segment: str) -> Dict
```

#### Agent 4: Execution & Simulation
```python
# src/agents/simulation.py
class SimulationAgent(BaseAgent):
    """
    Responsible for:
    - Action simulation on test data
    - Outcome prediction
    - Reward calculation
    - Audit trail maintenance
    """
    
    def simulate_action(self, action: str, players: List) -> Dict
    def calculate_reward(self, before: Dict, after: Dict) -> float
    def log_execution(self, action: str, outcome: Dict)
```

#### Agent 5: Monitoring & Adaptive Learning
```python
# src/agents/monitoring.py
class MonitoringAgent(BaseAgent):
    """
    Responsible for:
    - Model drift detection (KS test, PSI)
    - Performance monitoring
    - RL policy updates
    - Auto-retraining triggers
    """
    
    def detect_drift(self, reference: np.ndarray, current: np.ndarray) -> bool
    def monitor_performance(self) -> Dict
    def trigger_retraining(self) -> bool
    def update_rl_policy(self, experiences: List)
```

### 2. Guardrails (`src/guardrails/`)

#### Layer 1: Input Validation
```python
# src/guardrails/input_validator.py
class InputValidator(BaseGuardrail):
    """
    Validates:
    - Schema compliance (Pydantic)
    - Range checks (age, playtime, etc.)
    - Data type enforcement
    - Adversarial input detection
    """
    
    def validate(self, data: Dict) -> Tuple[bool, str]
    def check_schema(self, data: Dict) -> bool
    def check_ranges(self, data: Dict) -> bool
    def detect_adversarial(self, data: Dict) -> bool
```

#### Layer 2: Prediction Validation
```python
# src/guardrails/prediction_validator.py
class PredictionValidator(BaseGuardrail):
    """
    Validates:
    - Cross-model consistency
    - Confidence thresholds
    - Anomaly detection in outputs
    - Hallucination detection
    """
    
    def validate(self, predictions: Dict) -> Tuple[bool, str]
    def check_consistency(self, predictions: List) -> bool
    def check_confidence(self, confidence: float) -> bool
    def detect_hallucination(self, predictions: Dict) -> bool
```

#### Layer 3: Action Validation
```python
# src/guardrails/action_validator.py
class ActionValidator(BaseGuardrail):
    """
    Validates:
    - Rule-based safety constraints
    - High-risk decision flagging
    - Business logic compliance
    - Output monitoring
    """
    
    def validate(self, action: Dict) -> Tuple[bool, str]
    def check_safety(self, action: Dict) -> bool
    def assess_risk(self, action: Dict) -> float
    def enforce_business_rules(self, action: Dict) -> bool
```

### 3. LLM Chat Interface (`src/chat/`)

#### Main Orchestrator
```python
# src/chat/llm_interface.py
class AgenticChatInterface:
    """
    LLM-powered conversational interface that:
    - Interprets natural language queries
    - Plans multi-step analyses
    - Orchestrates agent tool calls
    - Generates natural language responses
    """
    
    def __init__(self):
        self.llm = ChatGroq(model="llama3-70b-8192")
        self.tools = self._initialize_tools()
        self.memory = ConversationBufferMemory()
        self.agent = create_react_agent(self.llm, self.tools)
    
    def query(self, user_input: str) -> str
    def _initialize_tools(self) -> List[Tool]
    def reset_memory(self)
```

#### Tool Definitions
```python
# src/chat/tools.py

@tool
def segment_players(criteria: str) -> Dict:
    """Segments players based on criteria like age, genre, or engagement"""
    pass

@tool
def predict_engagement(player_ids: List[int]) -> Dict:
    """Predicts engagement levels for specified players"""
    pass

@tool
def recommend_strategy(segment: str) -> Dict:
    """Recommends optimal retention/monetization strategy"""
    pass

@tool
def simulate_intervention(action: str, players: List[int]) -> Dict:
    """Simulates the impact of an intervention"""
    pass

@tool
def monitor_performance() -> Dict:
    """Returns current model performance and drift metrics"""
    pass
```

### 4. Models (`src/models/`)

#### Ensemble Model
```python
# src/models/ensemble.py
class EnsembleModel:
    """
    Combines multiple models:
    - RandomForestClassifier
    - XGBoostClassifier
    - Neural Network (TensorFlow)
    
    Uses weighted voting based on validation performance
    """
    
    def fit(self, X: pd.DataFrame, y: pd.Series)
    def predict(self, X: pd.DataFrame) -> np.ndarray
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray
    def get_feature_importance(self) -> Dict
```

#### Contextual Bandit
```python
# src/models/bandit.py
class ContextualBandit:
    """
    Thompson Sampling / UCB implementation for action selection
    
    Context: Player features (age, genre, playtime, etc.)
    Actions: [no_action, discount_offer, push_notification, content_recommendation]
    Reward: Change in engagement level
    """
    
    def select_action(self, context: np.ndarray) -> str
    def update(self, context: np.ndarray, action: str, reward: float)
    def get_best_action(self, context: np.ndarray) -> str
```

---

## Usage Guide

### 1. Streamlit Web Interface

Launch the app:
```bash
streamlit run app.py
```

Navigate to `http://localhost:8501`

**Example Interactions:**

```
User: "Load the gaming dataset and show me key statistics"

Agent: "I've loaded 40,036 players. Here are the key statistics:
       - Age range: 16-49 (median: 30)
       - 6 game genres (Strategy: 22%, RPG: 18%, ...)
       - 3 engagement levels (High: 33%, Medium: 34%, Low: 33%)
       - In-game purchase rate: 8%
       
       What would you like to explore?"

---

User: "Find players at risk of churning"

Agent: "Running churn prediction model...
       
       Found 3,247 high-risk players (8.1% of total).
       
       Top risk factors:
       1. SessionsPerWeek < 3 (67% churn probability)
       2. Hard difficulty + PlayerLevel < 20 (75% churn)
       3. Zero achievements in last 30 days (58% churn)
       
       Would you like me to recommend interventions?"

---

User: "Yes, recommend strategies for each risk group"

Agent: "Here are targeted strategies:
       
       **Low-Frequency Players (< 3 sessions/week)**
       - Weekly re-engagement emails with exclusive rewards
       - Push notifications for time-limited events
       - Expected impact: +15% retention
       
       **Difficulty Struggle (Hard + Low Level)**
       - Auto-adjust to Medium after 3 consecutive losses
       - Tutorial recommendations
       - Expected impact: +22% retention
       
       **Achievement Gap (0 recent achievements)**
       - Easy achievement recommendations
       - Daily login rewards
       - Expected impact: +12% retention
       
       Should I simulate these interventions on test data?"
```

### 2. CLI Interface

```bash
python main.py

# Interactive mode
>>> analyze engagement patterns for Strategy players

>>> predict churn for players aged 18-25

>>> what's the best retention strategy for low-frequency players?

>>> monitor model performance
```

### 3. Python API

```python
from src.chat.llm_interface import AgenticChatInterface

# Initialize chat interface
chat = AgenticChatInterface()

# Query
response = chat.query("Find players likely to quit in the next month")
print(response)

# Multi-turn conversation
chat.query("Segment players by engagement level")
chat.query("For the low engagement group, what's driving disengagement?")
chat.query("Recommend interventions for this group")

# Reset conversation
chat.reset_memory()
```

### 4. Direct Agent Usage

```python
from src.agents import DataIngestionAgent, PredictionAgent

# Load data
data_agent = DataIngestionAgent()
df = data_agent.load_data("data/raw/online_gaming_behavior_dataset.csv")

# Validate and engineer features
df = data_agent.engineer_features(df)

# Predict engagement
prediction_agent = PredictionAgent()
predictions = prediction_agent.predict(df)

# Get explanations
explanations = prediction_agent.explain_prediction(df)
```

---

## API Reference

### Agent Base Class

```python
class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    @abstractmethod
    def execute(self, *args, **kwargs):
        """Main execution method - must be implemented by subclasses"""
        pass
    
    def log(self, message: str, level: str = "INFO"):
        """Log agent actions"""
        pass
    
    def get_status(self) -> Dict:
        """Return agent status"""
        pass
```

### Guardrail Base Class

```python
class BaseGuardrail(ABC):
    """Abstract base class for all guardrails"""
    
    @abstractmethod
    def validate(self, data: Any) -> Tuple[bool, str]:
        """
        Validate input data
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        pass
    
    def log_violation(self, violation: str):
        """Log guardrail violations"""
        pass
```

### Tool Signatures

```python
# Segment Players
def segment_players(
    criteria: str,  # "age" | "genre" | "engagement" | "playtime"
    num_clusters: int = 5
) -> Dict[str, Any]:
    """
    Returns:
        {
            "segments": [...],
            "cluster_centers": [...],
            "segment_sizes": [...]
        }
    """

# Predict Engagement
def predict_engagement(
    player_ids: List[int]
) -> Dict[str, Any]:
    """
    Returns:
        {
            "predictions": [...],
            "confidence": [...],
            "risk_factors": [...]
        }
    """

# Recommend Strategy
def recommend_strategy(
    segment: str,
    objective: str = "retention"  # "retention" | "monetization" | "engagement"
) -> Dict[str, Any]:
    """
    Returns:
        {
            "recommended_action": str,
            "expected_impact": float,
            "confidence": float,
            "alternatives": [...]
        }
    """
```

---

## Development Guide

### Adding a New Agent

1. Create file in `src/agents/`
2. Inherit from `BaseAgent`
3. Implement `execute()` method
4. Add tests in `tests/test_agents.py`
5. Register as a tool in `src/chat/tools.py`

```python
# src/agents/new_agent.py
from src.agents.base_agent import BaseAgent

class NewAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.name = "NewAgent"
    
    def execute(self, *args, **kwargs):
        # Implementation
        pass

# src/chat/tools.py
@tool
def new_agent_tool(param: str) -> Dict:
    """Tool description"""
    agent = NewAgent()
    return agent.execute(param)
```

### Adding a New Guardrail

```python
# src/guardrails/new_guardrail.py
from src.guardrails.base_guardrail import BaseGuardrail

class NewGuardrail(BaseGuardrail):
    def validate(self, data: Any) -> Tuple[bool, str]:
        # Validation logic
        if condition:
            return True, "Valid"
        else:
            return False, "Error message"
```

### Testing Guidelines

```python
# tests/test_new_feature.py
import pytest
from src.agents.new_agent import NewAgent

def test_new_agent_execute():
    agent = NewAgent()
    result = agent.execute(test_input)
    assert result == expected_output

def test_new_agent_error_handling():
    agent = NewAgent()
    with pytest.raises(ValueError):
        agent.execute(invalid_input)
```

### Logging Best Practices

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)

# In your code
logger.info("Agent started processing")
logger.debug(f"Intermediate result: {result}")
logger.warning("Low confidence prediction")
logger.error("Prediction failed", exc_info=True)
```

### Configuration Management

```yaml
# configs/agents.yaml
data_ingestion:
  batch_size: 1000
  validation_threshold: 0.95
  
prediction:
  models:
    - random_forest
    - xgboost
    - neural_network
  ensemble_method: "weighted_voting"
  confidence_threshold: 0.75
```

```python
# Load config
from src.utils.config import load_config

config = load_config("configs/agents.yaml")
batch_size = config['data_ingestion']['batch_size']
```

---

## Performance Metrics

### Agent Metrics
- Response time per agent
- Decision accuracy
- Multi-agent coordination efficiency
- Autonomous action success rate

### Guardrail Metrics
- False positive/negative rates
- Layer-by-layer effectiveness
- Hallucination detection accuracy
- Average validation time

### RL Metrics
- Cumulative regret
- Convergence rate
- Reward improvement over time
- Exploration vs exploitation ratio

---

## Troubleshooting

### Common Issues

**1. LLM API errors**
```
Error: Groq API rate limit exceeded
Solution: Check .env for valid API key, consider implementing retry logic
```

**2. Memory issues with large datasets**
```
Error: MemoryError during model training
Solution: Use batch processing, reduce ensemble size, or increase system RAM
```

**3. Guardrail false positives**
```
Error: Action blocked by guardrail despite being safe
Solution: Adjust thresholds in configs/guardrails.yaml
```

**4. Vector database not initialized**
```
Error: Chroma collection not found
Solution: Run python -c "from src.chat.memory import initialize_vector_db; initialize_vector_db()"
```

---

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Update documentation
6. Submit pull request

---

## License

MIT License - see LICENSE file

---

## Citation

If you use this system in your research, please cite:

```bibtex
@software{agentic_gaming_analytics,
  title = {Agentic Gaming Analytics: Multi-Agent AI for Predictive Player Behavior},
  author = {Agriya},
  year = {2024},
  url = {https://github.com/PuffBear/agentic-ai}
}
```

---

## Acknowledgments

- Inspired by [Tredence's Agentic AI vision](https://www.tredence.com/blog/predictive-analytics-with-agentic-ai)
- Dataset from [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset)
- Built with LangChain, Groq, scikit-learn, XGBoost, and Streamlit

---

**Built by**: Agriya Yadav
**Institution**: Ashoka University  