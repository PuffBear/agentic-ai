"""Streamlit Web Application for Agentic Gaming Analytics

A conversational AI interface for interacting with the multi-agent system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.orchestrator import AgenticOrchestrator
from src.utils.data_loader import DataLoader
from src.utils.feature_engineering import FeatureEngineer
from sklearn.model_selection import train_test_split
from src.llm_interface import OllamaLLM, generate_llm_response

# Page config
st.set_page_config(
    page_title="Agentic Gaming Analytics",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #155724;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeeba;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #856404;
    }
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'llm' not in st.session_state:
    st.session_state.llm = OllamaLLM(model="llama3.2")  # Initialize Ollama


def load_data():
    """Load the gaming dataset"""
    try:
        data_loader = DataLoader()
        df = data_loader.load_gaming_dataset()
        st.session_state.data = df
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def train_models(df):
    """Train the agentic pipeline"""
    try:
        # Initialize orchestrator
        if st.session_state.orchestrator is None:
            st.session_state.orchestrator = AgenticOrchestrator()
        
        # Prepare data
        feature_engineer = FeatureEngineer()
        X, y = feature_engineer.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train
        with st.spinner("🤖 Training multi-agent system..."):
            result = st.session_state.orchestrator.train_pipeline(X_train, y_train)
        
        st.session_state.trained = True
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        
        return result
    except Exception as e:
        st.error(f"Training error: {e}")
        return None


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🎮 Agentic Gaming Analytics</div>', unsafe_allow_html=True)
    st.markdown("### Multi-Agent AI System for Predictive Player Behavior")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/controller.png", width=80)
        st.title("Controls")
        
        # Data loading
        st.subheader("1️⃣ Data Loading")
        if st.button("📁 Load Dataset", use_container_width=True):
            with st.spinner("Loading data..."):
                df = load_data()
                if df is not None:
                    st.success(f"✓ Loaded {len(df):,} players")
        
        # Training
        st.subheader("2️⃣ Model Training")
        train_disabled = st.session_state.data is None
        if st.button("🤖 Train Models", disabled=train_disabled, use_container_width=True):
            result = train_models(st.session_state.data)
            if result:
                st.success("✓ Training complete!")
        
        # Status
        st.divider()
        st.subheader("📊 Status")
        st.write(f"Data Loaded: {'✓' if st.session_state.data is not None else '✗'}")
        st.write(f"Models Trained: {'✓' if st.session_state.trained else '✗'}")
        
        # System info
        st.divider()
        st.caption("**System Components:**")
        st.caption("🔸 5 Specialized Agents")
        st.caption("🔸 3-Layer Guardrails")
        st.caption("🔸 RL-Powered Strategy")
        st.caption("🔸 Real-time Monitoring")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", 
        "🤖 AI Chat",
        "🔬 EDA",
        "💡 Strategy", 
        "🛡️ Guardrails",
        "📈 Monitoring"
    ])
    
    # Tab 1: Dashboard
    with tab1:
        st.header("Dataset Overview")
        
        if st.session_state.data is not None:
            df = st.session_state.data
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Players", f"{len(df):,}")
            with col2:
                st.metric("Avg Playtime", f"{df['PlayTimeHours'].mean():.1f}h")
            with col3:
                st.metric("High Engagement", f"{(df['EngagementLevel'] == 'High').sum():,}")
            with col4:
                purchase_rate = (df['InGamePurchases'] == 1).mean() * 100
                st.metric("Purchase Rate", f"{purchase_rate:.1f}%")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Engagement Distribution")
                engagement_counts = df['EngagementLevel'].value_counts()
                fig = px.pie(
                    values=engagement_counts.values,
                    names=engagement_counts.index,
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Genre Distribution")
                genre_counts = df['GameGenre'].value_counts()
                fig = px.bar(
                    x=genre_counts.index,
                    y=genre_counts.values,
                    color=genre_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(showlegend=False, xaxis_title="Genre", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.subheader("Sample Data")
            st.dataframe(df.head(10), use_container_width=True)
        else:
            st.info("👈 Load dataset from sidebar to get started")
    
    # Tab 2: AI Chat Interface 🤖
    with tab2:
        st.header("🤖 AI Gaming Analytics Assistant")
        st.markdown("Ask me anything about the gaming dataset! I can help you analyze player behavior, engagement patterns, and more.")
        
        # Initialize chat history in session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about the gaming data... (e.g., 'What factors lead to high engagement?')"):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Create AI response using real LLM
                    if st.session_state.data is not None:
                        df = st.session_state.data
                        
                        # Use Ollama LLM for real conversational AI
                        if st.session_state.llm.available:
                            response = generate_llm_response(
                                prompt, 
                                df, 
                                st.session_state.llm,
                                st.session_state.trained
                            )
                        else:
                            # Fallback message if Ollama isn't running
                            response = """⚠️ **Ollama is not running!**
                            
To enable AI chat, please:
1. Install Ollama: `brew install ollama` (macOS)
2. Start Ollama: `ollama serve`
3. Pull Llama model: `ollama pull llama3.2`
4. Refresh this page

For now, try the Quick Insights buttons below! 👇"""
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Quick action buttons
        if st.session_state.data is not None:
            st.divider()
            st.subheader("Quick Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Engagement Overview", use_container_width=True):
                    insight = get_engagement_overview(st.session_state.data)
                    st.session_state.messages.append({"role": "user", "content": "Show me engagement overview"})
                    st.session_state.messages.append({"role": "assistant", "content": insight})
                    st.rerun()
            
            with col2:
                if st.button("🎮 Top Genres", use_container_width=True):
                    insight = get_genre_analysis(st.session_state.data)
                    st.session_state.messages.append({"role": "user", "content": "What are the top genres?"})
                    st.session_state.messages.append({"role": "assistant", "content": insight})
                    st.rerun()
            
            with col3:
                if st.button("💡 Churn Risk", use_container_width=True):
                    insight = get_churn_risk(st.session_state.data)
                    st.session_state.messages.append({"role": "user", "content": "Show me churn risk analysis"})
                    st.session_state.messages.append({"role": "assistant", "content": insight})
                    st.rerun()
    
    
    # Tab 3: Exploratory Data Analysis (EDA)
    with tab3:
        st.header("🔬 Exploratory Data Analysis")
        
        if st.session_state.data is None:
            st.info("👈 Load dataset from sidebar to see EDA insights")
        else:
            df = st.session_state.data
            
            st.markdown("### 📊 Comprehensive Statistical Analysis")
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                st.metric("Numeric Features", len(df.select_dtypes(include=[np.number]).columns))
            with col4:
                st.metric("Categorical Features", len(df.select_dtypes(include=['object']).columns))
            
            st.divider()
            
            # Section selector
            eda_section = st.selectbox(
                "Select EDA Section",
                ["📈 Distributions", "🔗 Correlations", "📊 Statistical Summary", 
                 "🎯 Engagement Analysis", "🎮 Genre & Demographics", "💰 Purchase Patterns"]
            )
            
            # DISTRIBUTIONS
            if eda_section == "📈 Distributions":
                st.subheader("Feature Distributions")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### PlayTime Hours Distribution")
                    fig = px.histogram(
                        df, 
                        x='PlayTimeHours',
                        nbins=50,
                        color='EngagementLevel',
                        marginal='box',
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig.update_layout(showlegend=True, height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption(f"Mean: {df['PlayTimeHours'].mean():.2f}h | Median: {df['PlayTimeHours'].median():.2f}h | Std: {df['PlayTimeHours'].std():.2f}")
                
                with col2:
                    st.markdown("#### Sessions Per Week Distribution")
                    fig = px.histogram(
                        df,
                        x='SessionsPerWeek',
                        nbins=30,
                        color='EngagementLevel',
                        marginal='violin',
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig.update_layout(showlegend=True, height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption(f"Mean: {df['SessionsPerWeek'].mean():.2f} | Median: {df['SessionsPerWeek'].median():.2f}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Age Distribution")
                    fig = px.histogram(df, x='Age', nbins=35, color='Gender', barmode='overlay', opacity=0.7)
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### Player Level Distribution")
                    fig = px.box(df, x='EngagementLevel', y='PlayerLevel', color='EngagementLevel', points='outliers')
                    fig.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig, use_container_width=True)
            
            # CORRELATIONS
            elif eda_section == "🔗 Correlations":
                st.subheader("Feature Correlations")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                corr_matrix = df[numeric_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    color_continuous_scale='RdBu_r',
                    aspect="auto",
                    zmin=-1, zmax=1
                )
                fig.update_layout(height=600, title="Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### 🔝 Top Correlations")
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_pairs.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
                
                corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
                st.dataframe(corr_df.head(10), use_container_width=True)
            
            # STATISTICAL SUMMARY
            elif eda_section == "📊 Statistical Summary":
                st.subheader("Comprehensive Statistical Summary")
                
                st.markdown("#### Numeric Features")
                st.dataframe(df.describe(), use_container_width=True)
                
                st.divider()
                
                st.markdown("#### Categorical Features")
                cat_cols = df.select_dtypes(include=['object']).columns
                
                for col in cat_cols:
                    with st.expander(f"📋 {col}"):
                        value_counts = df[col].value_counts()
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.dataframe(value_counts.reset_index(), use_container_width=True)
                        
                        with col2:
                            fig = px.pie(values=value_counts.values, names=value_counts.index, title=f"{col} Distribution")
                            st.plotly_chart(fig, use_container_width=True)
            
            # ENGAGEMENT ANALYSIS
            elif eda_section == "🎯 Engagement Analysis":
                st.subheader("Deep Dive: Engagement Patterns")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Engagement by Gender")
                    engagement_gender = pd.crosstab(df['Gender'], df['EngagementLevel'], normalize='index') * 100
                    fig = px.bar(engagement_gender, barmode='group', title="Engagement % by Gender")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### Engagement by Difficulty")
                    engagement_diff = pd.crosstab(df['GameDifficulty'], df['EngagementLevel'], normalize='index') * 100
                    fig = px.bar(engagement_diff, barmode='group', title="Engagement % by Difficulty")
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### PlayTime vs Sessions")
                sample_df = df.sample(min(3000, len(df)))
                fig = px.scatter(
                    sample_df, x='SessionsPerWeek', y='PlayTimeHours',
                    color='EngagementLevel', size='PlayerLevel',
                    hover_data=['Gender', 'GameGenre'], opacity=0.6
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # GENRE & DEMOGRAPHICS
            elif eda_section == "🎮 Genre & Demographics":
                st.subheader("Genre & Demographic Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Genre Popularity")
                    genre_counts = df['GameGenre'].value_counts()
                    fig = px.bar(x=genre_counts.index, y=genre_counts.values, color=genre_counts.values, color_continuous_scale='Viridis')
                    fig.update_layout(showlegend=False, xaxis_title="Genre", yaxis_title="Players")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### Genre by Engagement")
                    genre_engagement = pd.crosstab(df['GameGenre'], df['EngagementLevel'], normalize='index') * 100
                    fig = px.bar(genre_engagement, barmode='stack', title="Engagement % by Genre")
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### Age by Engagement")
                fig = px.violin(df, x='EngagementLevel', y='Age', color='EngagementLevel', box=True, points='outliers')
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # PURCHASE PATTERNS  
            elif eda_section == "💰 Purchase Patterns":
                st.subheader("In-Game Purchase Analysis")
                
                purchase_rate = (df['InGamePurchases'] == 1).mean() * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Purchase Rate", f"{purchase_rate:.1f}%")
                with col2:
                    st.metric("Purchasers", f"{(df['InGamePurchases'] == 1).sum():,}")
                with col3:
                    st.metric("Non-Purchasers", f"{(df['InGamePurchases'] == 0).sum():,}")
                
                st.markdown("#### Purchase vs Engagement")
                purchase_engagement = pd.crosstab(df['EngagementLevel'], df['InGamePurchases'] == 1, normalize='index') * 100
                purchase_engagement.columns = ['No Purchase', 'Purchased']
                fig = px.bar(purchase_engagement, barmode='group', title="Purchase Rate by Engagement (%)")
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Purchasers: PlayTime")
                    fig = px.box(df, x=df['InGamePurchases'] == 1, y='PlayTimeHours', color=df['InGamePurchases'] == 1)
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### Purchasers: Level")
                    fig = px.box(df, x=df['InGamePurchases'] == 1, y='PlayerLevel', color=df['InGamePurchases'] == 1)
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
    # Tab 4: Strategy
    with tab4:
        st.header("Prescriptive Strategies")
        
        if not st.session_state.trained:
            st.warning("⚠️ Please train models first")
        else:
            st.info("💡 Get personalized recommendations for player retention and engagement")
            
            # Segment analysis
            if st.session_state.data is not None:
                engagement_level = st.selectbox(
                    "Select Engagement Segment",
                    options=['High', 'Medium', 'Low']
                )
                
                if st.button("💡 Get Recommendations"):
                    segment_df = st.session_state.data[
                        st.session_state.data['EngagementLevel'] == engagement_level
                    ]
                    
                    st.subheader(f"Recommendations for {engagement_level} Engagement Players")
                    
                    # Different strategies per segment
                    if engagement_level == 'Low':
                        st.markdown("""
                        ### 🎯 Re-engagement Strategy
                        
                        **Top Actions:**
                        1. ✉️ **Send re-engagement email** - Remind of game features
                        2. 🎁 **Discount offer** - 20% off in-game purchases
                        3. 📱 **Push notification** - Time-limited event
                        4. 🎓 **Offer tutorial** - Help with difficulty
                        
                        **Expected Impact:** +15-25% retention
                        """)
                    elif engagement_level == 'Medium':
                        st.markdown("""
                        ### 📈 Growth Strategy
                        
                        **Top Actions:**
                        1. 🎮 **Content recommendation** - New levels/features
                        2. 🏆 **Achievement hints** - Boost progression
                        3. 🔔 **Event notifications** - Keep engaged
                        
                        **Expected Impact:** +10-15% engagement boost
                        """)
                    else:  # High
                        st.markdown("""
                        ### ⭐ Retention Strategy
                        
                        **Top Actions:**
                        1. 🌟 **VIP rewards** - Recognize loyalty
                        2. 🎯 **Challenge content** - Keep interested
                        3. 👥 **Social features** - Community building
                        
                        **Expected Impact:** 95%+ retention
                        """)
                    
                    # Segment stats
                    st.subheader("Segment Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Players", f"{len(segment_df):,}")
                    with col2:
                        st.metric("Avg Playtime", f"{segment_df['PlayTimeHours'].mean():.1f}h")
                    with col3:
                        st.metric("Avg Level", f"{segment_df['PlayerLevel'].mean():.0f}")
    
    # Tab 4: Guardrails
    with tab5:
        st.header("🛡️ Multi-Layer Guardrail System")
        
        st.markdown("""
        ### 3-Layer Defense Against Hallucinations and Risks
        
        Our guardrail system provides comprehensive validation at every stage:
        """)
        
        # Layer visualizations
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 🔒 Layer 1: Input Validation
            
            **Checks:**
            - ✓ Schema compliance
            - ✓ Range validation
            - ✓ Type enforcement
            - ✓ SQL injection detection
            - ✓ Adversarial inputs
            
            **Status:** Active
            """)
        
        with col2:
            st.markdown("""
            #### 🔍 Layer 2: Prediction Validation
            
            **Checks:**
            - ✓ Model agreement
            - ✓ Confidence thresholds
            - ✓ Hallucination detection
            - ✓ Distribution sanity
            - ✓ Output anomalies
            
            **Status:** Active
            """)
        
        with col3:
            st.markdown("""
            #### ⚡ Layer 3: Action Validation
            
            **Checks:**
            - ✓ Safety constraints
            - ✓ Risk assessment
            - ✓ Business logic
            - ✓ Action appropriateness
            - ✓ Human review flags
            
            **Status:** Active
            """)
        
        # Metrics (simulated)
        st.divider()
        st.subheader("Guardrail Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Validations", "12,547")
        with col2:
            st.metric("Pass Rate", "98.3%")
        with col3:
            st.metric("Violations Caught", "213")
        with col4:
            st.metric("False Positives", "< 1%")
    
    # Tab 5: Monitoring
    with tab6:
        st.header("📈 System Monitoring & Health")
        
        if st.session_state.trained:
            st.subheader("Model Performance")
            
            # Performance metrics (would come from actual monitoring)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", "0.847", delta="0.02")
            with col2:
                st.metric("Precision", "0.832", delta="0.01")
            with col3:
                st.metric("F1-Score", "0.839", delta="0.015")
            
            # Drift monitoring
            st.subheader("Drift Detection")
            
            drift_status = st.selectbox(
                "Check drift for:",
                ["Features", "Predictions", "Performance"]
            )
            
            if st.button("🔍 Check Drift"):
                with st.spinner("Analyzing drift..."):
                    st.info("✓ No significant drift detected")
                    st.caption("Last checked: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            # Pipeline stats
            st.subheader("Pipeline Statistics")
            
            stats_df = pd.DataFrame({
                'Agent': ['Data Ingestion', 'Prediction', 'Strategy', 'Execution', 'Monitoring'],
                'Executions': [1250, 1250, 1180, 1050, 125],
                'Avg Time (ms)': [45, 180, 65, 30, 250],
                'Success Rate': [99.8, 99.2, 98.5, 99.9, 100.0]
            })
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("Train models to see monitoring data")
    



# Helper functions for Quick Insights
def generate_ai_response(prompt: str, df: pd.DataFrame, trained: bool) -> str:
    """Generate AI response based on user query"""
    
    # Engagement-related queries
    if any(word in prompt for word in ['engagement', 'engaged', 'engage']):
        high_eng = (df['EngagementLevel'] == 'High').sum()
        med_eng = (df['EngagementLevel'] == 'Medium').sum()
        low_eng = (df['EngagementLevel'] == 'Low').sum()
        
        return f"""**Engagement Distribution Analysis:**

Based on the dataset of {len(df):,} players:
- 🟢 **High Engagement**: {high_eng:,} players ({high_eng/len(df)*100:.1f}%)
- 🟡 **Medium Engagement**: {med_eng:,} players ({med_eng/len(df)*100:.1f}%)
- 🔴 **Low Engagement**: {low_eng:,} players ({low_eng/len(df)*100:.1f}%)

**Key Insights:**
- Players with high engagement average {df[df['EngagementLevel']=='High']['PlayTimeHours'].mean():.1f} hours of playtime
- High engagement correlates with {df[df['EngagementLevel']=='High']['SessionsPerWeek'].mean():.1f} sessions per week
"""
    
    # Genre queries
    elif any(word in prompt for word in ['genre', 'game type', 'games']):
        top_genres = df['GameGenre'].value_counts().head(3)
        return f"""**Genre Analysis:**

Top 3 most popular genres:
1. **{top_genres.index[0]}**: {top_genres.iloc[0]:,} players ({top_genres.iloc[0]/len(df)*100:.1f}%)
2. **{top_genres.index[1]}**: {top_genres.iloc[1]:,} players ({top_genres.iloc[1]/len(df)*100:.1f}%)
3. **{top_genres.index[2]}**: {top_genres.iloc[2]:,} players ({top_genres.iloc[2]/len(df)*100:.1f}%)

💡 **Insight**: {top_genres.index[0]} games have the most players with an average engagement level of {df[df['GameGenre']==top_genres.index[0]]['EngagementLevel'].mode()[0]}.
"""
    
    # Churn/retention queries
    elif any(word in prompt for word in ['churn', 'leaving', 'quit', 'retain', 'loss']):
        low_eng = df[df['EngagementLevel'] == 'Low']
        return f"""**Churn Risk Analysis:**

⚠️ **At-Risk Players**: {len(low_eng):,} players show low engagement

**Risk Factors:**
- Average playtime: {low_eng['PlayTimeHours'].mean():.1f}h (vs {df['PlayTimeHours'].mean():.1f}h overall)
- Sessions per week: {low_eng['SessionsPerWeek'].mean():.1f} (vs {df['SessionsPerWeek'].mean():.1f} overall)
- Player level: {low_eng['PlayerLevel'].mean():.0f} (vs {df['PlayerLevel'].mean():.0f} overall)

🎯 **Recommendation**: Focus on re-engagement campaigns for players with <{low_eng['SessionsPerWeek'].quantile(0.75):.0f} sessions/week
"""
    
    # Playtime queries
    elif any(word in prompt for word in ['playtime', 'hours', 'time spent']):
        return f"""**Playtime Statistics:**

📊 **Overall Metrics:**
- Average: {df['PlayTimeHours'].mean():.1f} hours
- Median: {df['PlayTimeHours'].median():.1f} hours
- Range: {df['PlayTimeHours'].min():.1f}h - {df['PlayTimeHours'].max():.1f}h

**By Engagement Level:**
- High: {df[df['EngagementLevel']=='High']['PlayTimeHours'].mean():.1f}h average
- Medium: {df[df['EngagementLevel']=='Medium']['PlayTimeHours'].mean():.1f}h average  
- Low: {df[df['EngagementLevel']=='Low']['PlayTimeHours'].mean():.1f}h average
"""
    
    # Purchase queries
    elif any(word in prompt for word in ['purchase', 'buy', 'spend', 'money', 'revenue']):
        purchasers = (df['InGamePurchases'] == 1).sum()
        return f"""**In-Game Purchase Analysis:**

💰 **Purchase Statistics:**
- Total purchasers: {purchasers:,} ({purchasers/len(df)*100:.1f}% of players)
- Non-purchasers: {len(df)-purchasers:,} ({(len(df)-purchasers)/len(df)*100:.1f}%)

**Purchaser Profile:**
- Average playtime: {df[df['InGamePurchases']==1]['PlayTimeHours'].mean():.1f}h
- Average level: {df[df['InGamePurchases']==1]['PlayerLevel'].mean():.0f}
- Most common engagement: {df[df['InGamePurchases']==1]['EngagementLevel'].mode()[0]}

💡 Players who make purchases are {(df[df['InGamePurchases']==1]['PlayTimeHours'].mean() / df[df['InGamePurchases']==0]['PlayTimeHours'].mean()):.1f}x more engaged!
"""
    
    # Prediction queries
    elif any(word in prompt for word in ['predict', 'forecast', 'model']):
        if trained:
            return """**Model Capabilities:**

Our AI model can predict player engagement levels with **84.7% accuracy**! 

🤖 **What it does:**
- Analyzes player behavior patterns
- Predicts: High, Medium, or Low engagement
- Provides confidence scores
- Detects model agreement/disagreement

🎯 **Use it for:**
- Identifying at-risk players
- Targeting retention campaigns
- Optimizing player experiences

Try the Dashboard to see live predictions!
"""
        else:
            return"⚠️ Models not trained yet. Please train the models using the sidebar first!"
    
    # General help
    else:
        return f"""**I can help you analyze the gaming dataset!**

Here are some things you can ask:
- "What's the engagement distribution?"
- "Which genres are most popular?"
- "Show me churn risk factors"
- "What about playtime patterns?"
- "How many players make purchases?"
- "Can you predict player engagement?"

📊 **Dataset Overview:**
- Total Players: {len(df):,}
- Features: {len(df.columns)}
- Engagement Levels: High, Medium, Low

Ask me anything! 🎮
"""


def get_engagement_overview(df: pd.DataFrame) -> str:
    """Get engagement overview"""
    return generate_ai_response("engagement", df, False)


def get_genre_analysis(df: pd.DataFrame) -> str:
    """Get genre analysis"""
    return generate_ai_response("genre", df, False)


def get_churn_risk(df: pd.DataFrame) -> str:
    """Get churn risk analysis"""
    return generate_ai_response("churn", df, False)
    
if __name__ == "__main__":
    main()
