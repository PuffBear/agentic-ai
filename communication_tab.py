"""
Streamlit tab content for Communication Intelligence Agent
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.agents.communication_agent import CommunicationIntelligenceAgent


def render_communication_tab():
    """Render the Communication Intelligence tab"""
    
    st.header("💬 Communication Intelligence")
    st.markdown("**Agent 6:** Analyze player communication for sentiment, emotions, toxicity & patterns")
    
    # Initialize agent
    if 'comm_agent' not in st.session_state:
        with st.spinner("Loading NLP models (first time may take a minute)..."):
            try:
                st.session_state.comm_agent = CommunicationIntelligenceAgent()
                st.success("✓ Communication Agent loaded!")
            except Exception as e:
                st.error(f"Error loading agent: {e}")
                st.info("Install dependencies: `pip install transformers torch detoxify`")
                return
    
    agent = st.session_state.comm_agent
    
    # Mode selector
    mode = st.selectbox(
        "Analysis Mode",
        ["📝 Single Message", "💬 Conversation", "📊 Player History", "🎮 Demo"]
    )
    
    st.divider()
    
    # SINGLE MESSAGE ANALYSIS
    if mode == "📝 Single Message":
        st.subheader("Analyze a Single Message")
        
        message = st.text_area(
            "Enter message:",
            placeholder="e.g., 'This game is amazing but the lag is terrible!'",
            height=100
        )
        
        if st.button("🔍 Analyze", type="primary"):
            if message:
                with st.spinner("Analyzing..."):
                    result = agent.execute({
                        'mode': 'analyze_message',
                        'message': message
                    })
                
                if 'error' not in result:
                    # Display Results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Sentiment",
                            result['sentiment']['label'],
                            f"{result['sentiment']['score']:.2%}"
                        )
                    
                    with col2:
                        st.metric(
                            "Emotion",
                            result['emotion']['label'].title(),
                            f"{result['emotion']['score']:.2%}"
                        )
                    
                    with col3:
                        if result.get('toxicity'):
                            tox_score = result['toxicity'].get('toxicity', 0)
                            st.metric(
                                "Toxicity",
                                "High" if tox_score > 0.5 else "Low",
                                f"{tox_score:.2%}"
                            )
                        else:
                            st.metric("Toxicity", "N/A", "Install detoxify")
                    
                    # Insights
                    if result.get('insights'):
                        st.subheader("💡 Insights")
                        for insight in result['insights']:
                            st.info(insight)
                    
                    # Alerts
                    if result.get('alerts'):
                        st.subheader("⚠️ Alerts")
                        for alert in result['alerts']:
                            st.warning(f"**{alert['type']}:** {alert['action']} (Severity: {alert['severity']})")
                    
                    # Detailed Emotions
                    with st.expander("📊 All Emotions Detected"):
                        emotions_df = pd.DataFrame(result['emotion']['all_emotions'])
                        emotions_df = emotions_df.sort_values('score', ascending=False)
                        
                        fig = px.bar(
                            emotions_df,
                            x='emotion',
                            y='score',
                            color='score',
                            color_continuous_scale='RdYlGn'
                        )
                        fig.update_layout(showlegend=False, height=300)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(result['error'])
            else:
                st.warning("Please enter a message to analyze")
    
    # CONVERSATION ANALYSIS
    elif mode == "💬 Conversation":
        st.subheader("Analyze a Conversation")
        
        st.markdown("Enter messages (one per line):")
        conversation = st.text_area(
            "Messages:",
            placeholder="Let's go team!\nNice shot!\nThis lag is terrible...\nI'm done",
            height=200
        )
        
        if st.button("🔍 Analyze Conversation", type="primary"):
            if conversation:
                messages = [msg.strip() for msg in conversation.split('\n') if msg.strip()]
                
                with st.spinner("Analyzing conversation..."):
                    result = agent.execute({
                        'mode': 'analyze_conversation',
                        'messages': messages
                    })
                
                if 'error' not in result:
                    # Summary
                    st.subheader("📊 Conversation Summary")
                    
                    summary = result['summary']
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Messages", result['total_messages'])
                    with col2:
                        st.metric("Overall Sentiment", summary['overall_sentiment'])
                    with col3:
                        st.metric("Positive %", f"{summary['positive_percentage']:.1f}%")
                    with col4:
                        st.metric("Alerts", summary['total_alerts'])
                    
                    # Emotional Timeline
                    st.subheader("🎭 Emotional Journey")
                    
                    if result['emotional_timeline']:
                        timeline_df = pd.DataFrame(result['emotional_timeline'])
                        
                        fig = px.line(
                            timeline_df,
                            x='timestamp',
                            y='score',
                            color='emotion',
                            markers=True,
                            title="Emotion Intensity Over Time"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Sentiment Timeline  
                    st.subheader("📈 Sentiment Timeline")
                    
                    if result['sentiment_timeline']:
                        sent_df = pd.DataFrame(result['sentiment_timeline'])
                        
                        # Convert to numeric
                        sent_df['sentiment_score'] = sent_df.apply(
                            lambda x: x['score'] if x['sentiment'] == 'POSITIVE' else -x['score'],
                            axis=1
                        )
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=sent_df['timestamp'],
                            y=sent_df['sentiment_score'],
                            mode='lines+markers',
                            name='Sentiment',
                            line=dict(color='green' if sent_df['sentiment_score'].mean() > 0 else 'red')
                        ))
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig.update_layout(
                            title="Sentiment Progression",
                            yaxis_title="Sentiment Score",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Patterns
                    if result.get('patterns'):
                        st.subheader("🔍 Detected Patterns")
                        pattern = result['patterns']
                        
                        st.info(f"**Pattern:** {pattern.get('pattern', 'N/A').title()}")
                        if 'description' in pattern:
                            st.write(pattern['description'])
                        if 'risk' in pattern:
                            risk_color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
                            st.write(f"Risk Level: {risk_color.get(pattern['risk'], '⚪')} {pattern['risk'].upper()}")
                    
                    # Detailed Messages
                    with st.expander("📝 Message-by-Message Analysis"):
                        for i, msg_analysis in enumerate(result['message_analyses']):
                            st.markdown(f"**Message {i+1}:** {msg_analysis['message']}")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.caption(f"Sentiment: {msg_analysis.get('sentiment', {}).get('label', 'N/A')}")
                            with col2:
                                st.caption(f"Emotion: {msg_analysis.get('emotion', {}).get('label', 'N/A')}")
                            with col3:
                                if msg_analysis.get('alerts'):
                                    st.caption(f"⚠️ {len(msg_analysis['alerts'])} alerts")
                            st.divider()
                else:
                    st.error(result['error'])
            else:
                st.warning("Please enter a conversation to analyze")
    
    # PLAYER HISTORY
    elif mode == "📊 Player History":
        st.subheader("Analyze Player Communication History")
        
        player_id = st.text_input("Player ID:", value="Player_001")
        
        st.markdown("Enter player's message history (one per line):")
        history = st.text_area(
            "Message History:",
            placeholder="GG!\nNice game everyone\nThis is fun...",
            height=200
        )
        
        if st.button("🔍 Analyze Player", type="primary"):
            if history:
                messages = [msg.strip() for msg in history.split('\n') if msg.strip()]
                
                with st.spinner("Analyzing player history..."):
                    result = agent.execute({
                        'mode': 'analyze_player_history',
                        'player_id': player_id,
                        'messages': messages
                    })
                
                if 'error' not in result:
                    # Player Profile
                    st.subheader(f"👤 Player Profile: {player_id}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Messages", result['total_messages'])
                    with col2:
                        sentiment_label = "Positive" if result['average_sentiment'] > 0 else "Negative"
                        st.metric("Avg Sentiment", sentiment_label, f"{result['average_sentiment']:.2f}")
                    with col3:
                        st.metric("Dominant Emotion", result['dominant_emotion'].title())
                    with col4:
                        risk_colors = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}
                        st.metric("Risk Level", f"{risk_colors.get(result['risk_level'], '')} {result['risk_level']}")
                    
                    # Communication Style
                    st.info(f"**Communication Style:** {result['communication_style']}")
                    
                    # Emotion Distribution
                    st.subheader("🎭 Emotion Distribution")
                    
                    emotion_df = pd.DataFrame(
                        list(result['emotion_distribution'].items()),
                        columns=['Emotion', 'Count']
                    )
                    
                    fig = px.pie(
                        emotion_df,
                        values='Count',
                        names='Emotion',
                        title="Player's Emotional Profile"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Toxicity
                    if result['average_toxicity'] > 0:
                        st.subheader("⚠️ Toxicity Analysis")
                        tox_pct = result['average_toxicity'] * 100
                        
                        if tox_pct > 50:
                            st.error(f"High toxicity detected: {tox_pct:.1f}%")
                        elif tox_pct > 30:
                            st.warning(f"Moderate toxicity: {tox_pct:.1f}%")
                        else:
                            st.success(f"Low toxicity: {tox_pct:.1f}%")
                else:
                    st.error(result['error'])
            else:
                st.warning("Please enter message history")
    
    # DEMO
    elif mode == "🎮 Demo":
        st.subheader("Demo: Analyze Sample Gaming Chat")
        
        st.markdown("Try out the agent with pre-generated demo messages!")
        
        demo_messages = agent.generate_demo_data()
        
        st.code('\n'.join(demo_messages), language=None)
        
        if st.button("🚀 Run Demo Analysis", type="primary"):
            with st.spinner("Analyzing demo conversation..."):
                result = agent.execute({
                    'mode': 'analyze_conversation',
                    'messages': demo_messages
                })
            
            if 'error' not in result:
                # Quick Summary
                st.success("✓ Demo analysis complete!")
                
                summary = result['summary']
                st.metric("Overall Sentiment", summary['overall_sentiment'])
                st.metric("Most Common Emotion", summary['most_common_emotion'].title())
                
                # Show timeline
                if result['emotional_timeline']:
                    timeline_df = pd.DataFrame(result['emotional_timeline'])
                    
                    fig = px.line(
                        timeline_df,
                        x='timestamp',
                        y='score',
                        color='emotion',
                        markers=True,
                        title="Demo: Emotional Journey"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 Try the other modes with your own data!")
            else:
                st.error(result['error'])
    
    # Info section
    st.divider()
    with st.expander("ℹ️ About Communication Intelligence Agent"):
        st.markdown("""
        **Agent 6: Communication Intelligence**
        
        This agent analyzes player communication to extract insights about:
        - **Sentiment:** Positive/Negative tone detection
        - **Emotions:** 7 emotions (joy, sadness, anger, fear, love, surprise, neutral)
        - **Toxicity:** Harmful content detection
        - **Patterns:** Rage spirals, positive momentum, volatility
        
        **Use Cases:**
        - Predict player churn from sentiment shifts
        - Moderate toxic behavior automatically
        - Identify frustrated players before rage quit
        - Track team cohesion and dynamics
        - Personalize player experience based on emotional state
        
        **Models Used (All FREE):**
        - Sentiment: DistilBERT (Hugging Face)
        - Emotions: RoBERTa-emotion (Hugging Face)
        - Toxicity: Detoxify (local model)
        """)
