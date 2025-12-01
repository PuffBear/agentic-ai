"""
Communication Intelligence Agent
Agent 6: Analyzes text communication for insights
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)


class CommunicationIntelligenceAgent:
    """
    Agent 6: Communication Intelligence
    
    Analyzes player communication for:
    - Sentiment analysis
    - Toxicity detection  
    - Emotional state tracking
    - Topic extraction
    - Engagement patterns
    """
    
    def __init__(self):
        self.name = "communication_intelligence_agent"
        logger.info(f"Initializing {self.name}...")
        
        # Initialize NLP models (all FREE from Hugging Face)
        try:
            # Sentiment Analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            
            # Emotion Detection (7 emotions: joy, sadness, anger, fear, love, surprise, neutral)
            self.emotion_detector = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None
            )
            
            # Toxicity Detection
            try:
                from detoxify import Detoxify
                self.toxicity_detector = Detoxify('original')
                self.has_toxicity = True
            except:
                logger.warning("Detoxify not available. Install with: pip install detoxify")
                self.has_toxicity = False
            
            logger.info(f"✓ {self.name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NLP models: {e}")
            raise
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute communication analysis
        
        Args:
            input_data: Dict with 'mode' and relevant data
            
        Returns:
            Analysis results
        """
        mode = input_data.get('mode', 'analyze')
        
        try:
            if mode == 'analyze_message':
                return self._analyze_single_message(input_data)
            
            elif mode == 'analyze_conversation':
                return self._analyze_conversation(input_data)
            
            elif mode == 'analyze_player_history':
                return self._analyze_player_history(input_data)
            
            elif mode == 'detect_patterns':
                return self._detect_communication_patterns(input_data)
            
            else:
                return {'error': f'Unknown mode: {mode}'}
        
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            return {'error': str(e)}
    
    def _analyze_single_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single message"""
        message = data.get('message', '')
        
        if not message or len(message.strip()) == 0:
            return {'error': 'Empty message'}
        
        # Sentiment Analysis
        sentiment = self.sentiment_analyzer(message[:512])[0]  # Limit length
        
        # Emotion Detection
        emotions = self.emotion_detector(message[:512])
        top_emotion = max(emotions[0], key=lambda x: x['score'])
        
        # Toxicity Detection
        toxicity = {}
        if self.has_toxicity:
            toxicity = self.toxicity_detector.predict(message)
        
        # Generate insights
        insights = self._generate_message_insights(
            message, sentiment, top_emotion, toxicity
        )
        
        return {
            'message': message,
            'sentiment': {
                'label': sentiment['label'],
                'score': float(sentiment['score'])
            },
            'emotion': {
                'label': top_emotion['label'],
                'score': float(top_emotion['score']),
                'all_emotions': [
                    {'emotion': e['label'], 'score': float(e['score'])} 
                    for e in emotions[0]
                ]
            },
            'toxicity': toxicity if toxicity else None,
            'insights': insights,
            'alerts': self._generate_alerts(sentiment, top_emotion, toxicity)
        }
    
    def _analyze_conversation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a sequence of messages (conversation)"""
        messages = data.get('messages', [])
        
        if not messages:
            return {'error': 'No messages provided'}
        
        results = []
        emotional_timeline = []
        sentiment_timeline = []
        toxicity_scores = []
        
        for i, msg in enumerate(messages):
            # Analyze each message
            if isinstance(msg, str):
                text = msg
                timestamp = i
            else:
                text = msg.get('text', msg.get('message', ''))
                timestamp = msg.get('timestamp', i)
            
            analysis = self._analyze_single_message({'message': text})
            
            results.append({
                'timestamp': timestamp,
                'message': text,
                **analysis
            })
            
            # Track timelines
            if 'emotion' in analysis:
                emotional_timeline.append({
                    'timestamp': timestamp,
                    'emotion': analysis['emotion']['label'],
                    'score': analysis['emotion']['score']
                })
            
            if 'sentiment' in analysis:
                sentiment_timeline.append({
                    'timestamp': timestamp,
                    'sentiment': analysis['sentiment']['label'],
                    'score': analysis['sentiment']['score']
                })
            
            if analysis.get('toxicity'):
                toxicity_scores.append({
                    'timestamp': timestamp,
                    'score': analysis['toxicity'].get('toxicity', 0)
                })
        
        # Analyze patterns
        patterns = self._detect_emotional_patterns(emotional_timeline)
        
        return {
            'total_messages': len(messages),
            'message_analyses': results,
            'emotional_timeline': emotional_timeline,
            'sentiment_timeline': sentiment_timeline,
            'toxicity_timeline': toxicity_scores,
            'patterns': patterns,
            'summary': self._summarize_conversation(results)
        }
    
    def _analyze_player_history(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a player's communication history"""
        player_id = data.get('player_id')
        messages = data.get('messages', [])
        
        if not messages:
            return {'error': 'No message history'}
        
        # Analyze all messages
        conversation_analysis = self._analyze_conversation({'messages': messages})
        
        # Player-specific insights
        avg_sentiment = np.mean([
            m['sentiment']['score'] if m['sentiment']['label'] == 'POSITIVE' else -m['sentiment']['score']
            for m in conversation_analysis['message_analyses']
            if 'sentiment' in m
        ])
        
        toxicity_avg = 0
        if self.has_toxicity and conversation_analysis['toxicity_timeline']:
            toxicity_avg = np.mean([
                t['score'] for t in conversation_analysis['toxicity_timeline']
            ])
        
        # Emotional profile
        emotion_counts = {}
        for msg in conversation_analysis['message_analyses']:
            if 'emotion' in msg:
                emotion = msg['emotion']['label']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else 'neutral'
        
        return {
            'player_id': player_id,
            'total_messages': len(messages),
            'average_sentiment': float(avg_sentiment),
            'average_toxicity': float(toxicity_avg),
            'dominant_emotion': dominant_emotion,
            'emotion_distribution': emotion_counts,
            'communication_style': self._classify_communication_style(
                avg_sentiment, emotion_counts, len(messages)
            ),
            'risk_level': self._assess_risk_level(avg_sentiment, toxicity_avg),
            'full_analysis': conversation_analysis
        }
    
    def _detect_communication_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect patterns across multiple players or sessions"""
        conversations = data.get('conversations', [])
        
        if not conversations:
            return {'error': 'No conversations provided'}
        
        all_sentiments = []
        all_emotions = []
        all_toxicity = []
        
        for conv in conversations:
            analysis = self._analyze_conversation({'messages': conv})
            
            all_sentiments.extend([
                m['sentiment']['score'] 
                for m in analysis['message_analyses'] 
                if 'sentiment' in m
            ])
            
            all_emotions.extend([
                m['emotion']['label']
                for m in analysis['message_analyses']
                if 'emotion' in m
            ])
            
            if self.has_toxicity:
                all_toxicity.extend([
                    t['score'] for t in analysis.get('toxicity_timeline', [])
                ])
        
        return {
            'total_conversations': len(conversations),
            'average_sentiment': float(np.mean(all_sentiments)) if all_sentiments else 0,
            'average_toxicity': float(np.mean(all_toxicity)) if all_toxicity else 0,
            'emotion_distribution': self._count_emotions(all_emotions),
            'insights': self._generate_pattern_insights(all_sentiments, all_emotions, all_toxicity)
        }
    
    def _generate_message_insights(
        self, 
        message: str, 
        sentiment: Dict, 
        emotion: Dict, 
        toxicity: Dict
    ) -> List[str]:
        """Generate human-readable insights"""
        insights = []
        
        # Sentiment insights
        if sentiment['label'] == 'POSITIVE' and sentiment['score'] > 0.8:
            insights.append("💚 Highly positive sentiment detected")
        elif sentiment['label'] == 'NEGATIVE' and sentiment['score'] > 0.8:
            insights.append("❤️ Highly negative sentiment - player may be frustrated")
        
        # Emotion insights
        if emotion['label'] == 'anger' and emotion['score'] > 0.7:
            insights.append("😠 Strong anger detected - potential rage quit risk")
        elif emotion['label'] == 'joy' and emotion['score'] > 0.7:
            insights.append("😊 Player is enjoying the experience")
        elif emotion['label'] == 'sadness' and emotion['score'] > 0.7:
            insights.append("😢 Sadness detected - player may be discouraged")
        
        # Toxicity insights
        if toxicity and toxicity.get('toxicity', 0) > 0.7:
            insights.append("⚠️ High toxicity - moderation recommended")
        
        return insights
    
    def _generate_alerts(
        self, 
        sentiment: Dict, 
        emotion: Dict, 
        toxicity: Dict
    ) -> List[Dict]:
        """Generate actionable alerts"""
        alerts = []
        
        # Toxicity alert
        if toxicity and toxicity.get('toxicity', 0) > 0.7:
            alerts.append({
                'type': 'TOXIC_CONTENT',
                'severity': 'HIGH',
                'action': 'Consider muting or warning player',
                'score': float(toxicity['toxicity'])
            })
        
        # Rage alert
        if emotion['label'] == 'anger' and emotion['score'] > 0.8:
            alerts.append({
                'type': 'RAGE_DETECTED',
                'severity': 'MEDIUM',
                'action': 'Suggest break or reduce difficulty',
                'score': float(emotion['score'])
            })
        
        # Extreme negativity
        if sentiment['label'] == 'NEGATIVE' and sentiment['score'] > 0.9:
            alerts.append({
                'type': 'EXTREME_NEGATIVITY',
                'severity': 'MEDIUM',
                'action': 'Check for underlying issues',
                'score': float(sentiment['score'])
            })
        
        return alerts
    
    def _detect_emotional_patterns(self, timeline: List[Dict]) -> Dict[str, Any]:
        """Detect patterns in emotional timeline"""
        if len(timeline) < 2:
            return {'pattern': 'insufficient_data'}
        
        emotions = [e['emotion'] for e in timeline]
        
        # Detect rage spiral (progression to anger)
        anger_indices = [i for i, e in enumerate(emotions) if e == 'anger']
        if len(anger_indices) >= 2:
            if anger_indices[-1] - anger_indices[0] <= len(emotions) * 0.5:
                return {
                    'pattern': 'rage_spiral',
                    'description': 'Player showing increasing anger',
                    'risk': 'high'
                }
        
        # Detect positive momentum
        positive_emotions = ['joy', 'love', 'surprise']
        positive_count = sum(1 for e in emotions if e in positive_emotions)
        if positive_count / len(emotions) > 0.6:
            return {
                'pattern': 'positive_momentum',
                'description': 'Player is having a good time',
                'risk': 'low'
            }
        
        # Detect emotional volatility
        unique_emotions = len(set(emotions))
        if unique_emotions >= len(emotions) * 0.7:
            return {
                'pattern': 'volatile',
                'description': 'Rapidly changing emotions',
                'risk': 'medium'
            }
        
        return {'pattern': 'stable', 'description': 'Consistent emotional state'}
    
    def _summarize_conversation(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Summarize overall conversation"""
        sentiments = [a['sentiment']['label'] for a in analyses if 'sentiment' in a]
        emotions = [a['emotion']['label'] for a in analyses if 'emotion' in a]
        
        positive_pct = (sentiments.count('POSITIVE') / len(sentiments) * 100) if sentiments else 0
        negative_pct = (sentiments.count('NEGATIVE') / len(sentiments) * 100) if sentiments else 0
        
        return {
            'overall_sentiment': 'Positive' if positive_pct > negative_pct else 'Negative',
            'positive_percentage': round(positive_pct, 1),
            'negative_percentage': round(negative_pct, 1),
            'most_common_emotion': max(set(emotions), key=emotions.count) if emotions else 'neutral',
            'total_alerts': sum(len(a.get('alerts', [])) for a in analyses)
        }
    
    def _classify_communication_style(
        self, 
        avg_sentiment: float, 
        emotion_counts: Dict, 
        message_count: int
    ) -> str:
        """Classify player's communication style"""
        if message_count < 3:
            return "Silent"
        elif message_count > 20:
            chatty = "Chatty"
        else:
            chatty = "Moderate"
        
        if avg_sentiment > 0.3:
            tone = "Positive"
        elif avg_sentiment < -0.3:
            tone = "Negative"
        else:
            tone = "Neutral"
        
        return f"{chatty}, {tone}"
    
    def _assess_risk_level(self, avg_sentiment: float, avg_toxicity: float) -> str:
        """Assess player risk level"""
        if avg_toxicity > 0.5 or avg_sentiment < -0.5:
            return "HIGH"
        elif avg_toxicity > 0.3 or avg_sentiment < -0.2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _count_emotions(self, emotions: List[str]) -> Dict[str, int]:
        """Count emotion occurrences"""
        counts = {}
        for emotion in emotions:
            counts[emotion] = counts.get(emotion, 0) + 1
        return counts
    
    def _generate_pattern_insights(
        self, 
        sentiments: List[float], 
        emotions: List[str], 
        toxicity: List[float]
    ) -> List[str]:
        """Generate insights from patterns"""
        insights = []
        
        if sentiments:
            avg_sent = np.mean(sentiments)
            if avg_sent > 0.5:
                insights.append("Overall community sentiment is very positive")
            elif avg_sent < -0.3:
                insights.append("Community showing signs of dissatisfaction")
        
        if toxicity and np.mean(toxicity) > 0.3:
            insights.append("Elevated toxicity levels detected across conversations")
        
        if emotions:
            emotion_counts = self._count_emotions(emotions)
            top_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
            insights.append(f"Most common emotion: {top_emotion}")
        
        return insights
    
    def generate_demo_data(self) -> List[str]:
        """Generate demo chat messages for testing"""
        return [
            "Let's go team! We got this!",
            "Nice shot! Great play!",
            "Can someone help mid?",
            "This lag is terrible...",
            "Come on guys, focus!",
            "WTF is this matchmaking",
            "You're all trash",
            "I'm done with this game",
            "Actually that was pretty good",
            "GG everyone, fun game!"
        ]
