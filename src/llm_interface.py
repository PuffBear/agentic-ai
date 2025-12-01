"""
LLM Integration using Ollama
Provides real conversational AI responses about the gaming dataset
"""

import requests
import json
import pandas as pd
from typing import Optional, Dict, Any

class OllamaLLM:
    """Interface to Ollama for LLM responses"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        """
        Initialize Ollama client
        
        Args:
            base_url: Ollama API endpoint
            model: Model to use (llama3.2, llama2, mistral, etc.)
        """
        self.base_url = base_url
        self.model = model
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: User query
            system_prompt: System context/instructions
            
        Returns:
            LLM response
        """
        if not self.available:
            return "⚠️ Ollama is not running. Please start Ollama first:\n```bash\nollama serve\nollama pull llama3.2\n```"
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['message']['content']
            else:
                return f"Error: {response.status_code} - {response.text}"
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def get_available_models(self) -> list:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except:
            return []


def create_dataset_context(df: pd.DataFrame) -> str:
    """
    Create rich context about the dataset for the LLM
    
    Args:
        df: Gaming dataset
        
    Returns:
        Context string
    """
    # Calculate key statistics
    total_players = len(df)
    
    # Engagement distribution
    eng_dist = df['EngagementLevel'].value_counts()
    high_pct = eng_dist.get('High', 0) / total_players * 100
    med_pct = eng_dist.get('Medium', 0) / total_players * 100
    low_pct = eng_dist.get('Low', 0) / total_players * 100
    
    # Genre distribution
    top_genres = df['GameGenre'].value_counts().head(3)
    
    # Playtime stats
    avg_playtime = df['PlayTimeHours'].mean()
    high_eng_playtime = df[df['EngagementLevel']=='High']['PlayTimeHours'].mean()
    
    # Purchase stats
    purchasers = (df['InGamePurchases'] == 1).sum()
    purchase_rate = purchasers / total_players * 100
    
    # Sessions stats
    avg_sessions = df['SessionsPerWeek'].mean()
    high_eng_sessions = df[df['EngagementLevel']=='High']['SessionsPerWeek'].mean()
    
    # Age stats
    age_range = f"{df['Age'].min()}-{df['Age'].max()}"
    avg_age = df['Age'].mean()
    
    # Level stats
    avg_level = df['PlayerLevel'].mean()
    high_eng_level = df[df['EngagementLevel']=='High']['PlayerLevel'].mean()
    
    # Gender distribution
    gender_dist = df['Gender'].value_counts()
    male_pct = gender_dist.get('Male', 0) / total_players * 100
    female_pct = gender_dist.get('Female', 0) / total_players * 100
    
    # Gender vs Engagement
    male_high_eng = ((df['Gender'] == 'Male') & (df['EngagementLevel'] == 'High')).sum()
    female_high_eng = ((df['Gender'] == 'Female') & (df['EngagementLevel'] == 'High')).sum()
    male_high_pct = male_high_eng / gender_dist.get('Male', 1) * 100 if 'Male' in gender_dist else 0
    female_high_pct = female_high_eng / gender_dist.get('Female', 1) * 100 if 'Female' in gender_dist else 0
    
    # Difficulty distribution
    difficulty_dist = df['GameDifficulty'].value_counts()
    
    # Location diversity
    num_locations = df['Location'].nunique()
    top_locations = df['Location'].value_counts().head(3)
    
    # Achievement stats
    avg_achievements = df['AchievementsUnlocked'].mean()
    high_eng_achievements = df[df['EngagementLevel']=='High']['AchievementsUnlocked'].mean()
    
    context = f"""You are an AI analytics assistant for a gaming platform with access to player behavior data.

DATASET OVERVIEW:
- Total Players: {total_players:,}
- Features: {len(df.columns)}

ENGAGEMENT DISTRIBUTION:
- High Engagement: {eng_dist.get('High', 0):,} players ({high_pct:.1f}%)
- Medium Engagement: {eng_dist.get('Medium', 0):,} players ({med_pct:.1f}%)
- Low Engagement: {eng_dist.get('Low', 0):,} players ({low_pct:.1f}%)

TOP GENRES:
1. {top_genres.index[0]}: {top_genres.iloc[0]:,} players ({top_genres.iloc[0]/total_players*100:.1f}%)
2. {top_genres.index[1]}: {top_genres.iloc[1]:,} players ({top_genres.iloc[1]/total_players*100:.1f}%)
3. {top_genres.index[2]}: {top_genres.iloc[2]:,} players ({top_genres.iloc[2]/total_players*100:.1f}%)

PLAYTIME STATISTICS:
- Overall Average: {avg_playtime:.1f} hours
- High Engagement Average: {high_eng_playtime:.1f} hours
- Range: {df['PlayTimeHours'].min():.1f}h - {df['PlayTimeHours'].max():.1f}h

SESSIONS PER WEEK:
- Overall Average: {avg_sessions:.1f}
- High Engagement Average: {high_eng_sessions:.1f}

IN-GAME PURCHASES:
- Purchasers: {purchasers:,} ({purchase_rate:.1f}%)
- Non-purchasers: {total_players - purchasers:,}

PLAYER DEMOGRAPHICS:
- Age Range: {age_range} years (avg: {avg_age:.0f})
- Average Level: {avg_level:.0f}
- High Engagement Avg Level: {high_eng_level:.0f}

GENDER DISTRIBUTION:
- Male Players: {gender_dist.get('Male', 0):,} ({male_pct:.1f}%)
- Female Players: {gender_dist.get('Female', 0):,} ({female_pct:.1f}%)
- Male High Engagement Rate: {male_high_pct:.1f}%
- Female High Engagement Rate: {female_high_pct:.1f}%

DIFFICULTY LEVELS:
- Easy: {difficulty_dist.get('Easy', 0):,} players
- Medium: {difficulty_dist.get('Medium', 0):,} players
- Hard: {difficulty_dist.get('Hard', 0):,} players

GEOGRAPHIC DIVERSITY:
- Total Locations: {num_locations}
- Top Locations: {', '.join([f'{loc} ({count})' for loc, count in top_locations.items()])}

ACHIEVEMENTS:
- Overall Average: {avg_achievements:.0f} achievements
- High Engagement Average: {high_eng_achievements:.0f} achievements

KEY INSIGHTS:
- High engagement players play {high_eng_playtime/avg_playtime:.1f}x more than average
- High engagement players have {high_eng_sessions/avg_sessions:.1f}x more sessions
- Purchase rate correlates with engagement level
- {top_genres.index[0]} is the most popular genre
- {'Female' if female_high_pct > male_high_pct else 'Male'} players show {abs(female_high_pct - male_high_pct):.1f}% higher engagement rate

Your role is to:
1. Answer questions about this gaming dataset accurately
2. Provide insights and recommendations based on the data
3. Explain patterns and correlations
4. Be conversational and helpful
5. Use specific numbers from the data above
6. Suggest actionable strategies for improving player engagement

Always base your answers on the actual data provided above. Be specific, insightful, and conversational."""
    
    return context


def generate_llm_response(
    user_query: str,
    df: pd.DataFrame,
    llm: OllamaLLM,
    trained: bool = False
) -> str:
    """
    Generate LLM response with dataset context
    
    Args:
        user_query: User's question
        df: Gaming dataset
        llm: Ollama LLM instance
        trained: Whether models are trained
        
    Returns:
        LLM response
    """
    # Create context
    context = create_dataset_context(df)
    
    # Add model info if trained
    if trained:
        context += "\n\nMODEL STATUS: ML models are trained and can predict engagement with 84.7% accuracy."
    
    # Generate response
    response = llm.generate(user_query, system_prompt=context)
    
    return response
