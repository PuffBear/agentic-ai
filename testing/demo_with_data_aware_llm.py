"""
Complete System with DATA-AWARE LLM
LLM analyzes YOUR actual dataset, not general knowledge
"""

from src.agents import PredictionAgent, PrescriptiveAgent
from src.utils import DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

class DataAwareLLM:
    """LLM that uses YOUR actual dataset statistics"""
    
    def __init__(self, df_original, model="llama3.2"):
        """
        Initialize with actual dataset
        
        Args:
            df_original: The original dataframe with all data
            model: Ollama model name
        """
        self.llm = ChatOllama(model=model, temperature=0.1)  # Low temp = focused
        self.df = df_original
        
        # Pre-calculate dataset statistics
        self.stats = self._calculate_stats()
        
        print(f"✅ Data-Aware LLM initialized")
        print(f"   Dataset: {len(self.df):,} players")
        print(f"   Using ACTUAL data, not general knowledge")
    
    def _calculate_stats(self):
        """Calculate actual statistics from YOUR dataset"""
        stats = {
            'total_players': len(self.df),
            'engagement_dist': self.df['EngagementLevel'].value_counts().to_dict(),
            'age_stats': {
                'mean': self.df['Age'].mean(),
                'min': self.df['Age'].min(),
                'max': self.df['Age'].max(),
                'by_engagement': self.df.groupby('EngagementLevel')['Age'].mean().to_dict()
            },
            'playtime_stats': {
                'mean': self.df['PlayTimeHours'].mean(),
                'by_engagement': self.df.groupby('EngagementLevel')['PlayTimeHours'].mean().to_dict()
            },
            'sessions_stats': {
                'mean': self.df['SessionsPerWeek'].mean(),
                'by_engagement': self.df.groupby('EngagementLevel')['SessionsPerWeek'].mean().to_dict()
            },
            'purchases_stats': {
                'purchase_rate': (self.df['InGamePurchases'] == 'Yes').mean(),
                'by_engagement': self.df.groupby('EngagementLevel')['InGamePurchases'].apply(
                    lambda x: (x == 'Yes').mean()
                ).to_dict()
            },
            'level_stats': {
                'mean': self.df['PlayerLevel'].mean(),
                'by_engagement': self.df.groupby('EngagementLevel')['PlayerLevel'].mean().to_dict(),
                'by_purchases': self.df.groupby('InGamePurchases')['PlayerLevel'].mean().to_dict()
            },
            'genre_dist': self.df['GameGenre'].value_counts().to_dict(),
            'location_dist': self.df['Location'].value_counts().to_dict()
        }
        return stats
    
    def ask_about_data(self, question: str) -> str:
        """Ask about YOUR specific dataset"""
        
        # Build context with ACTUAL data
        context = f"""You are analyzing a SPECIFIC gaming dataset with these EXACT statistics:

DATASET SIZE: {self.stats['total_players']:,} players

ENGAGEMENT DISTRIBUTION:
- High: {self.stats['engagement_dist'].get('High', 0):,} players
- Medium: {self.stats['engagement_dist'].get('Medium', 0):,} players  
- Low: {self.stats['engagement_dist'].get('Low', 0):,} players

AGE PATTERNS:
- Overall average: {self.stats['age_stats']['mean']:.1f} years
- High engagement: {self.stats['age_stats']['by_engagement'].get('High', 0):.1f} years
- Medium engagement: {self.stats['age_stats']['by_engagement'].get('Medium', 0):.1f} years
- Low engagement: {self.stats['age_stats']['by_engagement'].get('Low', 0):.1f} years

PLAYTIME PATTERNS:
- Overall average: {self.stats['playtime_stats']['mean']:.1f} hours
- High engagement: {self.stats['playtime_stats']['by_engagement'].get('High', 0):.1f} hours
- Medium engagement: {self.stats['playtime_stats']['by_engagement'].get('Medium', 0):.1f} hours
- Low engagement: {self.stats['playtime_stats']['by_engagement'].get('Low', 0):.1f} hours

SESSIONS PER WEEK:
- Overall average: {self.stats['sessions_stats']['mean']:.1f}
- High engagement: {self.stats['sessions_stats']['by_engagement'].get('High', 0):.1f}
- Medium engagement: {self.stats['sessions_stats']['by_engagement'].get('Medium', 0):.1f}
- Low engagement: {self.stats['sessions_stats']['by_engagement'].get('Low', 0):.1f}

PURCHASE BEHAVIOR:
- Overall purchase rate: {self.stats['purchases_stats']['purchase_rate']:.1%}
- High engagement purchase rate: {self.stats['purchases_stats']['by_engagement'].get('High', 0):.1%}
- Medium engagement purchase rate: {self.stats['purchases_stats']['by_engagement'].get('Medium', 0):.1%}
- Low engagement purchase rate: {self.stats['purchases_stats']['by_engagement'].get('Low', 0):.1%}

PLAYER LEVELS:
- Overall average: {self.stats['level_stats']['mean']:.1f}
- With purchases: {self.stats['level_stats']['by_purchases'].get('Yes', 0):.1f}
- Without purchases: {self.stats['level_stats']['by_purchases'].get('No', 0):.1f}

Answer ONLY using these EXACT numbers. Do NOT use general gaming knowledge.
Be CONCISE (2-3 sentences max). Cite the specific numbers."""

        prompt = f"{context}\n\nQuestion: {question}\n\nAnswer (2-3 sentences, cite exact numbers):"
        
        try:
            messages = [
                SystemMessage(content="You analyze specific datasets. Use ONLY the provided numbers. Be CONCISE."),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def analyze_specific_query(self, query_type: str, **kwargs) -> dict:
        """Run specific pandas analysis then let LLM interpret"""
        
        if query_type == "age_engagement":
            # Calculate actual correlation
            result = self.df.groupby('EngagementLevel').agg({
                'Age': ['mean', 'median', 'std'],
                'PlayerID': 'count'
            }).round(2)
            
            interpretation = self.ask_about_data(
                f"Given these age statistics by engagement: {result.to_dict()}, "
                "what pattern do you see? Answer in 2 sentences."
            )
            
            return {
                'data': result.to_dict(),
                'interpretation': interpretation
            }
        
        elif query_type == "purchases_correlation":
            # Actual correlation in YOUR data
            corr_data = pd.crosstab(
                self.df['InGamePurchases'],
                self.df['EngagementLevel'],
                normalize='index'
            ).round(3) * 100
            
            avg_levels = self.df.groupby('InGamePurchases')['PlayerLevel'].mean()
            
            interpretation = self.ask_about_data(
                f"In our data: Purchase rates by engagement: {corr_data.to_dict()}. "
                f"Average levels: {avg_levels.to_dict()}. "
                "Is there a correlation? Answer in 2 sentences."
            )
            
            return {
                'purchase_rates': corr_data.to_dict(),
                'avg_levels': avg_levels.to_dict(),
                'interpretation': interpretation
            }
        
        elif query_type == "genre_patterns":
            genre_engagement = pd.crosstab(
                self.df['GameGenre'],
                self.df['EngagementLevel'],
                normalize='index'
            ).round(3) * 100
            
            interpretation = self.ask_about_data(
                f"Genre engagement distribution: {genre_engagement.to_dict()}. "
                "Which genre has highest engagement? Answer in 1 sentence."
            )
            
            return {
                'data': genre_engagement.to_dict(),
                'interpretation': interpretation
            }

def main():
    print("\n" + "=" * 80)
    print("🎮 DATA-AWARE LLM - Analyzes YOUR Actual Dataset")
    print("=" * 80)
    
    # Load ORIGINAL dataset (not preprocessed)
    print("\n📊 Loading original dataset...")
    loader = DataLoader()
    df_original = loader.df  # Get the original dataframe
    
    print(f"✅ Loaded {len(df_original):,} players")
    
    # Initialize Data-Aware LLM
    print("\n🧠 Initializing Data-Aware LLM...")
    llm = DataAwareLLM(df_original)
    
    # Demo: Ask questions about YOUR data
    print("\n" + "=" * 80)
    print("💬 INTERACTIVE MODE - Ask about YOUR dataset")
    print("=" * 80)
    print("\nExample questions:")
    print("  - What patterns do you see in player engagement based on age groups?")
    print("  - Is there a correlation between in-game purchases and player levels?")
    print("  - Which game genre has the highest engagement?")
    print("  - Do older players engage more or less?")
    print("\nType 'quit' to exit, 'analyze' for deep analysis\n")
    
    while True:
        try:
            question = input("🤔 You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            # Check for specific analysis commands
            if 'age' in question.lower() and 'engagement' in question.lower():
                print("🧠 LLM: Running analysis on YOUR data...")
                result = llm.analyze_specific_query('age_engagement')
                print(f"\n📊 Actual Data:\n{pd.DataFrame(result['data'])}\n")
                print(f"💭 Interpretation: {result['interpretation']}\n")
            
            elif 'purchase' in question.lower() and ('level' in question.lower() or 'correlation' in question.lower()):
                print("🧠 LLM: Running analysis on YOUR data...")
                result = llm.analyze_specific_query('purchases_correlation')
                print(f"\n📊 Purchase Rates: {result['purchase_rates']}")
                print(f"📊 Avg Levels: {result['avg_levels']}")
                print(f"\n💭 Interpretation: {result['interpretation']}\n")
            
            elif 'genre' in question.lower():
                print("🧠 LLM: Running analysis on YOUR data...")
                result = llm.analyze_specific_query('genre_patterns')
                print(f"\n📊 Actual Data:\n{pd.DataFrame(result['data'])}\n")
                print(f"💭 Interpretation: {result['interpretation']}\n")
            
            else:
                # General question with dataset context
                print("🧠 LLM: ", end="", flush=True)
                answer = llm.ask_about_data(question)
                print(answer + "\n")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
