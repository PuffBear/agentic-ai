"""
Simple chat interface using the existing chat_interface.py
to enable LLM-powered conversations about the dataset
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Use existing chat interface if available
try:
    from chat_interface import GamingAnalyticsChat
    
    def create_llm_chat(data):
        """Create LLM chat interface with dataset"""
        return GamingAnalyticsChat()
    
    LLM_AVAILABLE = True
except Exception as e:
    print(f"LLM chat not available: {e}")
    LLM_AVAILABLE = False
    
    def create_llm_chat(data):
        return None
