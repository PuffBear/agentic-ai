#!/bin/bash

# Startup script for Agentic Gaming Analytics with LLM Chat

echo "🎮 Starting Agentic Gaming Analytics..."
echo ""

# Check if Ollama is running
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is running"
    
    # Check if llama3.2 model is available
    if ollama list | grep -q "llama3.2"; then
        echo "✅ Llama 3.2 model found"
    else
        echo "⚠️  Llama 3.2 not found. Pulling now..."
        ollama pull llama3.2
    fi
else
    echo "⚠️  Ollama is not running!"
    echo ""
    echo "To enable AI Chat, run in another terminal:"
    echo "  ollama serve"
    echo ""
    echo "Then pull the model:"
    echo "  ollama pull llama3.2"
    echo ""
    echo "App will still work with Quick Insights buttons!"
    echo ""
fi

echo ""
echo "🚀 Launching Streamlit app..."
echo ""

# Activate virtual environment and run app
source agenticenv/bin/activate
streamlit run app.py
