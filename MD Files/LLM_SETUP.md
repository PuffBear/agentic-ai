# 🤖 LLM Chat Setup Guide (Ollama + Llama)

## Quick Setup (5 minutes)

The AI Chat interface now uses **real LLM models** via Ollama for natural, conversational responses!

### Step 1: Install Ollama

#### macOS:
```bash
brew install ollama
```

#### Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### Windows:
Download from: https://ollama.com/download

### Step 2: Start Ollama Server

```bash
# Start the Ollama service (in a terminal)
ollama serve
```

Keep this terminal running!

### Step 3: Pull Llama Model

Open a **new terminal** and run:

```bash
# Pull Llama 3.2 (recommended, ~2GB)
ollama pull llama3.2

# OR pull Llama 2 (alternative)
ollama pull llama2

# OR pull Mistral (faster, smaller)
ollama pull mistral
```

### Step 4: Run the App

```bash
# In your project terminal
source agenticenv/bin/activate
streamlit run app.py
```

That's it! The AI Chat will now use real LLM responses! 🎉

---

## What You Get

### With Ollama Running ✅
- **Natural conversations** with the AI
- **Context-aware** responses about your gaming data
- **Smart insights** and recommendations
- **Conversational** explanations

**Example Interaction:**

**You**: "What factors lead to high engagement?"

**AI**: "Based on analyzing our 40,034 players, high engagement correlates strongly with several key factors. First, playtime is crucial - highly engaged players average 16.2 hours compared to 12.1 overall. Second, session frequency matters significantly, with engaged players logging 8.5 sessions weekly versus 6.5 average. Interestingly, genre preference plays a role too, with Strategy and RPG games showing higher engagement rates. I'd recommend focusing retention efforts on players showing declining session frequency, as that's often an early warning sign..."

### Without Ollama (Fallback) ⚠️
- Quick insight buttons still work
- Dashboard shows stats
- You'll see a message to install Ollama

---

## Changing Models

Edit `app.py` line 84:

```python
# Use different models
st.session_state.llm = OllamaLLM(model="llama3.2")  # Default
# st.session_state.llm = OllamaLLM(model="llama2")   # Alternative
# st.session_state.llm = OllamaLLM(model="mistral")  # Faster
# st.session_state.llm = OllamaLLM(model="codellama") # Code-focused
```

## Available Models

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| llama3.2 | ~2GB | Medium | Excellent | General chat (Recommended) |
| llama2 | ~3.8GB | Slower | Great | Detailed analysis |
| mistral | ~4GB | Fast | Very Good | Quick responses |
| codellama | ~3.8GB | Medium | Good | Technical questions |

## Troubleshooting

### "Ollama is not running"
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve
```

### "Model not found"
```bash
# Pull the model
ollama pull llama3.2

# List available models
ollama list
```

### Slow responses
- Try a smaller model (mistral)
- Check system resources
- Or use the Quick Insights buttons for instant results

### Port conflict
```bash
# Ollama uses port 11434 by default
# Check if something else is using it
lsof -i :11434
```

---

## Performance Tips

1. **First question is slower** - Model loads into memory
2. **Subsequent questions are faster** - Model stays loaded
3. **Close other apps** - For better performance
4. **Use MacBook M-series** - Optimized for Apple Silicon

---

## Example Questions to Try

Once Ollama is running, ask things like:

- "What patterns do you see in the engagement data?"
- "How can we reduce churn?"
- "Which player segments should we target?"
- "What's the correlation between playtime and purchases?"
- "Explain the difference between high and low engagement players"
- "What strategies would you recommend for player retention?"

The AI will analyze the actual dataset and give you intelligent, context-aware responses!

---

## System Requirements

- **RAM**: 8GB minimum (16GB recommended)
- **Disk**: 5-10GB for models
- **CPU**: Modern processor (M1/M2 ideal)
- **OS**: macOS, Linux, or Windows

---

**Ready to chat with your data!** 🚀

For more info: https://ollama.com
