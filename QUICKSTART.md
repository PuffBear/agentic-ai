# 🚀 QUICK START GUIDE

## Run the AI Chat Interface (Recommended!)

```bash
# 1. Activate environment
source agenticenv/bin/activate

# 2. Launch Streamlit app
streamlit run app.py
```

Then:
1. Click "📁 Load Dataset" (sidebar)
2. Click "🤖 Train Models" (sidebar)  
3. Go to "🤖 AI Chat" tab
4. Start asking questions!

## Example Questions to Ask the AI:

- "What's the engagement distribution?"
- "Which genres are most popular?"
- "Show me churn risk factors"
- "What about playtime patterns?"
- "How many players make purchases?"
- "Can you predict player engagement?"
- "What drives high engagement?"
- "Who are the at-risk players?"

## Or Try the CLI Demo:

```bash
python main.py --mode demo
```

This runs the full pipeline and shows:
- Dataset loading
- Model training
- Sample predictions
- Guardrail validation
- System health

## System Test:

```bash
python test_system.py
```

Validates all components are working correctly.

---

## Features Available:

### 📊 Dashboard Tab
- Dataset statistics
- Engagement charts
- Genre distribution  
- Sample data

### 🤖 AI Chat Tab (NEW!)
- Natural language Q&A
- Data-driven insights
- Quick insight buttons
- Chat history

### 💡 Strategy Tab
- Segment-based recommendations
- Expected impact metrics
- Retention strategies

### 🛡️ Guardrails Tab
- 3-layer validation overview
- Performance metrics
- Violation tracking

### 📈 Monitoring Tab
- Model performance
- Drift detection
- Pipeline statistics

---

**That's it! You're ready to go!** 🎉

For more details, see `FINAL_SUMMARY.md`
