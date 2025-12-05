#!/bin/bash

###############################################################################
# Enhanced Startup Script for Agentic Gaming Analytics with LLM Chat
# 
# Features:
# - Checks Ollama availability
# - Optional Ollama auto-start (background mode)
# - Colored output for better UX
# - Error handling
# - Virtual environment activation
###############################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
AUTO_START_OLLAMA=true  # Set to true to auto-start Ollama in background
OLLAMA_PORT=11434
STREAMLIT_PORT=8501

echo -e "${PURPLE}═══════════════════════════════════════════════════════${NC}"
echo -e "${PURPLE} 🎮 Agentic Gaming Analytics with LLM Chat              ${NC}"
echo -e "${PURPLE}═══════════════════════════════════════════════════════${NC}"
echo ""

###############################################################################
# 1. Virtual Environment Activation
###############################################################################
echo -e "${CYAN}🔧 Activating virtual environment...${NC}"
if [ -d "agenticenv" ]; then
    source agenticenv/bin/activate
    echo -e "${GREEN}✅ Virtual environment activated${NC}"
else
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo -e "${YELLOW}Please create it first:${NC}"
    echo -e "  python -m venv agenticenv"
    echo -e "  source agenticenv/bin/activate"
    echo -e "  pip install -r requirements.txt"
    exit 1
fi

echo ""

###############################################################################
# 2. Ollama Check and Start
###############################################################################
echo -e "${CYAN}🤖 Checking Ollama availability...${NC}"

# Function to check if Ollama is running
check_ollama() {
    curl -s http://localhost:${OLLAMA_PORT}/api/tags > /dev/null 2>&1
    return $?
}

# Check if Ollama is running
if check_ollama; then
    echo -e "${GREEN}✅ Ollama is running on port ${OLLAMA_PORT}${NC}"
    
    # Check if llama3.2 model is available
    echo -e "${CYAN}🔍 Checking for Llama 3.2 model...${NC}"
    if ollama list | grep -q "llama3.2"; then
        echo -e "${GREEN}✅ Llama 3.2 model found${NC}"
    else
        echo -e "${YELLOW}⚠️  Llama 3.2 not found. Pulling now...${NC}"
        echo -e "${CYAN}This may take a few minutes...${NC}"
        ollama pull llama3.2
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Llama 3.2 successfully downloaded${NC}"
        else
            echo -e "${YELLOW}⚠️  Failed to download Llama 3.2${NC}"
            echo -e "${YELLOW}AI Chat will use fallback mode${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠️  Ollama is not running${NC}"
    
    # Auto-start Ollama if configured
    if [ "$AUTO_START_OLLAMA" = true ]; then
        echo -e "${CYAN}🚀 Auto-starting Ollama in background...${NC}"
        
        # Check if ollama command exists
        if command -v ollama &> /dev/null; then
            # Start Ollama in background
            ollama serve > /dev/null 2>&1 &
            OLLAMA_PID=$!
            
            # Wait for Ollama to start (max 10 seconds)
            echo -e "${CYAN}Waiting for Ollama to start...${NC}"
            for i in {1..10}; do
                sleep 1
                if check_ollama; then
                    echo -e "${GREEN}✅ Ollama started successfully (PID: $OLLAMA_PID)${NC}"
                    
                    # Pull model if needed
                    if ! ollama list | grep -q "llama3.2"; then
                        echo -e "${CYAN}📥 Pulling Llama 3.2 model...${NC}"
                        ollama pull llama3.2
                    fi
                    
                    # Setup cleanup on exit
                    trap "echo -e '\\n${YELLOW}Shutting down Ollama...${NC}'; kill $OLLAMA_PID 2>/dev/null" EXIT
                    break
                fi
                
                if [ $i -eq 10 ]; then
                    echo -e "${RED}❌ Failed to start Ollama${NC}"
                    echo -e "${YELLOW}AI Chat will use fallback mode${NC}"
                fi
            done
        else
            echo -e "${RED}❌ Ollama command not found${NC}"
            echo -e "${YELLOW}Please install Ollama first${NC}"
        fi
    else
        # Manual instructions
        echo ""
        echo -e "${BLUE}ℹ️  To enable AI Chat, run in another terminal:${NC}"
        echo -e "  ${CYAN}ollama serve${NC}"
        echo ""
        echo -e "${BLUE}Then pull the model:${NC}"
        echo -e "  ${CYAN}ollama pull llama3.2${NC}"
        echo ""
        echo -e "${GREEN}💡 App will still work with Quick Insights buttons!${NC}"
    fi
fi

echo ""

###############################################################################
# 3. Check Data Directory
###############################################################################
echo -e "${CYAN}📁 Checking data directory...${NC}"
if [ -d "data/raw" ]; then
    CSV_COUNT=$(find data/raw -name "*.csv" | wc -l)
    if [ $CSV_COUNT -gt 0 ]; then
        echo -e "${GREEN}✅ Found ${CSV_COUNT} CSV file(s) in data/raw/${NC}"
    else
        echo -e "${YELLOW}⚠️  No CSV files found in data/raw/${NC}"
        echo -e "${YELLOW}Please download the dataset first${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  data/raw directory not found${NC}"
    mkdir -p data/raw
    echo -e "${GREEN}✅ Created data/raw directory${NC}"
fi

echo ""

###############################################################################
# 4. Launch Streamlit App
###############################################################################
echo -e "${PURPLE}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🚀 Launching Streamlit app on port ${STREAMLIT_PORT}...${NC}"
echo -e "${PURPLE}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${CYAN}📊 Dashboard:${NC}     http://localhost:${STREAMLIT_PORT}"
echo -e "${CYAN}🤖 AI Chat:${NC}       http://localhost:${STREAMLIT_PORT}"
echo -e "${CYAN}🔬 EDA:${NC}           http://localhost:${STREAMLIT_PORT}"
echo -e "${CYAN}🛡️  Guardrails:${NC}   http://localhost:${STREAMLIT_PORT}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the application${NC}"
echo ""

# Launch Streamlit
streamlit run app.py --server.port ${STREAMLIT_PORT}

###############################################################################
# 5. Cleanup (if auto-started Ollama)
###############################################################################
# This is handled by the trap on EXIT (line 77)
