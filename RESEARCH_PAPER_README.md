# Research Paper - Compilation Instructions

## Document Details

**Title:** Multi-Agent Predictive Analytics with Layered Guardrails: A Case Study on Online Gaming Behavior Prediction

**Author:** Agriya Yadav  
**Format:** LaTeX (Academic Research Paper)  
**Pages:** ~25-30 pages (estimated)  
**File:** `research_paper.tex`

---

## Prerequisites

### Option 1: Local LaTeX Installation (Recommended)

**For macOS:**
```bash
# Install MacTeX (full TeX Live distribution)
brew install --cask mactex

# Or install BasicTeX (minimal, ~100MB)
brew install --cask basictex

# If using BasicTeX, install required packages:
sudo tlmgr update --self
sudo tlmgr install collection-latex
sudo tlmgr install tikz
sudo tlmgr install listings
sudo tlmgr install booktabs
sudo tlmgr install hyperref
```

**For Linux:**
```bash
sudo apt-get install texlive-full
# Or minimal:
sudo apt-get install texlive-latex-base texlive-latex-extra
```

### Option 2: Online (No Installation)

Use **Overleaf**: https://www.overleaf.com
1. Create free account
2. Upload `research_paper.tex`
3. Click "Recompile"

---

## Compilation

### Using Command Line (pdflatex)

```bash
cd /Users/Agriya/Desktop/monsoon25/AI/agentic-ai

# First pass (resolves references)
pdflatex research_paper.tex

# Second pass (builds bibliography and cross-refs)
pdflatex research_paper.tex

# Optional: Third pass for perfect references
pdflatex research_paper.tex
```

**Output:** `research_paper.pdf`

### Using VS Code

1. Install extension: **LaTeX Workshop**
2. Open `research_paper.tex`
3. Press `Cmd+Option+B` (macOS) or `Ctrl+Alt+B` (Linux/Windows)
4. View PDF: `Cmd+Option+V`

### Using TeXShop (macOS GUI)

1. Open `/Applications/TeX/TeXShop.app`
2. File → Open → `research_paper.tex`
3. Click "Typeset" button
4. PDF appears in right pane

---

## Document Structure

### Sections Included:

1. **Title & Abstract** (10 lines)
   - Dataset, task, architecture overview
   - Main results & challenges

2. **Introduction** (2 pages)
   - Context & motivation
   - Assignment requirements
   - Contributions (5 key points)
   - **Personal experience** with agentic AI

3. **Background & Related Work** (3 pages)
   - Tredence article summary (5 agentic features)
   - Medium guardrail article summary (3-layer approach)
   - Dataset description

4. **System Design** (5 pages)
   - Overall architecture diagram (TikZ)
   - 5 agent descriptions:
     - Agent 1: Data Ingestion
     - Agent 2: Multi-Model Prediction
     - Agent 3: Prescriptive Strategy (RL)
     - Agent 4: Execution & Simulation
     - Agent 5: Monitoring & Drift Detection
   - Orchestrator coordination logic

5. **Guardrail Design** (4 pages)
   - Layer 1: Input Validation
   - Layer 2: Prediction Validation (hallucination detection)
   - Layer 3: Action Validation
   - Intervention statistics table

6. **Implementation Details** (2 pages)
   - Environment & tech stack
   - Dataset preprocessing
   - Model training (RF, XGBoost, NN)

7. **Experiments & Results** (5 pages)
   - Model comparison table
   - Per-class performance
   - ROC-AUC analysis
   - Feature engineering impact
   - **Qualitative evaluation**: agent behavior, failure modes, guardrail examples

8. **Discussion** (3 pages)
   - How agentic AI helps predictive analytics
   - Limitations & risks
   - **Challenges faced** (engineering complexity, debugging, HITL)
   - Broader implications

9. **Conclusion** (1 page)
   - Summary of contributions
   - Future work (5 directions)

10. **Appendix** (2 pages)
    - Validation report example
    - Agent system prompt
    - Feature importance table

---

## Key Features

### Professional Academic Formatting

✅ IEEE/ACM-style research paper  
✅ Proper citations (Tredence, Medium, Kaggle)  
✅ Mathematical notation for algorithms  
✅ Code listings with syntax highlighting  
✅ Tables with booktabs styling  
✅ TikZ architecture diagram  

### Content Highlights

✅ **Actual project metrics** from your logs:
- 96.2% accuracy, 95.9% F1-score
- 96.8% hallucination detection
- 15.5ms average guardrail latency
- 2.8% transaction block rate

✅ **Real implementation details**:
- All 5 agents described
- Ensemble learning approach
- Thompson Sampling RL
- Kolmogorov-Smirnov drift detection

✅ **Personal experience section**:
- Emergent behavior observations
- Debugging challenges
- Human-AI collaboration insights

✅ **Qualitative analysis**:
- 3 detailed guardrail intervention episodes
- Agent failure modes
- False positive analysis

---

## Customization

### Before Compiling, Update:

1. **Line 18:** Add your email if different
   ```latex
   \texttt{agriya.yadav@ashoka.edu.in}
   ```

2. **Line 252:** Add instructor name
   ```latex
   We thank Prof. [Instructor Name] for valuable feedback
   ```

3. **Optional:** Add acknowledgments for teammates, TAs, etc.

---

## Troubleshooting

### Common Errors

**Error:** `! LaTeX Error: File 'tikz.sty' not found`
- **Fix:** Install TikZ package
  ```bash
  sudo tlmgr install pgf tikz
  ```

**Error:** `! Package hyperref Error: Wrong driver option 'pdftex'`
- **Fix:** Remove driver option from hyperref (already done in template)

**Error:** Bibliography not appearing
- **Fix:** Run pdflatex **twice** (first pass collects refs, second pass builds them)

**Error:** Code listings showing garbled text
- **Fix:** Ensure UTF-8 encoding:
  - VS Code: Bottom right → UTF-8
  - TeXShop: Preferences → Encoding → UTF-8

### File Size Too Large?

If PDF exceeds upload limits:

1. **Compress images** (if you add any):
   ```bash
   gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 \
      -dPDFSETTINGS=/ebook -dNOPAUSE -dQUIET -dBATCH \
      -sOutputFile=research_paper_compressed.pdf research_paper.pdf
   ```

2. **Remove appendix** sections if not required

---

## Converting to Word (if required)

```bash
# Install pandoc
brew install pandoc

# Convert (preserves most formatting)
pandoc research_paper.tex -o research_paper.docx
```

**Note:** Math equations and TikZ diagrams may need manual cleanup in Word.

---

## Quality Checklist

Before submission, verify:

- [ ] All citations compile without errors
- [ ] Tables fit within margins
- [ ] Code listings are readable (font size)
- [ ] Figures have captions
- [ ] No orphan/widow lines (single lines at page top/bottom)
- [ ] Abstract is 8-10 lines
- [ ] Section numbering is correct
- [ ] References are formatted consistently
- [ ] Your name & email are correct
- [ ] Page count is within assignment limits (if any)

---

## Page Count Estimate

- **Main Content:** 20-22 pages
- **Appendix:** 2-3 pages
- **Total:** ~23-25 pages

To reduce (if needed):
- Remove appendix sections
- Reduce table sizes
- Condense discussion section
- Use smaller margins: `\usepackage[margin=0.8in]{geometry}`

To increase (if needed):
- Add more result tables
- Include confusion matrices
- Add ROC curve figures
- Expand related work section

---

## PDF Preview

After compiling, you should see:
- Professional title page
- Well-formatted abstract
- Clear section hierarchy
- Academic-quality tables
- Syntax-highlighted code
- TikZ architecture diagram
- Properly cited references

---

**Good luck with your submission!** 🎓

If you encounter any LaTeX errors, let me know the exact error message and I'll help fix it.
