# Multi-Agent Debate AI

A sophisticated multi-agent debate system with integrated computer vision capabilities, powered by local LLM inference (Ollama).

## 🎯 Project Overview

This project implements a structured debate pipeline where multiple AI agents collaborate to explore questions from different perspectives:

1. **Answer Agent** - Generates two distinct answers (logical and creative)
2. **Critic Agent** - Evaluates answers on correctness, clarity, relevance, completeness, conciseness, and grammar
3. **Refiner Agent** - Improves answers based on critic feedback
4. **Judge Agent** - Determines the strongest answer with detailed reasoning
5. **Vision Agent** - Analyzes images using multimodal capabilities (optional)

The system logs all debates as JSON for reproducibility and analysis. RAG retrieval is available for context-aware discussions.

---

## ✨ Key Features

### Debate System
- **3-round structured debate** with progressive refinement
- **Multi-agent architecture** - Answer, Critic, Refiner, Judge roles
- **JSON logging** - Every debate is timestamped and logged to `logs/` directory
- **Interactive & CLI modes** - Ask interactively or pass question as argument
- **Answer normalization** - Auto-strips boilerplate prefixes from LLM outputs

### Computer Vision Module (5 modes)
- **Describe** - General image description and scene analysis
- **OCR** - Text extraction from images
- **QA** - Answer specific questions about image content
- **Classify** - Identify image category/type with confidence
- **Chart** - Analyze graphs, charts, and data visualizations with trend insights

### Retrieval-Augmented Generation (RAG)
- Pure Python keyword-based retrieval (no external embeddings)
- Context loading from `docs/` directory
- Integration with debate flow for grounded answers

### Testing & Quality
- **Comprehensive test suite** - 7/7 tests passing
  - Debate flow validation
  - CV routing and dispatch
  - Error handling
  - LLM unavailability graceful fallback
- Mock-based testing to isolate components
- Regression tests for stability

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Ollama** running locally (default: `http://localhost:11434`)
- **Virtual environment** (`.venv` included)

### Installation

1. **Clone or download the project**
```powershell
cd "c:\Users\ayush\Downloads\MULTI AGENT DEBATE AI"
```

2. **Activate virtual environment**
```powershell
.\.venv\Scripts\Activate.ps1
```

3. **Install dependencies**
```powershell
pip install -r requirements.txt
```

4. **Configure environment (optional)**
Copy `.env.example` to `.env` and customize:
```
OLLAMA_MODEL=llama3
OLLAMA_VISION_MODEL=moondream
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_REQUEST_TIMEOUT=60
```

---

## 📝 Usage

### Debate Mode

**Interactive mode** - Ask questions in real-time:
```powershell
python main.py
# Prompts for question input
```

**One-shot mode** - Pass question as argument:
```powershell
python main.py "What is binary search?"
python main.py "Explain recursion in one paragraph"
```

**Output:**
- Console: 3-round debate summary + judge decision
- Logs: `logs/debate_<timestamp>_<question>.json` with full details

---

### Computer Vision Mode

**Describe image:**
```powershell
python main.py cv --image "dog.png" --mode describe
```
Generates detailed scene description using multimodal vision model.

**Extract text (OCR):**
```powershell
python main.py cv --image "document.png" --mode ocr
```
Extracts all visible text preserving structure (or returns `NO_TEXT_FOUND`).

**Ask image question:**
```powershell
python main.py cv --image "kitchen.png" --mode qa --question "What appliances are visible?"
```
Two-step process: describe image → answer question via text LLM.

**Classify image:**
```powershell
python main.py cv --image "dog.png" --mode classify
```
Identifies image type (photo, drawing, screenshot, etc.) with explanation.

**Analyze chart:**
```powershell
python main.py cv --image "sales_chart.png" --mode chart --question "What is the trend?"
```
Analyzes graphs/charts for trends, peaks, dips, and notable values.

---

## 🏗️ Project Structure

```
MULTI AGENT DEBATE AI/
├── main.py              # Entrypoint (debate or CV routing)
├── agents.py            # Answer, Critic, Refiner, Judge agents
├── utils.py             # LLM & vision model calls
├── cv_tools.py          # Image analysis functions (5 modes)
├── rag.py               # Pure-Python retrieval system
├── scoring.py           # Answer scoring metrics
├── requirements.txt     # Dependencies
├── .env.example         # Environment template
├── README.md            # This file
├── tests/
│   ├── test_regression.py    # Debate flow tests (3 tests)
│   └── test_cv.py            # Vision routing tests (4 tests)
├── logs/                # Timestamped JSON debate logs
├── docs/                # Context documents for RAG
│   ├── binary_search.txt
│   ├── llm.txt
│   └── recursion.txt
└── .venv/              # Python virtual environment
```

---

## 🧪 Testing

Run all tests:
```powershell
python -m unittest discover -s tests -v
```

Run specific test file:
```powershell
python -m unittest tests.test_regression -v
python -m unittest tests.test_cv -v
```

**Test Results:** ✅ 7/7 passing
- `test_main_module_imports` - Validates imports
- `test_full_debate_run_produces_final_winner` - 3-round debate flow
- `test_call_llm_handles_unavailable_backend` - Error resilience
- `test_cv_cli_prints_result` - CV CLI parsing
- `test_run_cv_mode_classify_dispatches` - Classify mode routing
- `test_run_cv_mode_ocr_dispatches` - OCR mode routing
- `test_run_cv_mode_rejects_invalid_mode` - Invalid mode handling

---

## 🔧 Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_MODEL` | `llama3` | Text generation model |
| `OLLAMA_VISION_MODEL` | `moondream` | Multimodal vision model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_REQUEST_TIMEOUT` | `60` | Request timeout in seconds |

### Model Requirements

- **Text Model:** `llama3` (or compatible LLM)
- **Vision Model:** `moondream` (or compatible multimodal model)

Pull models if needed:
```powershell
ollama pull llama3
ollama pull moondream
```

---

## 📊 Debate Flow

```
User Question
    ↓
[Round 1: Answer Generation]
  ├─ Answer Agent (Logical)
  └─ Answer Agent (Creative)
    ↓
[Round 2: Criticism & Refinement]
  ├─ Critic Agent (Evaluates both)
  └─ Refiner Agent (Improves each)
    ↓
[Round 3: Final Judgment]
  ├─ Judge Agent (Compares final answers)
  └─ Scoring (C, CL, CF metrics)
    ↓
[Output]
  ├─ Console: Winner + reasoning
  └─ JSON Log: Full debate details
```

---

## 📸 Computer Vision Capabilities

| Mode | Input | Output | Use Case |
|------|-------|--------|----------|
| **Describe** | Image | Scene description | Understanding image content |
| **OCR** | Image | Extracted text | Digitizing documents, receipts |
| **QA** | Image + Question | Answer | Specific image analysis |
| **Classify** | Image | Category + confidence | Image categorization, sorting |
| **Chart** | Graph/Chart + Optional question | Trend analysis | Data visualization analysis |

All modes include fallback handling for empty or invalid responses.

---

## 💾 Logging

Every debate run generates a timestamped JSON log in `logs/`:

**Filename:** `debate_<YYYYMMDD>_<HHMMSS>_<question_slug>.json`

**Contents:**
```json
{
  "question": "What is binary search?",
  "timestamp": "2026-05-20T12:00:00",
  "rounds": [
    {
      "round": 1,
      "logical_answer": "...",
      "creative_answer": "..."
    },
    {
      "round": 2,
      "criticism": "...",
      "refined_logical": "...",
      "refined_creative": "..."
    },
    {
      "round": 3,
      "judge_decision": "...",
      "winner": "logical|creative",
      "scores": {...}
    }
  ]
}
```

---

## 🛠️ Technologies

- **Python 3.14** (Windows MinGW)
- **Ollama** - Local LLM inference
- **Requests** - HTTP communication
- **JSON** - Structured logging
- **Unittest** - Testing framework

---

## 📋 Requirements

See `requirements.txt`:
```
certifi
charset-normalizer
idna
requests
urllib3
```

All dependencies are lightweight and focused on HTTP communication (no heavy ML frameworks required).

---

## 🎓 Use Cases

### Educational
- Exploring different perspectives on a topic
- Understanding complex concepts through multi-angle debate
- Comparing logical vs. creative reasoning approaches

### Research
- Generating diverse viewpoints for literature reviews
- Structured argumentation analysis
- Debate quality assessment

### Data Processing
- Document OCR and digitization
- Chart and visualization analysis
- Multimodal image understanding
- Automated image categorization

### Interview Prep
- Practice debate structuring
- Multi-agent system design patterns
- LLM integration and prompt engineering

---

## 🚨 Troubleshooting

### Ollama Connection Error
```
LLM_ERROR: Connection refused
```
**Solution:** Ensure Ollama is running:
```powershell
ollama serve
```

### Vision Model Returns Empty
**Solution:** 
- Try PNG/JPG instead of AVIF
- Use smaller image file sizes
- Check model is loaded: `ollama pull moondream`

### AVIF Format Not Supported
**Solution:** Convert to PNG/JPG:
```powershell
# Using ImageMagick or online converter
convert image.avif image.png
```

### Pillow Build Error on First Run
**Note:** Chart mode may trigger Pillow install on first run (one-time). Wait for completion.

---

## 📜 License

This project is provided as-is for educational and research purposes.

---

## 👤 Author

Developed as a portfolio project showcasing multi-agent LLM systems and computer vision integration.

---

## 📞 Quick Reference

| Task | Command |
|------|---------|
| Ask a question | `python main.py "Your question?"` |
| Describe image | `python main.py cv --image img.png --mode describe` |
| Extract text | `python main.py cv --image doc.png --mode ocr` |
| Question image | `python main.py cv --image img.png --mode qa --question "What?"` |
| Classify image | `python main.py cv --image img.png --mode classify` |
| Analyze chart | `python main.py cv --image chart.png --mode chart` |
| Run tests | `python -m unittest discover -s tests -v` |

---

**Last Updated:** May 20, 2026
**Status:** ✅ Production Ready
