# Multimodal Debate AI

A local-first multi-agent debate system that combines retrieval, LLM-based debate agents, and optional computer vision analysis using Ollama. It is designed for educational, research, and experimentation use, with an emphasis on iterative reasoning, structured feedback, and explainable answer selection.

## Project at a Glance

- 1 CLI entry point: `main.py`
- 4 agent roles: Answer, Critic, Refiner, Judge
- 3 debate rounds by default
- 5 computer vision modes: `describe`, `ocr`, `qa`, `classify`, `chart`
- 2 Ollama model families: text and vision
- 3 source documents in `docs/`
- 2 test files covering regression and CV behavior
- 1 JSON log generated per debate question

## Why This Project Exists

This repository implements a compact multi-agent reasoning loop where two competing answers are generated, criticized, refined, and judged. The system is intentionally lightweight and local-first:

- it does not require a paid API
- it runs against a locally hosted Ollama backend
- it supports optional document retrieval from local text sources
- it adds CV functionality for OCR, image QA, chart explanation, and classification

It is a practical sandbox for experimenting with:

- adversarial answer generation
- iterative self-correction
- judge-based selection
- retrieval-augmented QA
- multimodal image understanding via Ollama vision models

## Architecture Overview

### 1) Debate pipeline

The core workflow in `main.py` is:

1. Retrieve context using `retrieve_context()` from `rag.py`
2. Generate two initial answers with `answer_agent()`
   - logical style
   - creative style
3. Normalize answers to remove meta-prefaces and keep them concise
4. Run 3 rounds of debate:
   - Judge picks a winner
   - Critic provides structured feedback
   - Refiner rewrites both answers
   - Scoring evaluates correctness, clarity, and confidence
5. Final judge selects the final winner
6. Save a structured JSON log under `logs/`

### 2) Agent roles

The `agents.py` module implements four specialized agents:

- `answer_agent(...)`: produces initial answers
- `critic_agent(...)`: reviews correctness, completeness, relevance, grammar
- `refiner_agent(...)`: rewrites answers using critic feedback
- `judge_agent(...)`: selects the preferred answer using quality criteria

### 3) Retrieval layer

`rag.py` provides document retrieval using:

- keyword overlap scoring with stopword filtering
- optional embedding-based retrieval if `sentence-transformers` and `faiss` are available

This allows the system to use a small local knowledge base from the `docs/` folder before answering a question.

### 4) Vision capabilities

`cv_tools.py` exposes five supported modes:

- `describe`: general image description
- `ocr`: text extraction from the image
- `qa`: answer a question about image content
- `classify`: classify the image into a category
- `chart`: analyze charts and trends

These call the Ollama vision model through `utils.py` using base64-encoded image payloads.

## Tech Stack

- Python 3.10+
- Ollama local inference server
- `requests` for HTTP calls to Ollama
- optional `sentence-transformers` + `faiss` for vector retrieval
- JSON-based logging and structured outputs
- `unittest`-based regression and CV tests

## Repository Structure

```text
Multimodal-Debate-AI-main/
├── main.py                 # CLI entry point and debate orchestration
├── agents.py               # Answer / Critic / Refiner / Judge agents
├── cv_tools.py             # Computer vision modes and image workflows
├── rag.py                  # Lightweight document retrieval
├── scoring.py              # Answer scoring logic
├── utils.py                # Ollama text and vision request utilities
├── create_chart.py         # Chart creation helper
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .env.example            # Example environment configuration
├── .gitignore              # Ignore generated files and local env state
├── docs/                   # Knowledge base documents used by RAG
│   ├── binary_search.txt
│   ├── llm.txt
│   └── recursion.txt
├── logs/                   # JSON debate outputs
├── tests/
│   ├── test_cv.py
│   └── test_regression.py
├── dog.png                 # Example test image
├── chart.png               # Example chart image
├── __pycache__/            # Generated Python cache files
└── .venv/                  # Local virtual environment (ignored)
```

## Requirements

### System requirements

- Python 3.10 or newer
- Ollama installed and running locally
- Access to a text model and vision model in Ollama

### Default models

- Text model: `llama3`
- Vision model: `moondream`

Install the models with:

```powershell
ollama pull llama3
ollama pull moondream
```

If Ollama is not running, start it with:

```powershell
ollama serve
```

## Environment Setup

### 1) Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Configure environment variables

Create a `.env` file based on `.env.example`:

```env
OLLAMA_MODEL=llama3
OLLAMA_VISION_MODEL=moondream
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_REQUEST_TIMEOUT=60
```

The project also reads environment variables directly from the OS at runtime, so setting them in PowerShell is also valid:

```powershell
$env:OLLAMA_MODEL="llama3"
$env:OLLAMA_VISION_MODEL="moondream"
$env:OLLAMA_BASE_URL="http://localhost:11434"
$env:OLLAMA_REQUEST_TIMEOUT="60"
```

## Usage

### Debate mode

Run the interactive prompt:

```powershell
python main.py
```

Or provide a question directly:

```powershell
python main.py "What is binary search?"
```

The program will:

- retrieve relevant content
- generate debate answers
- refine and score them
- print a final winner
- save a detailed JSON log in `logs/`

### CV mode

Describe an image:

```powershell
python main.py cv --image "dog.png" --mode describe
```

Extract text from an image:

```powershell
python main.py cv --image "document.png" --mode ocr
```

Ask a question about an image:

```powershell
python main.py cv --image "kitchen.png" --mode qa --question "What appliances are visible?"
```

Classify an image:

```powershell
python main.py cv --image "dog.png" --mode classify
```

Analyze a chart or graph:

```powershell
python main.py cv --image "chart.png" --mode chart --question "What trend is visible?"
```

## Data Flow and Runtime Behavior

The runtime flow is simple but effective:

```text
User question
    ↓
retrieve_context()
    ↓
answer_agent() x2
    ↓
judge_agent() + critic_agent() + refiner_agent()
    ↓
score_answers()
    ↓
final winner + JSON log
```

A typical log contains:

- timestamp
- original question
- retrieved context
- initial answers
- per-round judge decisions
- feedback from each critic
- refined answers
- scoring dictionary
- final winner

## Performance and Reliability Notes

This repository is intentionally lightweight and designed around local execution, which means the experience depends on:

- local Ollama responsiveness
- model quality and prompt formulation
- image size and compression quality
- retrieval coverage in `docs/`

Important operational characteristics:

- default text generation uses `max_length=400` for answer generation
- default CV responses are capped per mode to keep output compact
- response logs are intentionally stored as structured JSON for debugging and auditing
- error handling falls back to clear `LLM_ERROR:` messages when the model backend fails

## Testing

The project includes automated regression checks for the main debate loop and CV dispatch logic.

Run the full suite:

```powershell
python -m unittest discover -s tests -v
```

Current tests cover:

- debate flow execution
- final winner output generation
- CV mode dispatch
- invalid mode rejection
- LLM backend failure fallback

## Example Output

The program prints logs to the terminal and also writes structured JSON to `logs/` with names such as:

```text
logs/debate_20260520_124021_test_logging.json
```

These logs include the exact question, retrieved context, round-by-round feedback, and final score summary.

## Troubleshooting

### Ollama connection issues

If you see errors like `LLM_ERROR: request_failed` or connection refused:

```powershell
ollama serve
```

Then confirm your local endpoint:

```powershell
curl http://localhost:11434/api/tags
```

### Vision model issues

If the CV modes fail or produce weak output:

- use PNG or JPG files
- avoid extremely large images
- keep the image resolution moderate
- make sure `moondream` is installed

### AVIF compatibility

The project explicitly rejects AVIF files with a clear message because the current pipeline is most reliable on PNG and JPG inputs.

## Limitations

This is a research-friendly prototype, not a production-grade multi-agent system. Some limitations are expected:

- retrieval is lightweight and keyword-based
- judge decisions depend on LLM reasoning quality
- debate rounds are bounded at 3 for stability
- output quality can vary by model and prompt tuning
- the system is local and relies on Ollama uptime

## License

This project is provided for educational, research, and experimentation purposes. It is distributed on an as-is basis.

## Recommended Next Enhancements

- add a persistent vector database for stronger retrieval
- support multiple debate participants beyond 2 answers
- add streaming output for real-time debate visualization
- improve JSON schema validation for judge and critic outputs
- support more image formats and preprocessing
- add CLI flags for model selection and log directory customization

