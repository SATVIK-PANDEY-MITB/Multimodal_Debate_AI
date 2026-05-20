# Multimodal Debate AI

A local-first multi-agent debate system with optional computer vision tools, powered by Ollama.

## What It Does

- Runs a 3-round debate pipeline:
  - Answer Agent (logical + creative answers)
  - Critic Agent (quality feedback)
  - Refiner Agent (improves both answers)
  - Judge Agent (selects winner + reasoning)
- Supports optional RAG from `docs/` using lightweight keyword retrieval.
- Logs each debate run as structured JSON in `logs/`.
- Provides 5 CV modes:
  - `describe` (scene description)
  - `ocr` (text extraction)
  - `qa` (ask questions about an image)
  - `classify` (image type/category)
  - `chart` (chart/graph trend analysis)
- Includes regression tests for debate + CV routing.

## Requirements

- Python 3.10+
- Ollama running locally
- Models:
  - text: `llama3` (default)
  - vision: `moondream` (default)

Install models if needed:

```powershell
ollama pull llama3
ollama pull moondream
```

## Setup

```powershell
cd "c:\Users\ayush\Downloads\MULTI AGENT DEBATE AI"
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional environment config (`.env`):

```env
OLLAMA_MODEL=llama3
OLLAMA_VISION_MODEL=moondream
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_REQUEST_TIMEOUT=60
```

## Usage

### Debate Mode

Interactive:

```powershell
python main.py
```

One-shot:

```powershell
python main.py "What is binary search?"
```

Output includes final winner and a JSON log file under `logs/`.

### CV Mode

Describe:

```powershell
python main.py cv --image "dog.png" --mode describe
```

OCR:

```powershell
python main.py cv --image "document.png" --mode ocr
```

Image QA:

```powershell
python main.py cv --image "kitchen.png" --mode qa --question "What appliances are visible?"
```

Classify:

```powershell
python main.py cv --image "dog.png" --mode classify
```

Chart analysis:

```powershell
python main.py cv --image "sales_chart.png" --mode chart --question "What trend is shown?"
```

## Testing

Run all tests:

```powershell
python -m unittest discover -s tests -v
```

Covers:

- debate flow
- CV mode dispatch
- invalid mode handling
- backend error fallback

## Project Structure

```text
main.py          # Debate/CV entrypoint
agents.py        # Answer/Critic/Refiner/Judge agents
cv_tools.py      # CV features (describe/ocr/qa/classify/chart)
rag.py           # Lightweight retrieval from docs/
scoring.py       # Scoring utilities
utils.py         # Ollama text + vision calls
tests/           # Regression and CV tests
docs/            # RAG source docs
logs/            # Debate run JSON logs
```

## Troubleshooting

- `LLM_ERROR: Connection refused`
  - Start Ollama: `ollama serve`
- Vision output is weak or empty
  - Try PNG/JPG input and smaller images
  - Ensure `moondream` is installed
- AVIF not supported in your environment
  - Convert image to PNG/JPG first

## License

Provided as-is for educational and research use.
