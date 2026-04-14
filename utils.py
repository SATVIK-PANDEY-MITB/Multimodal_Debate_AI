import requests

OLLAMA_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434/api/generate"


def call_llm(prompt, max_length=220, temperature=0.2, timeout=60):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": max_length,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.RequestException as e:
        return f"LLM_ERROR: request_failed: {e}"
    except ValueError:
        return "LLM_ERROR: invalid_json_response"
   
