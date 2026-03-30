# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a Python Streamlit application for video understanding and chat. It extracts frames from uploaded videos, captions them using vision models (BLIP / Qwen2-VL), and allows users to chat about video content using local LLMs (Flan-T5) or Google Gemini API.

### Running the app

```bash
source .venv/bin/activate
export HF_HOME=/workspace/.cache/huggingface
streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

The app serves at `http://localhost:8501`.

### Key caveats

- **HF_HOME must be set before running.** The `.env` file and `app.py` hardcode a macOS SSD path (`/Volumes/PortableSSD/...`) that does not exist on Linux. Always `export HF_HOME=/workspace/.cache/huggingface` before launching, or the app will crash trying to `os.makedirs` on a nonexistent mount.
- **First run downloads ~1GB+ of ML models** (BLIP vision model, Flan-T5 chat model) from Hugging Face Hub. Subsequent runs use the cached models.
- **Gemini chat model requires `GEMINI_API_KEY`** environment variable. Use `google/flan-t5-base` (local) for testing without an API key.
- **No linter or test suite is configured** in this repository. There are no `pytest`, `flake8`, `mypy`, or similar configurations.
- Standard setup/run commands are documented in `README.md`.
