## Video Understanding & Chat System (MVP)

A simple Streamlit app that:
- Uploads a video
- Extracts frames every N seconds (default 2s)
- Sends each frame to a FREE vision model (BLIP-2) for descriptions
- Stores timestamped frame captions
- Lets users chat and ask questions answered by a small FREE LLM using the collected descriptions

### Tech Stack
- Python 3.10+
- Streamlit (web UI)
- OpenCV (video processing)
- Hugging Face Transformers (BLIP-2 + small LLM)
- PyTorch (CPU by default)

### Project Structure
```
video-chat-system/
├── app.py                # Streamlit UI
├── video_processor.py    # Frame extraction (OpenCV)
├── vision_analyzer.py    # BLIP-2 image captioning
├── chat_handler.py       # Q&A over frame descriptions
├── requirements.txt
├── .env                  # Optional environment variables
└── README.md
```

### Setup
1) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or on Windows: .venv\\Scripts\\activate
```

2) Install dependencies (CPU by default)
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3) (Optional) Configure environment
Create `.env` if needed. Example options:
```
# Cache directories (optional)
HF_HOME=.cache/huggingface
TRANSFORMERS_CACHE=.cache/transformers

# Use an alternative HF endpoint if needed
# HF_ENDPOINT=https://huggingface.co
```

### Running the App
```bash
streamlit run app.py
```

Then open the provided local URL in your browser.

### Notes on Models (Free/Open-Source)
- Vision: BLIP-2 (Salesforce) via `transformers` pipeline or model classes
- LLM for chat: start with `google/flan-t5-base` (free, CPU-friendly). You can swap to other small open-source models later.

### Roadmap
- [x] Project scaffold
- [x] Requirements
- [x] Implement frame extraction in `video_processor.py`
- [x] Add BLIP-2 captioning in `vision_analyzer.py`
- [x] Implement chat handler in `chat_handler.py`
- [x] Build Streamlit UI in `app.py`

### License
This project uses only FREE and open-source components. Verify model licenses before distribution.


