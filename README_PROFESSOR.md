# Video Understanding & Chat System with Qwen2-VL

## 🎓 HCI Project - Video Analysis with State-of-the-Art AI

**Student Project Submission**  
**Course**: Human-Computer Interaction  
**Date**: October 2025

---

## 📋 Project Overview

This project implements an intelligent video understanding system that:
1. **Extracts frames** from uploaded videos at configurable intervals
2. **Generates captions** using advanced vision models
3. **Enables natural language Q&A** about video content using AI chat models
4. **Integrates cutting-edge models** including Qwen2-VL-2B-Instruct and Gemini 2.5 Flash

### Key Features ✨

- **Multiple Vision Models**:
  - BLIP Image Captioning (Salesforce) - Lightweight, fast
  - **Qwen2-VL-2B-Instruct** - State-of-the-art 2B parameter vision-language model

- **Multiple Chat Models**:
  - Flan-T5 (Google) - Free, local, no API required
  - **Gemini 2.5 Flash** - Cloud-based, powerful conversational AI

- **Optimized Storage**: All models (~4GB+) cached on external SSD
- **Interactive UI**: Streamlit-based web interface
- **Export Capabilities**: Download captions (CSV/JSON) and chat history

---

## 🏗️ Technical Architecture

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | Streamlit 1.37+ | Interactive UI |
| **Video Processing** | OpenCV 4.9+ | Frame extraction |
| **Vision Models** | Transformers 4.44+ | Image captioning |
| **Deep Learning** | PyTorch + MPS | Model inference on M3 Mac |
| **Chat AI** | Gemini 2.5 Flash | Question answering |
| **Storage** | External SSD | Model caching (~8GB) |

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Web UI                      │
├─────────────────────────────────────────────────────────┤
│  Video Upload → Frame Extraction → Vision Model → Chat  │
├─────────────────────────────────────────────────────────┤
│  Components:                                             │
│  • video_processor.py  - OpenCV frame extraction        │
│  • vision_analyzer.py  - BLIP / Qwen2-VL captioning    │
│  • chat_handler.py     - Flan-T5 / Gemini Q&A          │
│  • app.py             - Main Streamlit interface        │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Setup & Installation

### Prerequisites

- macOS (tested on M3 MacBook)
- Python 3.10+
- External SSD (for model storage)
- Internet connection (first run only)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd video-chat-system
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment** (Optional - for Gemini chat):
   ```bash
   # Copy example and add your API key
   cp .env.example .env
   # Edit .env and add: GEMINI_API_KEY=your_key_here
   # Get free API key at: https://aistudio.google.com/apikey
   ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

6. **Open in browser**: http://localhost:8501

---

## 🎯 Usage Guide

### 1. Configure Settings (Sidebar)

**Frame Extraction**:
- Interval: 0.5-10 seconds (default: 2s)
- Max frames: Limit processing (0 = no limit)
- Max dimension: Resize for memory efficiency

**Vision Model Selection**:
- `Salesforce/blip-image-captioning-base` - Fast, lightweight
- `Qwen/Qwen2-VL-2B-Instruct` - **Recommended**, best quality

**Chat Model Selection**:
- `google/flan-t5-base` - Free, no API key needed
- `gemini-2.5-flash` - Best results (requires API key)

### 2. Upload Video

- Supported formats: MP4, MOV, AVI, MKV
- Recommended: 30 seconds - 2 minutes for testing
- First run downloads models (~4-8GB to SSD, 5-15 minutes)

### 3. Review Captions

- Automatically generated for each frame
- Timestamped (mm:ss.mmm format)
- Download as CSV or JSON

### 4. Ask Questions

- Natural language queries about video content
- Examples:
  - "What happens at 0:30?"
  - "Describe the main objects in the video"
  - "Summarize the video content"

---

## 🔬 Technical Innovations

### 1. Qwen2-VL Integration

**Implementation Highlights**:
- Custom `QwenVisionCaptioner` class in `vision_analyzer.py`
- Message-based format with chat templates
- Dynamic resolution support
- Float16 optimization for M3 Mac MPS acceleration

**Key Code** (lines 99-246 in `vision_analyzer.py`):
```python
class QwenVisionCaptioner:
    def __init__(self, model_name, device, ...):
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, trust_remote_code=True
        )
```

### 2. External SSD Caching

**Storage Optimization**:
- HuggingFace cache: `/Volumes/PortableSSD/huggingface_cache/`
- Models: ~4.1 GB (Qwen2-VL)
- Total cache: ~8.2 GB
- Prevents MacBook internal drive congestion

**Configuration** (lines 5-11 in `app.py`):
```python
if not os.environ.get('HF_HOME'):
    default_cache = "/Volumes/PortableSSD/huggingface_cache"
    os.makedirs(default_cache, exist_ok=True)
    os.environ['HF_HOME'] = default_cache
```

### 3. Gemini API Integration

**Dual SDK Support**:
- `google-genai` (new SDK)
- `google-generativeai` (legacy SDK)
- Automatic fallback mechanism
- Token usage tracking

**Implementation** (lines 114-345 in `chat_handler.py`):
- Conservative generation (temperature=0.2)
- Token budget management
- Relevance-based caption selection

### 4. MPS Acceleration

**Apple Silicon Optimization**:
- Automatic device detection (CUDA/MPS/CPU)
- Float16 for GPU/MPS, Float32 for CPU
- Efficient memory management

---

## 📊 Performance Benchmarks

### Model Performance (M3 MacBook)

| Model | Size | First Load | Inference/Frame | Quality |
|-------|------|-----------|----------------|---------|
| BLIP Base | ~400MB | 30s | ~0.5s | Good |
| Qwen2-VL-2B | ~4.1GB | 5-10min | ~2-3s | Excellent |

### System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space (for models)
- **GPU**: M1/M2/M3 (MPS) or CUDA GPU recommended
- **Network**: Required for first-time model download only

---

## 🗂️ Project Structure

```
video-chat-system/
├── app.py                    # Main Streamlit application
├── chat_handler.py           # Q&A models (Flan-T5, Gemini)
├── vision_analyzer.py        # Vision models (BLIP, Qwen2-VL)
├── video_processor.py        # Frame extraction (OpenCV)
├── requirements.txt          # Python dependencies
├── README.md                 # Original project README
├── QUICK_START.md           # Quick start guide
├── .env.example             # Environment template
├── .gitignore               # Git ignore rules
└── README_PROFESSOR.md      # This file
```

---

## 🔑 Key Dependencies

```
streamlit>=1.37.0              # Web UI
transformers>=4.44.0           # HuggingFace models
torch, torchvision            # Deep learning
opencv-python>=4.9.0.80       # Video processing
qwen-vl-utils                 # Qwen2-VL utilities
google-genai>=0.5.0           # Gemini SDK (new)
google-generativeai>=0.8.3    # Gemini SDK (legacy)
```

---

## 🎓 Learning Outcomes

This project demonstrates:

1. **AI Integration**: Successfully integrated cutting-edge vision-language models
2. **System Optimization**: Efficient model caching and memory management
3. **UI/UX Design**: Intuitive Streamlit interface with real-time feedback
4. **API Integration**: Multi-provider AI service integration (HuggingFace + Google)
5. **Performance Tuning**: Hardware-specific optimizations (MPS acceleration)
6. **Software Engineering**: Modular design, error handling, documentation

---

## 🐛 Known Limitations

1. **First Run**: Initial model download takes 5-15 minutes
2. **Processing Speed**: Qwen2-VL is slower than BLIP (~2-3s vs 0.5s per frame)
3. **Memory Usage**: Requires ~8GB RAM for Qwen2-VL model
4. **API Costs**: Gemini has free tier limits (15 requests/minute)
5. **External SSD**: Must stay connected during operation

---

## 🔮 Future Enhancements

- [ ] Add video timeline visualization
- [ ] Implement scene detection and segmentation
- [ ] Support for live video streams
- [ ] Multi-language caption support
- [ ] Object tracking across frames
- [ ] Export to video with embedded captions
- [ ] Batch processing for multiple videos

---

## 📚 References

1. **Qwen2-VL**: [Hugging Face Model Card](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
2. **BLIP**: Salesforce BLIP Image Captioning
3. **Gemini API**: [Google AI Studio](https://aistudio.google.com/)
4. **Streamlit**: [Documentation](https://docs.streamlit.io/)
5. **PyTorch**: [MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)

---

## 📞 Support

For questions or issues:
- Check `QUICK_START.md` for common troubleshooting
- Review `README.md` for original documentation
- Ensure `.env` file is configured correctly
- Verify SSD is connected and mounted

---

## ✅ Testing Checklist

- [x] Frame extraction with various intervals
- [x] BLIP model captioning
- [x] Qwen2-VL model captioning
- [x] Flan-T5 local chat
- [x] Gemini API chat
- [x] CSV/JSON export
- [x] Progress tracking
- [x] Error handling
- [x] SSD caching
- [x] MPS acceleration

---

**Project Status**: ✅ Complete and Functional  
**Last Updated**: October 20, 2025  
**Models Integrated**: BLIP, Qwen2-VL-2B-Instruct, Flan-T5, Gemini 2.5 Flash

