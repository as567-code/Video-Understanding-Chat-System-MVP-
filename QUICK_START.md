# 🚀 Video Chat System - Quick Start Guide

## ✅ Your Project is Now on SSD!

### 📁 Project Location
/Volumes/PortableSSD/video-chat-system/

### 💾 Models Will Be Stored
/Volumes/PortableSSD/huggingface_cache/
(This saves ~4GB on your MacBook!)

---

## 🎯 How to Run Your Project

### Step 1: Open Terminal

### Step 2: Navigate to Project
```bash
cd /Volumes/PortableSSD/video-chat-system
```

### Step 3: Install Dependencies (First Time Only)
```bash
pip install -r requirements.txt
```

### Step 4: Run the App
```bash
streamlit run app.py
```

### Step 5: Open in Browser
The app will open automatically at: http://localhost:8501

---

## 📊 What Happens on First Run

1. **Upload a video** (mp4, mov, avi)
2. **Wait 5-15 minutes** - Models download to SSD:
   - BLIP-2 vision model (~3GB)
   - Flan-T5 chat model (~1GB)
3. **After first run**: Instant! Models are cached

---

## 🔧 Configuration Options

In the sidebar you can adjust:
- **Frame interval**: How often to extract frames (default 2 seconds)
- **Max frames**: Limit number of frames (0 = no limit)
- **Max dimension**: Resize frames to save memory (default 960px)

---

## 💡 Tips for Best Results

1. **Use short videos first** (30 sec - 2 min) to test
2. **Keep SSD connected** when running the app
3. **First video is slow** (downloading models)
4. **Subsequent videos are fast** (models cached)

---

## 🎓 For Your Demo/Presentation

**Recommended video types:**
- ✅ Cooking tutorials
- ✅ Lecture clips
- ✅ How-to videos
- ✅ Product demos

**Questions to demonstrate:**
- "What happens at 0:30?"
- "Describe the main steps shown"
- "What objects appear in the video?"
- "Summarize what happened"

---

## ⚠️ Important Notes

1. **SSD must stay connected** - If disconnected, app can't find models
2. **Models take space** - ~4GB on SSD
3. **M3 acceleration enabled** - Uses your Mac's GPU for speed!

---

## 🆘 Troubleshooting

**App won't start?**
- Make sure SSD is connected
- Check you're in correct directory

**Models not downloading?**
- Check internet connection
- May take 5-15 minutes on first run

**Out of memory?**
- Reduce max_frames in sidebar
- Use smaller frame dimensions

---

## 📞 Need Help?

Check the README.md for more details!

---

Generated on: $(date)
