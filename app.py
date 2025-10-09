from __future__ import annotations

import os

# Set HF_HOME environment variable BEFORE any imports
# This ensures models download to SSD cache, not MacBook internal drive
os.environ['HF_HOME'] = '/Volumes/PortableSSD/video-chat-system/.cache/huggingface'

# Load other environment variables
from dotenv import load_dotenv
load_dotenv()  # Must be loaded before any HuggingFace imports

import io
import os
from typing import List, Tuple, Dict, Any

import streamlit as st
from PIL import Image
import hashlib
import json
import csv

from video_processor import extract_frames, format_timestamp, FrameData
from vision_analyzer import SimpleVisionCaptioner, Caption, make_captions_for_timestamps
from chat_handler import SimpleVideoChat


st.set_page_config(page_title="Video Understanding & Chat", layout="wide")
st.title("Video Understanding & Chat (MVP)")
st.write("Upload a video, extract frames, caption them with a free vision model, and ask questions.")


@st.cache_resource(show_spinner=False)
def get_captioner(
    model_name: str,
    max_new_tokens: int,
    prompt: str,
    batch_size: int,
    device_choice: str,
) -> SimpleVisionCaptioner:
    device = None if device_choice == "auto" else device_choice
    return SimpleVisionCaptioner(
        model_name=model_name,
        max_new_tokens=int(max_new_tokens),
        prompt=prompt,
        batch_size=int(batch_size),
        device=device,
    )


@st.cache_resource(show_spinner=False)
def get_chat_model(
    model_name: str,
    max_new_tokens: int,
    max_input_tokens: int,
    relevance_window: int,
    device_choice: str,
) -> SimpleVideoChat:
    device = None if device_choice == "auto" else device_choice
    return SimpleVideoChat(
        model_name=model_name,
        max_new_tokens=int(max_new_tokens),
        max_input_tokens=int(max_input_tokens),
        relevance_window=int(relevance_window),
        device=device,
    )


def main() -> None:
    # Initialize session state
    if "captions_cache" not in st.session_state:
        st.session_state["captions_cache"] = {}
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    with st.sidebar:
        st.header("Configuration")
        interval_s = st.number_input("Frame interval (seconds)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
        max_frames = st.number_input("Max frames (0 = no limit)", min_value=0, max_value=500, value=0, step=10)
        resize_max_dim = st.number_input("Max frame dimension", min_value=256, max_value=2048, value=960, step=64)

        st.subheader("Vision model")
        vision_model = st.selectbox(
            "Vision model",
            [
                "Salesforce/blip-image-captioning-base",
            ],
            index=0,
        )
        vision_prompt = st.text_input(
            "Vision prompt",
            value="Describe what's happening in this image in detail.",
        )
        vision_batch = st.number_input("Vision batch size", min_value=1, max_value=16, value=4, step=1)
        vision_max_new_tokens = st.number_input("Vision max new tokens", min_value=8, max_value=128, value=64, step=8)

        st.subheader("Chat model")
        chat_model = st.selectbox(
            "LLM",
            [
                "google/flan-t5-base",
                "google/flan-t5-small",
            ],
            index=0,
        )
        chat_max_input_tokens = st.number_input("Chat max input tokens", min_value=256, max_value=3000, value=1500, step=50)
        chat_max_new_tokens = st.number_input("Chat max new tokens", min_value=16, max_value=256, value=128, step=16)
        relevance_window = st.number_input("Relevance window (captions)", min_value=1, max_value=32, value=8, step=1)

        st.subheader("Runtime")
        device_choice = st.selectbox("Device", ["auto", "cpu", "mps", "cuda"], index=0)
        preview_count = st.number_input("Preview frames to display", min_value=0, max_value=20, value=10, step=1)
        use_cache = st.checkbox("Use caption cache for same video/config", value=True)
        delete_after = st.checkbox("Delete uploaded file after processing", value=True)

    uploaded = st.file_uploader("Upload a video (mp4, mov, avi)", type=["mp4", "mov", "avi", "mkv"]) 

    if uploaded is not None:
        # Compute a stable cache key using video bytes and config
        try:
            video_bytes = uploaded.getbuffer()
            digest = hashlib.sha256(video_bytes).hexdigest()
        except Exception:
            digest = "unknown"

        cache_key = "|".join(
            [
                digest,
                vision_model,
                str(interval_s),
                str(resize_max_dim),
                vision_prompt,
                str(vision_batch),
                str(vision_max_new_tokens),
            ]
        )

        # Save to a temporary file so OpenCV can read it
        tmp_dir = st.session_state.get("tmp_dir", "tmp_uploads")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_video_path = os.path.join(tmp_dir, uploaded.name)
        try:
            with open(tmp_video_path, "wb") as f:
                f.write(video_bytes)
        except Exception as e:
            st.error(f"Failed to save uploaded file: {e}")
            return

        st.success("Video uploaded. Extracting frames...")
        extract_progress = st.progress(0)

        def update_extract_progress(done: int, total: Any) -> None:
            try:
                if total is None or not total:
                    denom = int(max_frames) if int(max_frames) > 0 else max(done, 1)
                    percent = int(min(100, done / denom * 100))
                else:
                    percent = int(min(100, done / int(total) * 100))
                extract_progress.progress(percent)
            except Exception:
                pass

        try:
            frames: List[FrameData] = extract_frames(
                tmp_video_path,
                interval_seconds=float(interval_s),
                save_dir=None,
                max_frames=(int(max_frames) if int(max_frames) > 0 else None),
                resize_max_dim=int(resize_max_dim),
                progress_callback=update_extract_progress,
            )
        except Exception as e:
            st.error(f"Failed to extract frames: {e}")
            if delete_after:
                try:
                    os.remove(tmp_video_path)
                except Exception:
                    pass
            return

        if len(frames) == 0:
            st.error("No frames extracted. Try a different video or interval.")
            return

        st.write(f"Extracted {len(frames)} frames.")

        # Display a small gallery of frames
        preview_n = int(preview_count)
        if preview_n > 0:
            cols = st.columns(5)
            for idx, frame_data in enumerate(frames[:preview_n]):
                col = cols[idx % 5]
                with col:
                    st.image(frame_data.image, caption=format_timestamp(frame_data.timestamp_s), use_column_width=True)

        # Caption frames
        captions: List[Caption]
        if use_cache and cache_key in st.session_state["captions_cache"]:
            st.info("Loaded captions from cache.")
            stored = st.session_state["captions_cache"][cache_key]
            captions = [Caption(timestamp_s=item["timestamp_s"], text=item["text"]) for item in stored]
        else:
            with st.spinner("Captioning frames with BLIP-2 (first run may download models)..."):
                caption_progress = st.progress(0)

                def update_caption_progress(done: int, total: int) -> None:
                    try:
                        percent = int(min(100, done / max(1, int(total)) * 100))
                        caption_progress.progress(percent)
                    except Exception:
                        pass

                try:
                    captioner = get_captioner(
                        model_name=vision_model,
                        max_new_tokens=int(vision_max_new_tokens),
                        prompt=vision_prompt,
                        batch_size=int(vision_batch),
                        device_choice=device_choice,
                    )
                    images_with_ts = [(fd.timestamp_s, fd.image) for fd in frames]
                    captions = make_captions_for_timestamps(
                        captioner,
                        images_with_ts,
                        progress_callback=update_caption_progress,
                    )
                except Exception as e:
                    st.error(f"Failed to caption frames: {e}")
                    if delete_after:
                        try:
                            os.remove(tmp_video_path)
                        except Exception:
                            pass
                    return
            # Save in cache
            st.session_state["captions_cache"][cache_key] = [
                {"timestamp_s": c.timestamp_s, "text": c.text} for c in captions
            ]

        st.success("Captions generated.")
        for c in captions[: min(10, len(captions))]:
            st.write(f"[{format_timestamp(c.timestamp_s)}] {c.text}")

        # Prepare chat context
        time_str_and_caps: List[Tuple[str, str]] = [
            (format_timestamp(c.timestamp_s), c.text) for c in captions
        ]

        st.divider()
        st.subheader("Chat about the video")
        # Download buttons
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["timestamp", "caption"])
        for c in captions:
            writer.writerow([format_timestamp(c.timestamp_s), c.text])
        st.download_button(
            "Download captions (CSV)", data=csv_buf.getvalue(), file_name="captions.csv", mime="text/csv"
        )
        json_data = [
            {"timestamp_s": c.timestamp_s, "timestamp": format_timestamp(c.timestamp_s), "text": c.text}
            for c in captions
        ]
        st.download_button(
            "Download captions (JSON)",
            data=json.dumps(json_data, ensure_ascii=False, indent=2),
            file_name="captions.json",
            mime="application/json",
        )

        # Chat history
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_question = st.chat_input("Ask a question about the video")
        if user_question:
            st.session_state["messages"].append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        chat = get_chat_model(
                            model_name=chat_model,
                            max_new_tokens=int(chat_max_new_tokens),
                            max_input_tokens=int(chat_max_input_tokens),
                            relevance_window=int(relevance_window),
                            device_choice=device_choice,
                        )
                        result = chat.answer(time_str_and_caps, user_question)
                        st.markdown(result.answer)
                        st.session_state["messages"].append({"role": "assistant", "content": result.answer})
                    except Exception as e:
                        st.error(f"Chat failed: {e}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear chat"):
                st.session_state["messages"] = []
        with col2:
            if st.session_state["messages"]:
                st.download_button(
                    "Download chat (JSON)",
                    data=json.dumps(st.session_state["messages"], ensure_ascii=False, indent=2),
                    file_name="chat_history.json",
                    mime="application/json",
                )

        # Cleanup temporary file
        if delete_after:
            try:
                os.remove(tmp_video_path)
            except Exception:
                pass


if __name__ == "__main__":
    main()


