from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import os

import cv2
import numpy as np
from PIL import Image


@dataclass
class FrameData:
    """Container for a single extracted frame.

    Attributes:
        frame_index: Index of the frame within the video.
        timestamp_s: Timestamp in seconds for the frame.
        image: The frame as a PIL Image in RGB format.
        path: Optional path on disk where the frame image was saved.
    """
    frame_index: int
    timestamp_s: float
    image: Image.Image
    path: Optional[str] = None


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _cv2_to_pil(frame_bgr: np.ndarray, resize_max_dim: Optional[int] = None) -> Image.Image:
    """Convert an OpenCV BGR frame to a PIL RGB image, with optional max-dimension resize."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    if resize_max_dim is not None and max(image.size) > resize_max_dim:
        image.thumbnail((resize_max_dim, resize_max_dim), Image.Resampling.LANCZOS)
    return image


def extract_frames(
    video_path: str,
    interval_seconds: float = 2.0,
    save_dir: Optional[str] = None,
    max_frames: Optional[int] = None,
    resize_max_dim: Optional[int] = 960,
    progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
) -> List[FrameData]:
    """Extract frames from a video at regular intervals.

    Args:
        video_path: Path to the input video file.
        interval_seconds: Interval between frames in seconds (default 2.0s).
        save_dir: If provided, save extracted frames as PNGs to this directory.
        max_frames: Optional upper bound on number of frames to extract.
        resize_max_dim: If set, downscale largest dimension to this size to save memory.

    Returns:
        List of FrameData with timestamps and PIL images.
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be > 0")

    if save_dir is not None:
        _ensure_dir(save_dir)

    # Basic validation
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        frames: List[FrameData] = []

        # Prefer seeking by frame index when FPS is known and positive; otherwise fallback to time-based selection.
        if fps > 0 and total_frames > 0:
            step = max(int(round(fps * interval_seconds)), 1)
            # Estimate number of samples for progress
            total_steps = (total_frames + step - 1) // step
            for i, frame_index in enumerate(range(0, total_frames, step), start=1):
                if max_frames is not None and len(frames) >= max_frames:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = cap.read()
                if not success or frame is None:
                    # Stop early on read failure
                    break
                timestamp_s = frame_index / fps
                image = _cv2_to_pil(frame, resize_max_dim)
                path: Optional[str] = None
                if save_dir is not None:
                    filename = f"frame_{frame_index:07d}_{int(timestamp_s*1000):08d}ms.png"
                    path = os.path.join(save_dir, filename)
                    image.save(path)
                frames.append(FrameData(frame_index=frame_index, timestamp_s=timestamp_s, image=image, path=path))
                if progress_callback is not None:
                    progress_callback(i, total_steps)
        else:
            # Fallback: iterate sequentially and sample by elapsed time using POS_MSEC
            next_time_ms = 0.0
            interval_ms = interval_seconds * 1000.0
            current_index = 0
            produced = 0
            while True:
                if max_frames is not None and len(frames) >= max_frames:
                    break
                success, frame = cap.read()
                if not success or frame is None:
                    break
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC) or (current_index * 1000.0)
                if pos_ms + 1e-3 >= next_time_ms:
                    timestamp_s = pos_ms / 1000.0
                    image = _cv2_to_pil(frame, resize_max_dim)
                    path = None
                    if save_dir is not None:
                        filename = f"frame_{current_index:07d}_{int(timestamp_s*1000):08d}ms.png"
                        path = os.path.join(save_dir, filename)
                        image.save(path)
                    frames.append(FrameData(frame_index=current_index, timestamp_s=timestamp_s, image=image, path=path))
                    produced += 1
                    if progress_callback is not None:
                        progress_callback(produced, None)
                    next_time_ms += interval_ms
                current_index += 1

        if not frames:
            raise RuntimeError("No frames could be extracted. The video may be corrupted or unsupported.")
        return frames
    finally:
        cap.release()


def format_timestamp(seconds: float) -> str:
    """Human-friendly timestamp mm:ss.mmm from seconds."""
    if seconds < 0:
        seconds = 0.0
    whole = int(seconds)
    minutes = whole // 60
    secs = whole % 60
    millis = int(round((seconds - whole) * 1000))
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"
