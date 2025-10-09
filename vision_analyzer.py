from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Iterable, Sequence

from PIL import Image
import torch
import requests
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
)


@dataclass
class Caption:
    timestamp_s: float
    text: str


class SimpleVisionCaptioner:
    """Lightweight image captioning using smaller BLIP model.

    Uses Salesforce BLIP (not BLIP-2) for fast, lightweight image captioning.
    Much smaller than BLIP-2 and won't cause system hangs.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: Optional[str] = None,
        max_new_tokens: int = 64,
        prompt: str = "Describe what's happening in this image in detail.",
        batch_size: int = 4,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self.batch_size = max(1, int(batch_size))

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        print(f"Loading model {model_name} on {device}...")

        # Use smaller BLIP model instead of BLIP-2
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16 if device in ("cuda", "mps") else torch.float32
        )
        self.model.to(device)
        self.model.eval()
        print("✅ Model loaded successfully!")

    @torch.inference_mode()
    def caption_image(self, image: Image.Image) -> str:
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()

    @torch.inference_mode()
    def caption_frames(
        self,
        frames: Sequence[Image.Image],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[str]:
        captions: List[str] = []
        total = len(frames)
        processed = 0
        # Batch generation for performance
        for start in range(0, total, self.batch_size):
            batch = frames[start : start + self.batch_size]
            inputs = self.processor(list(batch), return_tensors="pt", padding=True).to(self.device)
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
            texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            captions.extend([t.strip() for t in texts])
            processed += len(batch)
            if progress_callback is not None:
                progress_callback(processed, total)
        return captions


def make_captions_for_timestamps(
    captioner: SimpleVisionCaptioner,
    images_with_ts: Iterable[tuple[float, Image.Image]],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Caption]:
    pairs = list(images_with_ts)
    timestamps = [ts for ts, _ in pairs]
    images = [img for _, img in pairs]
    texts = captioner.caption_frames(images, progress_callback=progress_callback)
    return [Caption(timestamp_s=ts, text=txt) for ts, txt in zip(timestamps, texts)]


