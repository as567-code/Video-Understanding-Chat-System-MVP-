from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Iterable, Sequence, Any
import io

from PIL import Image
import torch
import requests
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
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


class QwenVisionCaptioner:
    """Image captioning using Qwen2-VL vision-language model.

    Uses Qwen2-VL-2B-Instruct for high-quality image understanding and captioning.
    Supports dynamic resolution and advanced visual reasoning.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: Optional[str] = None,
        max_new_tokens: int = 128,
        prompt: str = "Describe what's happening in this image in detail.",
        batch_size: int = 1,
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

        print(f"Loading Qwen2-VL model {model_name} on {device}...")

        # Load Qwen2-VL model and processor
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        # Use appropriate dtype based on device
        if device in ("cuda", "mps"):
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
            
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device if device != "mps" else None,
            trust_remote_code=True,
        )
        
        # For MPS, manually move to device
        if device == "mps":
            self.model.to(device)
            
        self.model.eval()
        print("✅ Qwen2-VL model loaded successfully!")

    def _image_to_messages(self, image: Image.Image, prompt: str) -> List[dict]:
        """Convert PIL image to Qwen's message format."""
        # Save image to bytes buffer
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        
        # Create message in Qwen format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return messages

    @torch.inference_mode()
    def caption_image(self, image: Image.Image) -> str:
        """Generate caption for a single image."""
        try:
            # Import qwen_vl_utils here to avoid import errors if not installed
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError(
                "qwen-vl-utils is required for Qwen2-VL. Install with: pip install qwen-vl-utils"
            )
        
        messages = self._image_to_messages(image, self.prompt)
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision information
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device
        inputs = inputs.to(self.device)
        
        # Generate
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )
        
        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        return output_text.strip()

    @torch.inference_mode()
    def caption_frames(
        self,
        frames: Sequence[Image.Image],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[str]:
        """Generate captions for multiple frames.
        
        Note: Qwen2-VL processes images one at a time due to its architecture.
        Batching is less efficient for vision-language models compared to pure captioning models.
        """
        captions: List[str] = []
        total = len(frames)
        
        for idx, frame in enumerate(frames):
            caption = self.caption_image(frame)
            captions.append(caption)
            
            if progress_callback is not None:
                progress_callback(idx + 1, total)
        
        return captions


def make_captions_for_timestamps(
    captioner: Any,
    images_with_ts: Iterable[tuple[float, Image.Image]],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Caption]:
    pairs = list(images_with_ts)
    timestamps = [ts for ts, _ in pairs]
    images = [img for _, img in pairs]
    texts = captioner.caption_frames(images, progress_callback=progress_callback)
    return [Caption(timestamp_s=ts, text=txt) for ts, txt in zip(timestamps, texts)]


