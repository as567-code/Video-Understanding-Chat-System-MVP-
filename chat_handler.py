from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass
class QAResult:
    answer: str
    prompt_tokens: int
    generated_tokens: int


class SimpleVideoChat:
    """Lightweight Q&A over frame descriptions using a small free LLM (Flan-T5).

    The strategy is simple concatenation: join all timestamped captions and ask the
    model to answer the question using that context. This is an MVP approach and can
    be upgraded to retrieval/vector DB later.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: Optional[str] = None,
        max_input_tokens: int = 1500,
        max_new_tokens: int = 128,
        relevance_window: int = 8,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.relevance_window = max(1, int(relevance_window))

    def _select_relevant(self, captions: List[Tuple[str, str]], question: str) -> List[Tuple[str, str]]:
        # Very simple relevance: pick captions with most shared words; fallback to last N
        q_words = {w.lower() for w in question.split() if len(w) > 2}
        scored: List[Tuple[int, Tuple[str, str]]] = []
        for item in captions:
            text_words = {w.lower() for w in item[1].split() if len(w) > 2}
            score = len(q_words & text_words)
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [it for score, it in scored[: self.relevance_window]]
        if not selected:
            selected = captions[-self.relevance_window :]
        return selected

    def _build_context(self, captions: List[Tuple[str, str]], question: str) -> str:
        selected = self._select_relevant(captions, question)
        parts: List[str] = [
            "You are a helpful assistant. Use the following time-aligned descriptions of a video to answer the question."
        ]
        for ts, text in selected:
            parts.append(f"[{ts}] {text}")
        context = "\n".join(parts)
        # Token-safe truncation: ensure prompt token length stays under limit
        while True:
            tokens = self.tokenizer(context + "\nQuestion: " + question, return_tensors="pt").input_ids[0]
            if len(tokens) <= self.max_input_tokens:
                break
            # Drop the earliest entry (keep the most recent/relevant)
            if len(parts) > 2:
                parts.pop(1)
                context = "\n".join(parts)
            else:
                # If still too long, hard truncate tokens
                context = self.tokenizer.decode(tokens[-self.max_input_tokens :], skip_special_tokens=True)
                break
        return context

    @torch.inference_mode()
    def answer(self, captions: List[Tuple[str, str]], question: str) -> QAResult:
        context = self._build_context(captions, question)
        prompt = (
            "Context:\n" + context + "\n\n" + "Question: " + question + "\n" + "Answer concisely:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return QAResult(
            answer=answer,
            prompt_tokens=len(inputs.input_ids[0]),
            generated_tokens=len(output_ids[0]),
        )


