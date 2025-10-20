from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import os

from dotenv import load_dotenv
load_dotenv()


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


class GeminiVideoChat:
    """Chat over frame descriptions using Google's Gemini API.

    Keeps the same QAResult interface as SimpleVideoChat for drop-in use.
    Uses conservative, deterministic generation (temperature ~0.2).
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        max_input_tokens: int = 1500,
        max_new_tokens: int = 512,
        relevance_window: int = 8,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini models")

        self.model_name = model_name
        self.max_input_tokens = int(max_input_tokens)
        self.max_new_tokens = int(max_new_tokens)
        self.relevance_window = max(1, int(relevance_window))
        self._api_key = api_key

        # Prefer the new google-genai client but keep legacy client as fallback
        self._genai_new = None
        self._client_new = None
        try:
            from google import genai as genai_new  # type: ignore

            self._genai_new = genai_new
            self._client_new = genai_new.Client(api_key=api_key)
        except Exception:
            self._genai_new = None
            self._client_new = None

        self._genai_old = None
        self._model_old = None
        try:
            import google.generativeai as genai_old  # type: ignore

            self._genai_old = genai_old
            genai_old.configure(api_key=api_key)
            self._model_old = genai_old.GenerativeModel(model_name)
        except Exception:
            self._genai_old = None
            self._model_old = None

        if not self._client_new and not self._model_old:
            raise RuntimeError("Failed to initialize Gemini clients (google-genai or google-generativeai)")

    def _select_relevant(self, captions: List[Tuple[str, str]], question: str) -> List[Tuple[str, str]]:
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
            "You are a helpful assistant. Use the following time-aligned descriptions of a video to answer the question.",
        ]
        for ts, text in selected:
            parts.append(f"[{ts}] {text}")
        return self._trim_to_token_budget(parts, question)

    def _trim_to_token_budget(self, parts: List[str], question: str) -> str:
        """Drop oldest caption lines until we fit the desired token budget."""

        if self.max_input_tokens <= 0:
            return "\n".join(parts)

        prompt_template = "Context:\n{context}\n\nQuestion: {question}\nAnswer concisely:"

        while len(parts) > 2:
            context = "\n".join(parts)
            total_tokens = self._count_tokens(prompt_template.format(context=context, question=question))
            if total_tokens == 0 or total_tokens <= self.max_input_tokens:
                return context
            parts.pop(1)
        return "\n".join(parts)

    def _count_tokens(self, prompt: str) -> int:
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        try:
            if self._client_new:
                token_info = self._client_new.models.count_tokens(
                    model=self.model_name, contents=contents
                )
                return int(getattr(token_info, "total_tokens", 0) or getattr(token_info, "token_count", 0) or 0)
            if self._model_old:
                token_info = self._model_old.count_tokens(prompt)
                return int(getattr(token_info, "total_tokens", 0) or getattr(token_info, "token_count", 0) or 0)
        except Exception:
            pass
        return 0

    def _extract_text_from_parts(self, parts: Optional[List[Any]]) -> str:
        texts: List[str] = []
        if parts:
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    texts.append(text)
        return "\n".join(texts).strip()

    def _generate_with_new(self, prompt: str) -> QAResult:
        assert self._client_new and self._genai_new
        genai_new = self._genai_new
        cfg = genai_new.types.GenerateContentConfig(
            max_output_tokens=self.max_new_tokens,
            temperature=0.2,
            candidate_count=1,
            response_mime_type="text/plain",
        )
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        resp = self._client_new.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=cfg,
        )

        # Extract textual answer
        answer = getattr(resp, "text", None)
        if answer:
            answer = answer.strip()
        if not answer and getattr(resp, "candidates", None):
            collected: List[str] = []
            finish_reasons: List[str] = []
            for cand in resp.candidates or []:
                finish_reason = getattr(cand, "finish_reason", None)
                if finish_reason is not None:
                    finish_reasons.append(str(getattr(finish_reason, "value", finish_reason)))
                content = getattr(cand, "content", None)
                if content is not None:
                    collected_text = self._extract_text_from_parts(getattr(content, "parts", None))
                    if collected_text:
                        collected.append(collected_text)
            if collected:
                answer = "\n".join(collected).strip()
            elif any(fr for fr in finish_reasons if "MAX_TOKENS" in fr):
                raise ValueError(
                    "Gemini truncated the response at the token limit. Increase 'Chat max new tokens' or reduce the question/context size."
                )
        if not answer:
            extra = ""
            if getattr(resp, "prompt_feedback", None):
                feedback = resp.prompt_feedback
                block_reason = getattr(feedback, "block_reason", None)
                if block_reason:
                    extra = f" Block reason: {block_reason}."
            raise ValueError("Gemini returned no text." + extra)

        prompt_tokens = 0
        generated_tokens = 0
        usage = getattr(resp, "usage_metadata", None)
        if usage is not None:
            prompt_tokens = int(
                getattr(usage, "input_tokens", None)
                or getattr(usage, "prompt_token_count", None)
                or 0
            )
            generated_tokens = int(
                getattr(usage, "output_tokens", None)
                or getattr(usage, "candidates_token_count", None)
                or 0
            )
        return QAResult(answer=answer, prompt_tokens=prompt_tokens, generated_tokens=generated_tokens)

    def _generate_with_old(self, prompt: str) -> QAResult:
        assert self._model_old and self._genai_old
        genai_old = self._genai_old
        cfg = genai_old.types.GenerationConfig(
            max_output_tokens=self.max_new_tokens,
            temperature=0.2,
            candidate_count=1,
        )
        resp = self._model_old.generate_content(prompt, generation_config=cfg)
        answer = (getattr(resp, "text", None) or "").strip()
        if not answer:
            raise ValueError("Gemini (legacy client) returned no text")

        prompt_tokens = 0
        generated_tokens = 0
        usage = getattr(resp, "usage_metadata", None)
        if usage is not None:
            prompt_tokens = int(
                getattr(usage, "input_token_count", None)
                or getattr(usage, "prompt_token_count", None)
                or 0
            )
            generated_tokens = int(
                getattr(usage, "output_token_count", None)
                or getattr(usage, "candidates_token_count", None)
                or 0
            )
        return QAResult(answer=answer, prompt_tokens=prompt_tokens, generated_tokens=generated_tokens)

    def answer(self, captions: List[Tuple[str, str]], question: str) -> QAResult:
        context = self._build_context(captions, question)
        prompt = (
            "Context:\n" + context + "\n\n" + "Question: " + question + "\n" + "Answer concisely:"
        )

        errors: List[str] = []

        if self._client_new:
            try:
                return self._generate_with_new(prompt)
            except Exception as exc:
                errors.append(f"new-client error: {exc}")

        if self._model_old:
            try:
                return self._generate_with_old(prompt)
            except Exception as exc:
                errors.append(f"legacy-client error: {exc}")

        joined_errors = "; ".join(errors) if errors else "Unknown error"
        return QAResult(
            answer=f"Gemini API error: {joined_errors}",
            prompt_tokens=0,
            generated_tokens=0,
        )
