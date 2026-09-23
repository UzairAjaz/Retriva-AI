"""
LLM backend abstraction.

Two interchangeable implementations:

* **LocalLLM**  -- runs the model in-process with transformers (default, offline,
  what the CPU/GPU auto-detection feeds into).
* **RemoteLLM** -- talks to an OpenAI-compatible chat endpoint over HTTP
  (vLLM, TGI, Ollama, Azure OpenAI, Bedrock's OpenAI-compatible gateway, ...).

The RAG engine only calls ``generate()`` / ``stream()`` so scaling inference
independently from the API is a configuration change, not a code change.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Dict, Iterator, List

logger = logging.getLogger("retriva.llm")


class _EventStoppingCriteria:
    """Stops generation as soon as the event is set (client pressed Stop).

    Subclasses ``transformers.StoppingCriteria`` at runtime so importing
    transformers is not required for the remote backend.
    """

    def __new__(cls, event: threading.Event):
        from transformers import StoppingCriteria

        class _Impl(StoppingCriteria):
            def __init__(self, ev):
                self._ev = ev

            def __call__(self, input_ids, scores, **kwargs):  # noqa: ARG002
                return self._ev.is_set()

        return _Impl(event)


class BaseLLM:
    name = "base"

    def generate(
        self, messages: List[Dict], temperature: float = 0.1, max_new_tokens: int | None = None
    ) -> str:
        raise NotImplementedError

    def stream(self, messages: List[Dict], temperature: float = 0.1) -> Iterator[str]:
        raise NotImplementedError


class LocalLLM(BaseLLM):
    """In-process transformers text-generation pipeline."""

    name = "local"

    def __init__(self, config, device_arg, dtype):
        from transformers import AutoTokenizer, pipeline

        self.config = config
        self.device_arg = device_arg
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.LLM_MODEL_NAME, local_files_only=config.HF_LOCAL_ONLY
        )
        self.pipeline = pipeline(
            "text-generation",
            model=config.LLM_MODEL_NAME,
            dtype=dtype,
            device=device_arg,
        )
        self._lock = threading.Lock()

    # -- prompt / sampling -------------------------------------------------
    def _apply_template(self, messages: List[Dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _generation_kwargs(self, temperature: float, max_new_tokens: int) -> Dict:
        kwargs: Dict = {"max_new_tokens": max_new_tokens}
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if pad_id is not None:
            kwargs["pad_token_id"] = pad_id
        if temperature and temperature > 0.25:
            kwargs.update(
                do_sample=True,
                temperature=float(temperature),
                top_p=self.config.TOP_P,
                top_k=self.config.TOP_K,
            )
        else:
            # Low temperature (the default) uses greedy decoding: more stable and
            # more accurate for factual/finance answers.
            kwargs.update(do_sample=False)
        return kwargs

    # -- API ---------------------------------------------------------------
    def generate(self, messages, temperature=0.1, max_new_tokens=None) -> str:
        prompt = self._apply_template(messages)
        with self._lock:
            output = self.pipeline(
                prompt,
                return_full_text=False,
                **self._generation_kwargs(
                    temperature, max_new_tokens or self.config.MAX_NEW_TOKENS
                ),
            )
        return output[0]["generated_text"]

    def stream(self, messages, temperature=0.1) -> Iterator[str]:
        from transformers import StoppingCriteriaList, TextIteratorStreamer

        prompt = self._apply_template(messages)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        stop_event = threading.Event()
        kwargs = self._generation_kwargs(temperature, self.config.MAX_NEW_TOKENS)
        kwargs["text_inputs"] = prompt
        kwargs["streamer"] = streamer
        kwargs["stopping_criteria"] = StoppingCriteriaList([_EventStoppingCriteria(stop_event)])

        errors: Dict[str, BaseException] = {}

        def _run() -> None:
            try:
                with self._lock:
                    self.pipeline(**kwargs)
            except BaseException as exc:  # noqa: BLE001
                errors["error"] = exc
                logger.error("Streaming generation failed: %s", exc)
            finally:
                streamer.end()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        try:
            for piece in streamer:
                if piece:
                    yield piece
        finally:
            # Client disconnected / pressed Stop: stop at the next decode step.
            stop_event.set()
            streamer.end()
            thread.join(timeout=5)
        if "error" in errors:
            yield "\n\n*(generation interrupted)*"


class RemoteLLM(BaseLLM):
    """OpenAI-compatible chat completions over HTTP (vLLM/TGI/Ollama/etc.)."""

    name = "remote"

    def __init__(self, base_url: str, api_key: str = "", model: str = "", timeout: int = 300):
        if not base_url:
            raise ValueError("LLM_BASE_URL is required when LLM_BACKEND=remote")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, messages, temperature, max_tokens, stream):
        return {
            "model": self.model,
            "messages": messages,
            "temperature": max(float(temperature or 0.0), 0.0),
            "max_tokens": max_tokens,
            "stream": stream,
        }

    def generate(self, messages, temperature=0.1, max_new_tokens=None) -> str:
        import httpx

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            json=self._payload(
                messages, temperature, max_new_tokens or 512, False
            ),
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def stream(self, messages, temperature=0.1) -> Iterator[str]:
        import httpx

        with httpx.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=self._payload(messages, temperature, 512, True),
            headers=self._headers(),
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            for raw in response.iter_lines():
                if not raw:
                    continue
                line = raw[6:] if raw.startswith("data: ") else raw
                if line.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices") or [{}]
                content = (choices[0].get("delta") or {}).get("content")
                if content:
                    yield content


def build_llm(config, device_arg=None, dtype=None) -> BaseLLM:
    """Factory driven by ``LLM_BACKEND``."""
    backend = (config.LLM_BACKEND or "local").strip().lower()
    if backend == "remote":
        logger.info(
            "LLM backend: remote (%s, model=%s)", config.LLM_BASE_URL, config.LLM_MODEL_NAME
        )
        return RemoteLLM(
            config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            model=config.LLM_MODEL_NAME,
            timeout=config.LLM_REQUEST_TIMEOUT,
        )
    logger.info("LLM backend: local (%s)", config.LLM_MODEL_NAME)
    return LocalLLM(config, device_arg=device_arg, dtype=dtype)
