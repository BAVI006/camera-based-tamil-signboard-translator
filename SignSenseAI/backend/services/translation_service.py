"""Configurable Tamil-to-English translation adapters.

The IndicTrans2 adapter follows AI4Bharat's Hugging Face interface and imports
its model dependencies only when selected. Model weights are downloaded by
Hugging Face on first use unless a local model directory is configured.
"""

from dataclasses import dataclass
import os
from threading import Lock
from typing import Protocol


class TranslationUnavailable(RuntimeError):
    """Raised when no translator is configured or its dependencies are absent."""


class TranslationError(RuntimeError):
    """Raised when configured translation inference fails."""


class Translator(Protocol):
    def translate(
        self,
        tamil_text: str,
        target_language: str = "eng_Latn",
    ) -> "TranslationResult":
        ...


@dataclass(frozen=True)
class TranslationResult:
    text: str
    source_language: str = "tam_Taml"
    target_language: str = "eng_Latn"
    model: str = "configured-translator"


def _split_text(text: str, max_chars: int = 700) -> list[str]:
    chunks: list[str] = []
    for paragraph in text.splitlines():
        remaining = paragraph.strip()
        while len(remaining) > max_chars:
            split_at = remaining.rfind(" ", 0, max_chars + 1)
            if split_at < max_chars // 2:
                split_at = max_chars
            chunks.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()
        if remaining:
            chunks.append(remaining)
    return chunks


class IndicTrans2Translator:
    """Lazy, reusable IndicTrans2 Tamil-to-English Hugging Face adapter."""

    def __init__(self, model_name: str | None = None, timeout_seconds: int = 90):
        self.model_name = model_name or os.getenv(
            "SIGN_TRANSLATION_MODEL",
            "ai4bharat/indictrans2-indic-en-dist-200M",
        )
        self.timeout_seconds = timeout_seconds
        self._lock = Lock()
        self._model_components = None

    def _load(self):
        if self._model_components is not None:
            return self._model_components
        with self._lock:
            if self._model_components is not None:
                return self._model_components
            try:
                import torch
                from IndicTransToolkit.processor import IndicProcessor
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                model.eval()
                processor = IndicProcessor(inference=True)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                raise TranslationUnavailable(
                    "IndicTrans2 is not ready. Install its documented dependencies "
                    "and make the configured model available locally or online. "
                    f"Details: {exc}"
                ) from exc
            self._model_components = (torch, tokenizer, model, processor, device)
        return self._model_components

    def translate(
        self,
        tamil_text: str,
        target_language: str = "eng_Latn",
    ) -> TranslationResult:
        if not tamil_text or not tamil_text.strip():
            raise TranslationError("There is no recognized text to translate.")
        if len(tamil_text) > 5000:
            raise TranslationError("Text is too long to translate (maximum 5,000 characters).")
        if target_language != "eng_Latn":
            raise TranslationError("The configured translation model supports English only.")

        torch, tokenizer, model, processor, device = self._load()
        chunks = _split_text(tamil_text)
        translated_chunks: list[str] = []
        try:
            for chunk in chunks:
                prepared = processor.preprocess_batch(
                    [chunk],
                    src_lang="tam_Taml",
                    tgt_lang=target_language,
                )
                inputs = tokenizer(
                    prepared,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                ).to(device)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        use_cache=True,
                        min_length=0,
                        max_length=256,
                        num_beams=5,
                        num_return_sequences=1,
                    )
                decoded = tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                translated_chunks.extend(
                    processor.postprocess_batch(decoded, lang=target_language)
                )
        except (RuntimeError, OSError, ValueError, IndexError) as exc:
            raise TranslationError(f"IndicTrans2 inference failed: {exc}") from exc

        translation = " ".join(part.strip() for part in translated_chunks if part.strip())
        if not translation:
            raise TranslationError("IndicTrans2 did not return a translation.")
        return TranslationResult(text=translation, model=self.model_name)


class NLLBTranslator:
    """CPU-compatible public NLLB adapter for Tamil-to-English translation."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv(
            "SIGN_TRANSLATION_MODEL",
            "facebook/nllb-200-distilled-600M",
        )
        self._lock = Lock()
        self._model_components = None

    def _load(self):
        if self._model_components is not None:
            return self._model_components
        with self._lock:
            if self._model_components is not None:
                return self._model_components
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                tokenizer.src_lang = "tam_Taml"
                model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if device.type == "cpu":
                    torch.set_num_threads(min(4, os.cpu_count() or 1))
                model.to(device)
                model.eval()
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                raise TranslationUnavailable(
                    "NLLB is not ready. Install the optional translation dependencies "
                    "and make the configured model available locally or online. "
                    f"Details: {exc}"
                ) from exc
            self._model_components = (torch, tokenizer, model, device)
        return self._model_components

    def translate(
        self,
        tamil_text: str,
        target_language: str = "eng_Latn",
    ) -> TranslationResult:
        if not tamil_text or not tamil_text.strip():
            raise TranslationError("There is no recognized text to translate.")
        if len(tamil_text) > 5000:
            raise TranslationError("Text is too long to translate (maximum 5,000 characters).")
        if target_language != "eng_Latn":
            raise TranslationError("The configured translation model supports English only.")

        torch, tokenizer, model, device = self._load()
        translated_chunks: list[str] = []
        try:
            for chunk in _split_text(tamil_text):
                inputs = tokenizer(
                    chunk,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
                        max_new_tokens=128,
                        num_beams=2,
                        early_stopping=True,
                    )
                translated_chunks.extend(
                    tokenizer.batch_decode(
                        generated,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                )
        except (RuntimeError, OSError, ValueError, IndexError) as exc:
            raise TranslationError(f"NLLB inference failed: {exc}") from exc

        translation = " ".join(part.strip() for part in translated_chunks if part.strip())
        if not translation:
            raise TranslationError("NLLB did not return a translation.")
        return TranslationResult(text=translation, model=self.model_name)


class HostedTranslationAPI:
    """Adapter for a trusted, user-configured HTTP translation endpoint.

    Expected JSON request keys: source_text, source_language, target_language.
    Expected response key: translation. The endpoint and bearer token are never
    hardcoded and must be configured explicitly by the project owner.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str | None = None,
        timeout_seconds: float = 20,
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def translate(
        self,
        tamil_text: str,
        target_language: str = "eng_Latn",
    ) -> TranslationResult:
        if not tamil_text or not tamil_text.strip():
            raise TranslationError("There is no recognized text to translate.")
        if len(tamil_text) > 5000:
            raise TranslationError("Text is too long to translate (maximum 5,000 characters).")
        if target_language != "eng_Latn":
            raise TranslationError("Only English is enabled for translation at this time.")

        try:
            import httpx
        except ImportError as exc:
            raise TranslationUnavailable(
                "Install the backend requirements to use the hosted translation adapter."
            ) from exc

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = httpx.post(
                self.endpoint,
                json={
                    "source_text": tamil_text,
                    "source_language": "tam_Taml",
                    "target_language": target_language,
                },
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            translation = response.json().get("translation")
        except (httpx.HTTPError, ValueError) as exc:
            raise TranslationError(f"Hosted translation request failed: {exc}") from exc
        if not isinstance(translation, str) or not translation.strip():
            raise TranslationError("Translation endpoint returned no translation.")
        return TranslationResult(text=translation.strip(), model="hosted-api")


def create_translator() -> Translator:
    backend = os.getenv("SIGN_TRANSLATION_BACKEND", "").strip().lower()
    if backend == "indictrans2":
        return IndicTrans2Translator()
    if backend == "nllb":
        return NLLBTranslator()
    if backend == "http":
        endpoint = os.getenv("SIGN_TRANSLATION_API_URL")
        if not endpoint:
            raise TranslationUnavailable(
                "Set SIGN_TRANSLATION_API_URL to enable the hosted translation adapter."
            )
        return HostedTranslationAPI(
            endpoint=endpoint,
            api_key=os.getenv("SIGN_TRANSLATION_API_KEY"),
        )
    raise TranslationUnavailable(
        "Translation is not configured. Set SIGN_TRANSLATION_BACKEND to "
        "'indictrans2', 'nllb', or 'http'."
    )


def translate_tamil(
    tamil_text: str,
    target_language: str = "eng_Latn",
    translator: Translator | None = None,
) -> TranslationResult:
    selected = translator or create_translator()
    return selected.translate(tamil_text, target_language=target_language)
