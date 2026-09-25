"""Thin wrapper around the Gemini API: retries, JSON-schema output,
transcription, and logging. This replaces the OpenAI-based client - the
project switched providers from OpenAI back to Gemini.

Every call is logged (component + event + short detail), but request/
response bodies containing note text are not dumped into logs.detail
beyond a short excerpt, and the API key itself is never logged.
"""
from __future__ import annotations

import copy
import json
import mimetypes
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from google import genai
from google.genai import types

from app import db

MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 3.0
# Rate-limit and server-overload errors need longer backoff than a generic
# transient error - short retries just hit the same window.
OVERLOAD_BACKOFF_SECONDS = 8.0

DEFAULT_TRANSCRIBE_MODEL = "gemini-3.5-transcribe"

# Telegram voice notes are .ogg by default; mimetypes doesn't always know the
# others Meera might forward as regular audio files.
_AUDIO_MIME_OVERRIDES = {
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
}

TRANSCRIBE_INSTRUCTION = (
    "Transcribe this audio recording verbatim, word-for-word, in its "
    "original language. Return only the transcript text - no commentary, "
    "timestamps, or speaker labels."
)


class LLMError(RuntimeError):
    pass


def _is_overloaded(exc: Exception) -> bool:
    text = str(exc)
    return any(
        marker in text
        for marker in (
            "429",
            "RESOURCE_EXHAUSTED",
            "rate_limit",
            "rate limit",
            "503",
            "UNAVAILABLE",
            "overloaded",
            "server_error",
            "try again",
        )
    )


def _backoff_seconds(attempt: int, exc: Exception) -> float:
    base = OVERLOAD_BACKOFF_SECONDS if _is_overloaded(exc) else RETRY_BACKOFF_SECONDS
    return base * attempt


def _gemini_schema(schema: dict) -> dict:
    """Converts a plain JSON-schema dict (as written in app/drafting.py) into
    the subset Gemini's Schema type accepts: `type` values are upper-cased
    (Gemini's Type enum), and unsupported JSON-schema keys like
    `additionalProperties` are dropped rather than rejected.
    """
    schema = copy.deepcopy(schema)

    def walk(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        out: dict = {}
        t = node.get("type")
        if isinstance(t, str):
            out["type"] = t.upper()
        if "description" in node:
            out["description"] = node["description"]
        if "enum" in node:
            out["enum"] = node["enum"]
        if node.get("nullable"):
            out["nullable"] = True
        if "properties" in node:
            out["properties"] = {k: walk(v) for k, v in node["properties"].items()}
        if "required" in node:
            out["required"] = node["required"]
        if "items" in node:
            out["items"] = walk(node["items"])
        return out

    return walk(schema)


def _audio_mime_type(audio_path: Path) -> str:
    suffix = audio_path.suffix.lower()
    if suffix in _AUDIO_MIME_OVERRIDES:
        return _AUDIO_MIME_OVERRIDES[suffix]
    guessed, _ = mimetypes.guess_type(audio_path.name)
    if guessed and guessed.startswith("audio/"):
        return guessed
    raise LLMError(f"Don't know the audio MIME type for {audio_path.name!r}")


class LLMClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        conn: Optional[sqlite3.Connection] = None,
        *,
        timeout_seconds: Optional[float] = None,
        transcribe_model: str = DEFAULT_TRANSCRIBE_MODEL,
    ):
        http_options = types.HttpOptions(timeout=int(timeout_seconds * 1000)) if timeout_seconds else None
        self._client = genai.Client(api_key=api_key, http_options=http_options)
        self.model = model
        self.transcribe_model = transcribe_model
        self._conn = conn

    def _log(self, event: str, **detail: Any) -> None:
        if self._conn is not None:
            db.log(self._conn, "info", "llm", event, **detail)

    def generate_json(
        self,
        prompt: str,
        *,
        response_schema: dict,
        temperature: float = 0.4,
        schema_name: str = "response",
        max_retries: Optional[int] = None,
    ) -> dict:
        """Calls the model asking for JSON matching response_schema. Retries
        on transient failures and on invalid JSON.

        max_retries overrides the module default (MAX_RETRIES) for calls
        with their own tighter retry budget - e.g. the news-keyword
        extraction step (app/drafting.extract_news_keywords), which is
        specified to give up after a single retry and skip the news step
        rather than hold up drafting.
        """
        retries = max_retries if max_retries is not None else MAX_RETRIES
        schema = _gemini_schema(response_schema)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
        )
        last_error: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=config,
                )
                text = response.text
                if not text:
                    raise LLMError("empty response from Gemini")
                parsed = json.loads(text)
                self._log("generate_json.ok", attempt=attempt, chars=len(text))
                return parsed
            except Exception as exc:  # noqa: BLE001 - we want to retry broadly and log
                last_error = exc
                self._log("generate_json.error", attempt=attempt, error=str(exc))
                if attempt < retries:
                    time.sleep(_backoff_seconds(attempt, exc))
        raise LLMError(f"Gemini call failed after {retries} attempts: {last_error}")

    def transcribe_audio(self, audio_path: Path) -> str:
        """Transcribes a voice note by sending the audio bytes directly to
        Gemini. Retries on transient failures.
        """
        mime_type = _audio_mime_type(audio_path)
        audio_bytes = audio_path.read_bytes()
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.transcribe_model,
                    contents=[
                        TRANSCRIBE_INSTRUCTION,
                        types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                    ],
                )
                text = (response.text or "").strip()
                if not text:
                    raise LLMError("empty transcript from Gemini")
                self._log("transcribe.ok", attempt=attempt, path=str(audio_path))
                return text
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._log("transcribe.error", attempt=attempt, error=str(exc))
                if attempt < MAX_RETRIES:
                    time.sleep(_backoff_seconds(attempt, exc))
        raise LLMError(f"Transcription failed after {MAX_RETRIES} attempts: {last_error}")
