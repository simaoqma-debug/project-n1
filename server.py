"""Flask API for the browser-based Pocket Assistant."""

from __future__ import annotations

import io
import os
import re
import secrets
import sysconfig
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

from flask import Flask, Response, jsonify, request, send_from_directory
from openai import APIError, OpenAI

from prompts import SYSTEM_INSTRUCTIONS

MAX_AUDIO_BYTES = 25 * 1024 * 1024
CONVERSATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
MIME_EXTENSIONS = {
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/m4a": ".m4a",
    "audio/mp4": ".mp4",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/webm": ".webm",
    "video/mp4": ".mp4",
}
ALLOWED_EXTENSIONS = set(MIME_EXTENSIONS.values())
DEFAULT_ALLOWED_ORIGINS = {
    "http://127.0.0.1:4321",
    "http://127.0.0.1:5000",
    "http://localhost:4321",
    "http://localhost:5000",
    "http://[::1]:4321",
    "http://[::1]:5000",
}


class ConfigurationError(RuntimeError):
    """Raised when the web service is not configured."""


class VoiceInputError(ValueError):
    """Raised when uploaded audio cannot produce a useful transcript."""


@dataclass(frozen=True)
class WebSettings:
    """OpenAI settings used by the Flask voice pipeline."""

    llm_model: str = "gpt-5.6-luna"
    stt_model: str = "gpt-4o-transcribe"
    stt_language: str = "pt"
    stt_prompt: str = "O áudio está em português europeu (pt-PT)."
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "marin"
    tts_instructions: str = (
        "Fala em português europeu, de forma clara, calorosa e a um ritmo moderado."
    )
    reasoning_effort: str | None = "none"
    max_output_tokens: int = 300
    history_turns: int = 10
    max_conversations: int = 100

    @classmethod
    def from_environment(cls) -> WebSettings:
        return cls(
            llm_model=os.getenv("OPENAI_LLM_MODEL", cls.llm_model),
            stt_model=os.getenv("OPENAI_STT_MODEL", cls.stt_model),
            stt_language=os.getenv("OPENAI_STT_LANGUAGE", cls.stt_language),
            stt_prompt=os.getenv("OPENAI_STT_PROMPT", cls.stt_prompt),
            tts_model=os.getenv("OPENAI_TTS_MODEL", cls.tts_model),
            tts_voice=os.getenv("OPENAI_TTS_VOICE", cls.tts_voice),
            tts_instructions=os.getenv("OPENAI_TTS_INSTRUCTIONS", cls.tts_instructions),
            reasoning_effort=os.getenv(
                "OPENAI_REASONING_EFFORT", cls.reasoning_effort or ""
            )
            or None,
            max_output_tokens=int(
                os.getenv("OPENAI_MAX_OUTPUT_TOKENS", cls.max_output_tokens)
            ),
            history_turns=int(
                os.getenv("CONVERSATION_HISTORY_TURNS", cls.history_turns)
            ),
            max_conversations=int(
                os.getenv("MAX_WEB_CONVERSATIONS", cls.max_conversations)
            ),
        )

    def __post_init__(self) -> None:
        for name in (
            "llm_model",
            "stt_model",
            "stt_language",
            "tts_model",
            "tts_voice",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be empty")
        if self.max_output_tokens <= 0:
            raise ValueError("OPENAI_MAX_OUTPUT_TOKENS must be greater than zero")
        if self.history_turns < 0:
            raise ValueError("CONVERSATION_HISTORY_TURNS cannot be negative")
        if self.max_conversations <= 0:
            raise ValueError("MAX_WEB_CONVERSATIONS must be greater than zero")


@dataclass(frozen=True)
class VoiceResult:
    audio: bytes
    transcript: str
    answer: str


class VoiceProcessor(Protocol):
    def process(
        self,
        audio: bytes,
        *,
        filename: str,
        content_type: str | None,
        conversation_id: str,
    ) -> VoiceResult: ...


class SlidingWindowRateLimiter:
    """Bound paid voice requests per client address."""

    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute <= 0:
            raise ValueError("VOICE_REQUESTS_PER_MINUTE must be greater than zero")
        self.requests_per_minute = requests_per_minute
        self._requests: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, client_id: str) -> bool:
        now = time.monotonic()
        cutoff = now - 60
        with self._lock:
            timestamps = self._requests.setdefault(client_id, deque())
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()
            if len(timestamps) >= self.requests_per_minute:
                return False
            timestamps.append(now)
            return True


class ConversationStore:
    """Small in-memory LRU store that keeps browser conversations isolated."""

    def __init__(self, *, history_turns: int, max_conversations: int) -> None:
        self.history_turns = history_turns
        self.max_conversations = max_conversations
        self._histories: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, conversation_id: str) -> list[dict[str, str]]:
        with self._lock:
            history = self._histories.get(conversation_id, [])
            if conversation_id in self._histories:
                self._histories.move_to_end(conversation_id)
            return [message.copy() for message in history]

    def commit(
        self,
        conversation_id: str,
        input_messages: list[dict[str, str]],
        answer: str,
    ) -> None:
        complete_history = [
            *input_messages,
            {"role": "assistant", "content": answer},
        ]
        if self.history_turns:
            complete_history = complete_history[-(self.history_turns * 2) :]
        else:
            complete_history = []

        with self._lock:
            self._histories[conversation_id] = complete_history
            self._histories.move_to_end(conversation_id)
            while len(self._histories) > self.max_conversations:
                self._histories.popitem(last=False)


class OpenAIVoiceProcessor:
    """Run uploaded audio through OpenAI STT, LLM, and TTS."""

    def __init__(self, client: OpenAI, settings: WebSettings) -> None:
        self.client = client
        self.settings = settings
        self.conversations = ConversationStore(
            history_turns=settings.history_turns,
            max_conversations=settings.max_conversations,
        )

    def process(
        self,
        audio: bytes,
        *,
        filename: str,
        content_type: str | None,
        conversation_id: str,
    ) -> VoiceResult:
        upload = io.BytesIO(audio)
        upload.name = _audio_filename(filename, content_type)
        transcription = self.client.audio.transcriptions.create(
            model=self.settings.stt_model,
            file=upload,
            language=self.settings.stt_language,
            prompt=self.settings.stt_prompt,
        )
        transcript = transcription.text.strip()
        if not transcript:
            raise VoiceInputError("Não foi detetada fala na gravação.")

        input_messages = [
            *self.conversations.get(conversation_id),
            {"role": "user", "content": transcript},
        ]
        response_request: dict[str, Any] = {
            "model": self.settings.llm_model,
            "instructions": SYSTEM_INSTRUCTIONS,
            "input": input_messages,
            "max_output_tokens": self.settings.max_output_tokens,
        }
        if self.settings.reasoning_effort:
            response_request["reasoning"] = {"effort": self.settings.reasoning_effort}

        model_response = self.client.responses.create(**response_request)
        answer = model_response.output_text.strip()
        if not answer:
            raise RuntimeError("O modelo de linguagem não devolveu uma resposta.")

        with self.client.audio.speech.with_streaming_response.create(
            model=self.settings.tts_model,
            voice=self.settings.tts_voice,
            input=answer,
            instructions=self.settings.tts_instructions,
            response_format="wav",
        ) as speech_response:
            synthesized_audio = speech_response.read()
        if not synthesized_audio:
            raise RuntimeError("O modelo de voz não devolveu áudio.")

        self.conversations.commit(conversation_id, input_messages, answer)
        return VoiceResult(
            audio=synthesized_audio,
            transcript=transcript,
            answer=answer,
        )


def _audio_filename(filename: str, content_type: str | None) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        suffix = MIME_EXTENSIONS.get((content_type or "").split(";", 1)[0], ".webm")
    return f"recording{suffix}"


def _valid_conversation_id(value: str) -> bool:
    return bool(CONVERSATION_ID_PATTERN.fullmatch(value))


def _default_frontend_dist(root: Path) -> Path:
    configured_dist = os.getenv("POCKET_ASSISTANT_FRONTEND_DIST")
    if configured_dist:
        return Path(configured_dist)
    source_dist = root / "frontend" / "dist"
    if source_dist.is_dir():
        return source_dist
    return Path(sysconfig.get_path("data")) / "share" / "pocket-assistant" / "frontend"


def _is_loopback_address(value: str | None) -> bool:
    if not value:
        return False
    try:
        return ip_address(value.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _is_local_hostname(value: str | None) -> bool:
    return bool(value) and (
        value.casefold() == "localhost" or _is_loopback_address(value)
    )


def _allowed_origins() -> set[str]:
    configured = {
        origin.strip().rstrip("/")
        for origin in os.getenv("POCKET_ASSISTANT_ALLOWED_ORIGINS", "").split(",")
        if origin.strip()
    }
    return DEFAULT_ALLOWED_ORIGINS | configured


def create_app(
    voice_processor: VoiceProcessor | None = None,
    *,
    frontend_dist: Path | None = None,
) -> Flask:
    root = Path(__file__).resolve().parent
    dist_directory = frontend_dist or _default_frontend_dist(root)
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_AUDIO_BYTES

    processor = voice_processor
    processor_lock = threading.Lock()
    configuration_error: str | None = None
    try:
        rate_limiter = SlidingWindowRateLimiter(
            int(os.getenv("VOICE_REQUESTS_PER_MINUTE", "20"))
        )
    except ValueError:
        rate_limiter = SlidingWindowRateLimiter(20)
        configuration_error = "Configuração inválida do servidor."

    def get_processor() -> VoiceProcessor:
        nonlocal processor
        if processor is not None:
            return processor
        if not os.getenv("OPENAI_API_KEY"):
            raise ConfigurationError(
                "Defina OPENAI_API_KEY antes de utilizar a API de voz."
            )
        with processor_lock:
            if processor is None:
                try:
                    settings = WebSettings.from_environment()
                except (TypeError, ValueError) as error:
                    raise ConfigurationError(
                        "Configuração inválida do servidor."
                    ) from error
                processor = OpenAIVoiceProcessor(OpenAI(), settings)
        return processor

    @app.get("/api/health")
    def health() -> Response:
        configuration_valid = configuration_error is None
        if processor is None and os.getenv("OPENAI_API_KEY") and configuration_valid:
            try:
                WebSettings.from_environment()
            except (TypeError, ValueError):
                configuration_valid = False
        configured = processor is not None or (
            bool(os.getenv("OPENAI_API_KEY")) and configuration_valid
        )
        return jsonify(
            status="ok" if configuration_valid else "error",
            configured=configured,
        )

    @app.post("/api/voice")
    def voice() -> Response | tuple[Response, int]:
        if configuration_error:
            return jsonify(error=configuration_error), 503
        if not _is_loopback_address(request.remote_addr):
            return jsonify(error="A API de voz só está disponível localmente."), 403
        if request.headers.get("X-Pocket-Assistant") != "web":
            return jsonify(error="Acesso proibido."), 403

        origin = request.headers.get("Origin")
        request_origin = request.host_url.rstrip("/")
        request_hostname = urlsplit(request_origin).hostname
        same_local_origin = (
            _is_local_hostname(request_hostname)
            and origin is not None
            and origin.rstrip("/") == request_origin
        )
        if (
            origin
            and origin.rstrip("/") not in _allowed_origins()
            and not same_local_origin
        ):
            return jsonify(error="Origem não autorizada."), 403

        client_id = request.remote_addr or "unknown"
        if not rate_limiter.allow(client_id):
            response = jsonify(
                error="Demasiados pedidos de voz. Tente novamente dentro de instantes."
            )
            response.headers["Retry-After"] = "60"
            return response, 429

        uploaded_audio = request.files.get("audio")
        if uploaded_audio is None:
            return jsonify(error="Anexe uma gravação no campo 'audio'."), 400

        audio = uploaded_audio.read()
        if not audio:
            return jsonify(error="A gravação enviada está vazia."), 400

        conversation_id = request.form.get("conversation_id", "").strip()
        if not conversation_id:
            conversation_id = secrets.token_urlsafe(18)
        if not _valid_conversation_id(conversation_id):
            return jsonify(error="conversation_id inválido."), 400

        try:
            result = get_processor().process(
                audio,
                filename=uploaded_audio.filename or "recording.webm",
                content_type=uploaded_audio.mimetype,
                conversation_id=conversation_id,
            )
        except ConfigurationError as error:
            return jsonify(error=str(error)), 503
        except VoiceInputError as error:
            return jsonify(error=str(error)), 422
        except APIError:
            app.logger.exception("OpenAI voice request failed")
            return jsonify(
                error="O serviço de voz está temporariamente indisponível."
            ), 502
        except (TypeError, ValueError) as error:
            return jsonify(error=str(error)), 400
        except Exception:
            app.logger.exception("Voice request failed")
            return jsonify(error="Não foi possível concluir o pedido de voz."), 500

        response = Response(result.audio, mimetype="audio/wav")
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Conversation-ID"] = conversation_id
        response.headers["X-User-Text"] = quote(result.transcript, safe="")
        response.headers["X-Assistant-Text"] = quote(result.answer, safe="")
        return response

    @app.errorhandler(413)
    def request_too_large(_error: Exception) -> tuple[Response, int]:
        return jsonify(error="A gravação excede o limite de 25 MB."), 413

    @app.get("/")
    def frontend_index() -> Response | tuple[Response, int]:
        index_path = dist_directory / "index.html"
        if not index_path.is_file():
            return (
                jsonify(
                    error="Frontend por compilar. Execute 'npm run build' em frontend/."
                ),
                404,
            )
        return send_from_directory(dist_directory, "index.html")

    @app.get("/<path:asset_path>")
    def frontend_asset(asset_path: str) -> Response | tuple[Response, int]:
        requested_path = dist_directory / asset_path
        if requested_path.is_file():
            return send_from_directory(dist_directory, asset_path)
        return jsonify(error="Não encontrado."), 404

    return app


app = create_app()


def main() -> None:
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ConfigurationError(
            "O Pocket Assistant apenas suporta endereços de loopback."
        )
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
