from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import unquote

from server import (
    OpenAIVoiceProcessor,
    VoiceResult,
    WebSettings,
    _audio_filename,
    create_app,
)


class FakeVoiceProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def process(
        self,
        audio: bytes,
        *,
        filename: str,
        content_type: str | None,
        conversation_id: str,
    ) -> VoiceResult:
        self.calls.append(
            {
                "audio": audio,
                "filename": filename,
                "content_type": content_type,
                "conversation_id": conversation_id,
            }
        )
        return VoiceResult(
            audio=b"RIFF synthesized wav",
            transcript="olá",
            answer="Olá! Como posso ajudar?",
        )


class FlaskApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = FakeVoiceProcessor()
        self.app = create_app(self.processor)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        self.headers = {
            "X-Pocket-Assistant": "web",
            "Origin": "http://localhost:5000",
        }

    def test_health_endpoint(self) -> None:
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"configured": True, "status": "ok"})

    def test_voice_endpoint_returns_wav(self) -> None:
        response = self.client.post(
            "/api/voice",
            data={
                "audio": (io.BytesIO(b"webm audio"), "recording.webm"),
                "conversation_id": "browser-session-1",
            },
            content_type="multipart/form-data",
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "audio/wav")
        self.assertEqual(response.data, b"RIFF synthesized wav")
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(response.headers["X-Conversation-ID"], "browser-session-1")
        self.assertEqual(unquote(response.headers["X-User-Text"]), "olá")
        self.assertEqual(
            unquote(response.headers["X-Assistant-Text"]), "Olá! Como posso ajudar?"
        )
        self.assertEqual(self.processor.calls[0]["audio"], b"webm audio")

    def test_voice_endpoint_validates_input(self) -> None:
        missing = self.client.post("/api/voice", headers=self.headers)
        empty = self.client.post(
            "/api/voice",
            data={"audio": (io.BytesIO(), "recording.webm")},
            content_type="multipart/form-data",
            headers=self.headers,
        )
        invalid_id = self.client.post(
            "/api/voice",
            data={
                "audio": (io.BytesIO(b"audio"), "recording.webm"),
                "conversation_id": "not allowed/../",
            },
            content_type="multipart/form-data",
            headers=self.headers,
        )

        self.assertEqual(missing.status_code, 400)
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(invalid_id.status_code, 400)
        self.assertEqual(self.processor.calls, [])

    def test_unconfigured_voice_endpoint_returns_503(self) -> None:
        app = create_app()
        app.config["TESTING"] = True
        with patch.dict(os.environ, {}, clear=True):
            response = app.test_client().post(
                "/api/voice",
                data={"audio": (io.BytesIO(b"audio"), "recording.webm")},
                content_type="multipart/form-data",
                headers=self.headers,
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("OPENAI_API_KEY", response.get_json()["error"])

    def test_voice_endpoint_rejects_untrusted_browser_requests(self) -> None:
        missing_header = self.client.post(
            "/api/voice",
            data={"audio": (io.BytesIO(b"audio"), "recording.webm")},
            content_type="multipart/form-data",
        )
        bad_origin = self.client.post(
            "/api/voice",
            data={"audio": (io.BytesIO(b"audio"), "recording.webm")},
            content_type="multipart/form-data",
            headers={
                "X-Pocket-Assistant": "web",
                "Origin": "https://untrusted.example",
            },
        )

        self.assertEqual(missing_header.status_code, 403)
        self.assertEqual(bad_origin.status_code, 403)
        self.assertEqual(self.processor.calls, [])

    def test_voice_endpoint_rejects_non_loopback_clients(self) -> None:
        response = self.client.post(
            "/api/voice",
            data={"audio": (io.BytesIO(b"audio"), "recording.webm")},
            content_type="multipart/form-data",
            headers={"X-Pocket-Assistant": "web"},
            environ_overrides={"REMOTE_ADDR": "192.0.2.10"},
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(
            response.get_json()["error"],
            "A API de voz só está disponível localmente.",
        )
        self.assertEqual(self.processor.calls, [])

    def test_voice_endpoint_accepts_same_origin_on_custom_local_port(self) -> None:
        response = self.client.post(
            "/api/voice",
            data={"audio": (io.BytesIO(b"audio"), "recording.webm")},
            content_type="multipart/form-data",
            headers={
                "Host": "localhost:8000",
                "Origin": "http://localhost:8000",
                "X-Pocket-Assistant": "web",
            },
        )

        self.assertEqual(response.status_code, 200)

    def test_voice_endpoint_rejects_matching_non_local_host_and_origin(self) -> None:
        response = self.client.post(
            "/api/voice",
            data={"audio": (io.BytesIO(b"audio"), "recording.webm")},
            content_type="multipart/form-data",
            headers={
                "Host": "attacker.example:5000",
                "Origin": "http://attacker.example:5000",
                "X-Pocket-Assistant": "web",
            },
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.get_json()["error"], "Origem não autorizada.")
        self.assertEqual(self.processor.calls, [])

    def test_voice_endpoint_rate_limits_paid_requests(self) -> None:
        with patch.dict(os.environ, {"VOICE_REQUESTS_PER_MINUTE": "1"}, clear=True):
            app = create_app(self.processor)
        client = app.test_client()

        first = client.post(
            "/api/voice",
            data={"audio": (io.BytesIO(b"first"), "recording.webm")},
            content_type="multipart/form-data",
            headers=self.headers,
        )
        second = client.post(
            "/api/voice",
            data={"audio": (io.BytesIO(b"second"), "recording.webm")},
            content_type="multipart/form-data",
            headers=self.headers,
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)
        self.assertEqual(second.headers["Retry-After"], "60")

    def test_invalid_server_configuration_returns_503(self) -> None:
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test", "OPENAI_MAX_OUTPUT_TOKENS": "bad"},
            clear=True,
        ):
            app = create_app()
            client = app.test_client()
            health = client.get("/api/health")
            response = client.post(
                "/api/voice",
                data={"audio": (io.BytesIO(b"audio"), "recording.webm")},
                content_type="multipart/form-data",
                headers=self.headers,
            )

        self.assertEqual(health.get_json(), {"configured": False, "status": "error"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.get_json()["error"], "Configuração inválida do servidor."
        )

    def test_built_frontend_is_served(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dist = Path(directory)
            (dist / "index.html").write_text("<h1>Pocket Assistant</h1>")
            app = create_app(self.processor, frontend_dist=dist)
            response = app.test_client().get("/")
            response_data = response.get_data()
            response.close()

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Pocket Assistant", response_data)


class OpenAIVoiceProcessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MagicMock()
        self.client.audio.transcriptions.create.return_value.text = "  hello  "
        self.client.responses.create.return_value.output_text = "  Hi there.  "
        speech_response = MagicMock()
        speech_response.read.return_value = b"RIFF reply"
        speech_context = MagicMock()
        speech_context.__enter__.return_value = speech_response
        self.client.audio.speech.with_streaming_response.create.return_value = (
            speech_context
        )
        self.processor = OpenAIVoiceProcessor(self.client, WebSettings())

    def test_process_runs_stt_llm_and_tts(self) -> None:
        result = self.processor.process(
            b"browser audio",
            filename="recording.webm",
            content_type="audio/webm;codecs=opus",
            conversation_id="conversation-1",
        )

        self.assertEqual(result.transcript, "hello")
        self.assertEqual(result.answer, "Hi there.")
        self.assertEqual(result.audio, b"RIFF reply")
        transcription_request = self.client.audio.transcriptions.create.call_args.kwargs
        self.assertEqual(transcription_request["model"], "gpt-4o-transcribe")
        self.assertEqual(transcription_request["file"].name, "recording.webm")
        self.assertEqual(transcription_request["language"], "pt")
        self.assertIn("português europeu", transcription_request["prompt"])
        response_request = self.client.responses.create.call_args.kwargs
        self.assertEqual(response_request["model"], "gpt-5.6-luna")
        self.assertEqual(response_request["reasoning"], {"effort": "none"})
        speech_request = (
            self.client.audio.speech.with_streaming_response.create.call_args.kwargs
        )
        self.assertEqual(speech_request["model"], "gpt-4o-mini-tts")
        self.assertEqual(speech_request["response_format"], "wav")

    def test_process_retains_conversation_history(self) -> None:
        self.processor.process(
            b"first",
            filename="first.webm",
            content_type="audio/webm",
            conversation_id="same-conversation",
        )
        self.client.audio.transcriptions.create.return_value.text = "follow up"

        self.processor.process(
            b"second",
            filename="second.webm",
            content_type="audio/webm",
            conversation_id="same-conversation",
        )

        messages = self.client.responses.create.call_args.kwargs["input"]
        self.assertEqual(
            messages[-3:],
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hi there."},
                {"role": "user", "content": "follow up"},
            ],
        )

    def test_audio_filename_uses_supported_browser_format(self) -> None:
        self.assertEqual(_audio_filename("blob", "audio/mp4"), "recording.mp4")
        self.assertEqual(_audio_filename("clip.webm", None), "recording.webm")


if __name__ == "__main__":
    unittest.main()
