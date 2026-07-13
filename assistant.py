"""Assistente de voz para Raspberry Pi, em português europeu."""

from __future__ import annotations

import argparse
import io
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import speech_recognition as sr
from openai import APIError, OpenAI

from prompts import SYSTEM_INSTRUCTIONS


EXIT_COMMANDS = {
    "adeus",
    "exit",
    "goodbye",
    "parar",
    "quit",
    "sair",
    "stop",
    "terminar",
}


@dataclass(frozen=True)
class Settings:
    llm_model: str = "gpt-5.6"
    stt_model: str = "gpt-4o-transcribe"
    stt_language: str = "pt"
    stt_prompt: str = "O áudio está em português europeu (pt-PT)."
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "marin"
    tts_instructions: str = (
        "Fala em português europeu, de forma clara, calorosa e a um ritmo moderado."
    )
    reasoning_effort: str | None = "none"
    microphone_index: int | None = None
    listen_timeout: float = 5.0
    phrase_time_limit: float = 10.0
    max_output_tokens: int = 300
    history_turns: int = 10

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
        if self.microphone_index is not None and self.microphone_index < 0:
            raise ValueError("MICROPHONE_INDEX cannot be negative")
        if not math.isfinite(self.listen_timeout) or self.listen_timeout <= 0:
            raise ValueError(
                "LISTEN_TIMEOUT_SECONDS must be finite and greater than zero"
            )
        if not math.isfinite(self.phrase_time_limit) or self.phrase_time_limit <= 0:
            raise ValueError(
                "PHRASE_TIME_LIMIT_SECONDS must be finite and greater than zero"
            )
        if self.max_output_tokens <= 0:
            raise ValueError("OPENAI_MAX_OUTPUT_TOKENS must be greater than zero")
        if self.history_turns < 0:
            raise ValueError("CONVERSATION_HISTORY_TURNS cannot be negative")

    @classmethod
    def from_environment(cls) -> Settings:
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
            microphone_index=_optional_int(os.getenv("MICROPHONE_INDEX")),
            listen_timeout=float(
                os.getenv("LISTEN_TIMEOUT_SECONDS", cls.listen_timeout)
            ),
            phrase_time_limit=float(
                os.getenv("PHRASE_TIME_LIMIT_SECONDS", cls.phrase_time_limit)
            ),
            max_output_tokens=int(
                os.getenv("OPENAI_MAX_OUTPUT_TOKENS", cls.max_output_tokens)
            ),
            history_turns=int(
                os.getenv("CONVERSATION_HISTORY_TURNS", cls.history_turns)
            ),
        )


def _optional_int(value: str | None) -> int | None:
    return int(value) if value not in (None, "") else None


class MicrophoneError(RuntimeError):
    """Raised when the configured microphone cannot be opened."""


def play_wav(path: Path) -> None:
    """Play a WAV file with an available native player."""

    custom_player = os.getenv("AUDIO_PLAYER")
    if custom_player:
        command = [*shlex.split(custom_player), str(path)]
    elif sys.platform == "win32":
        import winsound

        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        return
    else:
        candidates = (
            ("afplay",),
            ("aplay",),
            ("paplay",),
            ("ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"),
        )
        command = []
        for candidate in candidates:
            if shutil.which(candidate[0]):
                command = [*candidate, str(path)]
                break

    if not command:
        raise RuntimeError(
            "Não foi encontrado um leitor WAV. Instale 'alsa-utils' (aplay) ou "
            "defina AUDIO_PLAYER."
        )

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class VoiceAssistant:
    def __init__(
        self,
        client: OpenAI,
        settings: Settings,
        *,
        recognizer: sr.Recognizer | None = None,
        audio_player: Callable[[Path], None] = play_wav,
    ) -> None:
        self.client = client
        self.settings = settings
        self.recognizer = recognizer or sr.Recognizer()
        self.audio_player = audio_player
        self._history: list[dict[str, str]] = []

    def capture_audio(self) -> sr.AudioData | None:
        """Capture one utterance from the selected microphone."""

        try:
            with sr.Microphone(device_index=self.settings.microphone_index) as source:
                print("\nA ouvir...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                return self.recognizer.listen(
                    source,
                    timeout=self.settings.listen_timeout,
                    phrase_time_limit=self.settings.phrase_time_limit,
                )
        except sr.WaitTimeoutError:
            return None
        except (AssertionError, AttributeError, OSError) as error:
            raise MicrophoneError(
                "Não foi possível abrir o microfone. Instale as dependências de "
                "áudio e verifique MICROPHONE_INDEX (execute --list-microphones)."
            ) from error

    def transcribe(self, audio: sr.AudioData) -> str:
        wav_file = io.BytesIO(audio.get_wav_data(convert_rate=16_000, convert_width=2))
        wav_file.name = "microphone.wav"
        transcript = self.client.audio.transcriptions.create(
            model=self.settings.stt_model,
            file=wav_file,
            language=self.settings.stt_language,
            prompt=self.settings.stt_prompt,
        )
        return transcript.text.strip()

    def respond(self, user_text: str) -> str:
        """Generate a concise answer with the OpenAI Responses API."""

        input_messages = [*self._history, {"role": "user", "content": user_text}]
        request: dict[str, Any] = {
            "model": self.settings.llm_model,
            "instructions": SYSTEM_INSTRUCTIONS,
            "input": input_messages,
            "max_output_tokens": self.settings.max_output_tokens,
        }
        if self.settings.reasoning_effort:
            request["reasoning"] = {"effort": self.settings.reasoning_effort}

        response = self.client.responses.create(**request)
        answer = response.output_text.strip()
        if not answer:
            raise RuntimeError("The language model returned no spoken response.")

        complete_history = [
            *input_messages,
            {"role": "assistant", "content": answer},
        ]
        if self.settings.history_turns:
            self._history = complete_history[-(self.settings.history_turns * 2) :]
        else:
            self._history = []
        return answer

    def speak(self, text: str) -> None:
        """Synthesize text with OpenAI GPT TTS and play the resulting WAV."""

        print(f"Assistente: {text}")
        file_descriptor, file_name = tempfile.mkstemp(suffix=".wav")
        os.close(file_descriptor)
        speech_path = Path(file_name)
        try:
            with self.client.audio.speech.with_streaming_response.create(
                model=self.settings.tts_model,
                voice=self.settings.tts_voice,
                input=text,
                instructions=self.settings.tts_instructions,
                response_format="wav",
            ) as response:
                response.stream_to_file(speech_path)
            self.audio_player(speech_path)
        finally:
            speech_path.unlink(missing_ok=True)

    def run(self, *, text_mode: bool = False) -> bool:
        """Run the conversation; return false after a terminal runtime error."""

        self.speak(
            "Olá! Sou o Zé Assistente, um assistente de voz com inteligência "
            "artificial. Como posso ajudar?"
        )

        while True:
            try:
                if text_mode:
                    user_text = input("Tu: ").strip()
                else:
                    audio = self.capture_audio()
                    if audio is None:
                        continue
                    user_text = self.transcribe(audio)
                    if user_text:
                        print(f"Tu: {user_text}")

                if not user_text:
                    continue
                if user_text.casefold().strip(" .!?") in EXIT_COMMANDS:
                    self.speak("Até breve!")
                    return True

                self.speak(self.respond(user_text))
            except (KeyboardInterrupt, EOFError):
                print("\nAté breve!")
                return True
            except MicrophoneError as error:
                print(f"Erro do microfone: {error}", file=sys.stderr)
                return False
            except APIError as error:
                print(f"Erro da API OpenAI: {error}", file=sys.stderr)
                return False
            except Exception as error:
                print(f"Erro do assistente: {error}", file=sys.stderr)


def list_microphones() -> int:
    try:
        names = sr.Microphone.list_microphone_names()
    except (AttributeError, OSError) as error:
        print(
            "Não foi possível enumerar os microfones. Instale primeiro as "
            "dependências de áudio: uv sync --extra audio",
            file=sys.stderr,
        )
        print(f"Details: {error}", file=sys.stderr)
        return 1

    for index, name in enumerate(names):
        print(f"{index}: {name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--microphone-index",
        type=int,
        help="índice do microfone (usa MICROPHONE_INDEX ou o dispositivo predefinido)",
    )
    parser.add_argument(
        "--list-microphones",
        action="store_true",
        help="lista os microfones e termina",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="permite escrever pedidos, mantendo o modelo GPT e a voz",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_microphones:
        return list_microphones()
    if not os.getenv("OPENAI_API_KEY"):
        print("Defina OPENAI_API_KEY antes de iniciar o assistente.", file=sys.stderr)
        return 2

    try:
        settings = Settings.from_environment()
        if args.microphone_index is not None:
            settings = replace(settings, microphone_index=args.microphone_index)
        assistant = VoiceAssistant(OpenAI(), settings)
        if not assistant.run(text_mode=args.text):
            return 1
    except Exception as error:
        print(f"Não foi possível iniciar o assistente: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
