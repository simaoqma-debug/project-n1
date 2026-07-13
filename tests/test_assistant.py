from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from openai import APIConnectionError

from assistant import MicrophoneError, Settings, VoiceAssistant, main


class VoiceAssistantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MagicMock()
        self.settings = Settings()
        self.assistant = VoiceAssistant(
            self.client,
            self.settings,
            audio_player=MagicMock(),
        )

    def test_transcribe_uploads_wav_to_gpt_stt(self) -> None:
        audio = MagicMock()
        audio.get_wav_data.return_value = b"RIFF fake wav"
        self.client.audio.transcriptions.create.return_value.text = "  hello Pi  "

        text = self.assistant.transcribe(audio)

        self.assertEqual(text, "hello Pi")
        audio.get_wav_data.assert_called_once_with(convert_rate=16_000, convert_width=2)
        request = self.client.audio.transcriptions.create.call_args.kwargs
        self.assertEqual(request["model"], "gpt-4o-transcribe")
        self.assertEqual(request["file"].name, "microphone.wav")
        self.assertEqual(request["language"], "pt")
        self.assertIn("português europeu", request["prompt"])

    def test_respond_uses_responses_api_and_retains_history(self) -> None:
        self.client.responses.create.side_effect = [
            MagicMock(output_text=" First answer. "),
            MagicMock(output_text="Second answer."),
        ]

        first = self.assistant.respond("First question")
        second = self.assistant.respond("Follow up")

        self.assertEqual(first, "First answer.")
        self.assertEqual(second, "Second answer.")
        request = self.client.responses.create.call_args.kwargs
        self.assertEqual(request["model"], "gpt-5.6")
        self.assertEqual(request["reasoning"], {"effort": "none"})
        self.assertEqual(
            request["input"][-3:],
            [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer."},
                {"role": "user", "content": "Follow up"},
            ],
        )

    def test_speak_uses_gpt_tts_wav_and_removes_temporary_file(self) -> None:
        streamed_response = MagicMock()
        streamed_response.stream_to_file.side_effect = lambda path: Path(
            path
        ).write_bytes(b"RIFF fake wav")
        context = MagicMock()
        context.__enter__.return_value = streamed_response
        self.client.audio.speech.with_streaming_response.create.return_value = context
        played_paths: list[Path] = []
        self.assistant.audio_player = lambda path: played_paths.append(path)

        self.assistant.speak("Hello")

        request = (
            self.client.audio.speech.with_streaming_response.create.call_args.kwargs
        )
        self.assertEqual(request["model"], "gpt-4o-mini-tts")
        self.assertEqual(request["voice"], "marin")
        self.assertEqual(request["response_format"], "wav")
        self.assertEqual(len(played_paths), 1)
        self.assertFalse(played_paths[0].exists())

    def test_invalid_microphone_index_is_a_terminal_microphone_error(self) -> None:
        self.assistant.speak = MagicMock()
        with patch.object(
            self.assistant,
            "capture_audio",
            side_effect=MicrophoneError("invalid device"),
        ) as capture:
            result = self.assistant.run()
        self.assertFalse(result)
        capture.assert_called_once_with()

    def test_capture_translates_device_index_assertion(self) -> None:
        with patch("assistant.sr.Microphone", side_effect=AssertionError("bad index")):
            with self.assertRaises(MicrophoneError):
                self.assistant.capture_audio()

    def test_zero_history_turns_retains_no_messages(self) -> None:
        assistant = VoiceAssistant(
            self.client,
            Settings(history_turns=0),
            audio_player=MagicMock(),
        )
        self.client.responses.create.return_value.output_text = "Answer"

        assistant.respond("Question")

        self.assertEqual(assistant._history, [])

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            Settings(microphone_index=-1)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            Settings(listen_timeout=0)
        for invalid_timeout in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                Settings(listen_timeout=invalid_timeout)
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            Settings(history_turns=-1)

    def test_api_error_is_terminal(self) -> None:
        self.assistant.speak = MagicMock()
        error = APIConnectionError(request=httpx.Request("POST", "https://api.test"))
        with (
            patch("builtins.input", return_value="hello") as user_input,
            patch.object(self.assistant, "respond", side_effect=error),
        ):
            result = self.assistant.run(text_mode=True)

        self.assertFalse(result)
        user_input.assert_called_once_with("Tu: ")

    def test_main_requires_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(main(["--text"]), 2)

    def test_main_reports_invalid_environment_configuration(self) -> None:
        environment = {
            "OPENAI_API_KEY": "test-key",
            "MICROPHONE_INDEX": "not-a-number",
        }
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(main(["--text"]), 1)

    def test_main_returns_failure_after_terminal_runtime_error(self) -> None:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch("assistant.OpenAI"),
            patch("assistant.VoiceAssistant.run", return_value=False),
        ):
            self.assertEqual(main(["--text"]), 1)


if __name__ == "__main__":
    unittest.main()
