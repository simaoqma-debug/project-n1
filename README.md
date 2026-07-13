# Pocket Assistant

A small push-to-talk web assistant with a Flask API and an Astro + SolidJS frontend. The browser records the microphone, and the Flask endpoint runs the recording through OpenAI for all three AI stages:

| Stage | Default model | API |
|---|---|---|
| Speech-to-text | `gpt-4o-transcribe` | Audio Transcriptions |
| Language model | `gpt-5.6-luna` | Responses |
| Text-to-speech | `gpt-4o-mini-tts` (`marin`) | Audio Speech |

The frontend deliberately contains only the **Pocket Assistant** heading and a push-to-talk button on a white background. The interface, transcription guidance, responses, and synthesized voice use European Portuguese (`pt-PT`).

> The spoken voice is AI-generated. Microphone recordings and conversation text are sent to OpenAI.

## Run with Nix

The flake builds the Astro/SolidJS frontend, prepares the Flask/OpenAI Python environment, and starts the complete application with one command:

```bash
export OPENAI_API_KEY="your-openai-api-key"
nix run
```

Open `http://127.0.0.1:5000`. The first run may take longer while Nix downloads and builds the pinned dependencies. A development shell is also available with `nix develop`.

## Development setup

The web version records in the browser, so it does **not** need PyAudio or PortAudio.

```bash
uv sync
cd frontend
npm install
```

Set the API key in the shell that runs Flask:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Start the backend:

```bash
uv run pocket-assistant-api
```

In another terminal, start Astro:

```bash
cd frontend
npm run dev
```

Open `http://localhost:4321`. Hold the button while speaking, then release it to send the recording. Astro proxies `/api` to Flask at `http://127.0.0.1:5000`.

Browser microphone access works on `localhost` or over HTTPS. A plain HTTP page opened from another machine will normally be denied microphone access.

## Production-style local run

Build the static frontend and let Flask serve it:

```bash
cd frontend
npm run build
cd ..
export OPENAI_API_KEY="your-openai-api-key"
uv run pocket-assistant-api
```

Then open `http://127.0.0.1:5000`.

## API endpoints

### `GET /api/health`

Returns the backend status and whether an OpenAI API key is configured.

### `POST /api/voice`

Accepts `multipart/form-data`:

- `audio`: browser recording (`webm`, `mp4`, `ogg`, `wav`, and other OpenAI-supported audio formats)
- `conversation_id`: optional identifier used for isolated, in-memory conversation history

A successful request returns `audio/wav`. The percent-encoded `X-User-Text` and `X-Assistant-Text` response headers contain the transcript and answer shown in the conversation panel. Errors return JSON. Uploads are limited to 25 MB. Browser requests must include `X-Pocket-Assistant: web`; the frontend adds it automatically.

Example:

```bash
curl -X POST http://127.0.0.1:5000/api/voice \
  -H 'X-Pocket-Assistant: web' \
  -F 'audio=@recording.webm' \
  -F 'conversation_id=demo' \
  --output reply.wav
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | required | OpenAI authentication |
| `OPENAI_LLM_MODEL` | `gpt-5.6-luna` | Responses API model |
| `OPENAI_STT_MODEL` | `gpt-4o-transcribe` | transcription model |
| `OPENAI_STT_LANGUAGE` | `pt` | transcription language code |
| `OPENAI_STT_PROMPT` | European Portuguese hint | steers transcription toward `pt-PT` |
| `OPENAI_TTS_MODEL` | `gpt-4o-mini-tts` | speech model |
| `OPENAI_TTS_VOICE` | `marin` | built-in TTS voice |
| `OPENAI_REASONING_EFFORT` | `none` | low-latency reasoning effort |
| `OPENAI_MAX_OUTPUT_TOKENS` | `300` | keeps spoken answers concise |
| `CONVERSATION_HISTORY_TURNS` | `10` | turns retained per browser conversation |
| `MAX_WEB_CONVERSATIONS` | `100` | maximum in-memory conversations |
| `VOICE_REQUESTS_PER_MINUTE` | `20` | per-client voice request limit |
| `FLASK_HOST` | `127.0.0.1` | loopback bind address (`127.0.0.1`, `localhost`, or `::1`) |
| `FLASK_PORT` | `5000` | backend port |
| `POCKET_ASSISTANT_ALLOWED_ORIGINS` | local origins | comma-separated additional browser origins |

The bundled Flask app is intentionally local-only: both the launcher and `/api/voice` reject non-loopback access. It also rejects unexpected browser origins, requires a custom browser header, and rate-limits the paid voice pipeline. Do not expose it directly to a network; a shared deployment needs a separately designed TLS and authentication layer.

## Optional local microphone CLI

The original terminal assistant remains available as `pi-assistant`. Only that version needs PyAudio:

```bash
# macOS
brew install portaudio
uv sync --extra audio

# Raspberry Pi OS / Debian
sudo apt install -y build-essential portaudio19-dev alsa-utils
uv sync --extra audio
```

Run it with:

```bash
uv run pi-assistant --list-microphones
uv run pi-assistant
```

## Tests and checks

```bash
uv run python -m unittest discover -s tests -v
cd frontend
npm test
npm run check
npm run build
```

Backend tests mock OpenAI and do not make paid API calls. Run `npm run build` before `uv build` so the generated Astro assets can be included in the Python wheel.

## Security note

The previous scripts contained Google API keys directly in source. They have been removed from current files, but remain in Git history. Revoke those keys immediately and purge the repository history before making it public.
