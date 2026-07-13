import { For, createMemo, createSignal, onCleanup } from "solid-js";

type ConversationMessage = {
  role: "assistant" | "user";
  text: string;
};

type AssistantState =
  | "idle"
  | "requesting"
  | "recording"
  | "processing"
  | "playing"
  | "error";

const labels: Record<AssistantState, string> = {
  idle: "Prima para falar",
  requesting: "Permita o microfone",
  recording: "Solte para enviar",
  processing: "A pensar…",
  playing: "A reproduzir…",
  error: "Tentar novamente",
};

function recorderOptions(): MediaRecorderOptions | undefined {
  const types = [
    "audio/webm;codecs=opus",
    "audio/mp4",
    "audio/webm",
    "audio/ogg;codecs=opus",
  ];
  const mimeType = types.find((type) => MediaRecorder.isTypeSupported(type));
  return mimeType ? { mimeType } : undefined;
}

function decodeTextHeader(value: string | null): string {
  if (!value) return "";
  try {
    return decodeURIComponent(value);
  } catch {
    return "";
  }
}

function recordingFilename(mimeType: string): string {
  if (mimeType.includes("mp4")) return "recording.mp4";
  if (mimeType.includes("ogg")) return "recording.ogg";
  if (mimeType.includes("wav")) return "recording.wav";
  return "recording.webm";
}

export default function PushToTalk() {
  const [state, setState] = createSignal<AssistantState>("idle");
  const [errorMessage, setErrorMessage] = createSignal("");
  const [messages, setMessages] = createSignal<ConversationMessage[]>([]);
  const label = createMemo(() => labels[state()]);
  const statusMessage = createMemo(() => errorMessage() || label());

  let mediaRecorder: MediaRecorder | undefined;
  let activeStream: MediaStream | undefined;
  let playback: HTMLAudioElement | undefined;
  let playbackUrl: string | undefined;
  let requestController: AbortController | undefined;
  let releaseRequested = false;
  let discardRecording = false;
  let suppressNextClick = false;
  let disposed = false;
  let conversationId = "";

  const stopStream = () => {
    activeStream?.getTracks().forEach((track) => track.stop());
    activeStream = undefined;
  };

  const clearPlayback = () => {
    playback?.pause();
    playback = undefined;
    if (playbackUrl) URL.revokeObjectURL(playbackUrl);
    playbackUrl = undefined;
  };

  const fail = (error: unknown) => {
    console.error(error);
    stopStream();
    clearPlayback();
    if (disposed) return;
    setErrorMessage(error instanceof Error ? error.message : "O pedido falhou.");
    setState("error");
  };

  const submitRecording = async (recording: Blob) => {
    if (disposed) return;
    if (!recording.size) {
      fail(new Error("A gravação está vazia."));
      return;
    }

    requestController = new AbortController();
    try {
      if (!conversationId) {
        conversationId = crypto.randomUUID();
      }
      const body = new FormData();
      body.append("audio", recording, recordingFilename(recording.type));
      body.append("conversation_id", conversationId);

      const headers = { "X-Pocket-Assistant": "web" };
      const response = await fetch("/api/voice", {
        method: "POST",
        body,
        headers,
        signal: requestController.signal,
      });
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as {
          error?: string;
        } | null;
        throw new Error(payload?.error ?? "O pedido de voz falhou.");
      }

      const userText = decodeTextHeader(response.headers.get("X-User-Text"));
      const assistantText = decodeTextHeader(
        response.headers.get("X-Assistant-Text"),
      );
      const reply = await response.blob();
      if (disposed || requestController.signal.aborted) return;
      if (userText && assistantText) {
        setMessages((current) => [
          ...current,
          { role: "user", text: userText },
          { role: "assistant", text: assistantText },
        ]);
      }
      clearPlayback();
      playbackUrl = URL.createObjectURL(reply);
      playback = new Audio(playbackUrl);
      playback.addEventListener(
        "ended",
        () => {
          clearPlayback();
          if (!disposed) setState("idle");
        },
        { once: true },
      );
      playback.addEventListener("error", () => fail(new Error("A reprodução falhou.")), {
        once: true,
      });
      setState("playing");
      await playback.play();
    } catch (error) {
      if (!disposed && !(error instanceof DOMException && error.name === "AbortError")) {
        fail(error);
      }
    } finally {
      requestController = undefined;
    }
  };

  const startRecording = async () => {
    if (state() !== "idle" && state() !== "error") return;
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      fail(new Error("Este navegador não permite gravar através do microfone."));
      return;
    }

    releaseRequested = false;
    discardRecording = false;
    setErrorMessage("");
    setState("requesting");
    try {
      activeStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (disposed || releaseRequested) {
        stopStream();
        setState("idle");
        return;
      }

      const chunks: BlobPart[] = [];
      const recorder = new MediaRecorder(activeStream, recorderOptions());
      mediaRecorder = recorder;
      recorder.addEventListener("dataavailable", (event) => {
        if (event.data.size) chunks.push(event.data);
      });
      recorder.addEventListener("error", (event) => {
        discardRecording = true;
        fail(event.error);
      });
      recorder.addEventListener(
        "stop",
        () => {
          mediaRecorder = undefined;
          stopStream();
          if (disposed || discardRecording) return;
          void submitRecording(
            new Blob(chunks, { type: recorder.mimeType || "audio/webm" }),
          );
        },
        { once: true },
      );
      recorder.start();
      setState("recording");
    } catch (error) {
      fail(error);
    }
  };

  const stopRecording = () => {
    releaseRequested = true;
    const recorder = mediaRecorder;
    if (state() !== "recording" || recorder?.state !== "recording") return;
    setState("processing");
    try {
      recorder.stop();
    } catch (error) {
      discardRecording = true;
      fail(error);
    }
  };

  const toggleRecording = () => {
    if (state() === "recording" || state() === "requesting") {
      stopRecording();
    } else {
      void startRecording();
    }
  };

  onCleanup(() => {
    disposed = true;
    discardRecording = true;
    requestController?.abort();
    if (mediaRecorder?.state === "recording") {
      try {
        mediaRecorder.stop();
      } catch {
        // The recorder may have become inactive between the state check and stop.
      }
    }
    stopStream();
    clearPlayback();
  });

  return (
    <div class="assistant-layout">
      <section class="assistant-controls">
        <h1>Pocket Assistant</h1>
        <button
          type="button"
          class="talk-button"
          data-state={state()}
          disabled={state() === "processing" || state() === "playing"}
          aria-label={label()}
          onPointerDown={(event) => {
            event.preventDefault();
            suppressNextClick = true;
            event.currentTarget.setPointerCapture(event.pointerId);
            void startRecording();
          }}
          onPointerUp={(event) => {
            event.preventDefault();
            stopRecording();
          }}
          onPointerCancel={stopRecording}
          onKeyDown={(event) => {
            if (!event.repeat && (event.key === " " || event.key === "Enter")) {
              event.preventDefault();
              suppressNextClick = true;
              void startRecording();
            }
          }}
          onKeyUp={(event) => {
            if (event.key === " " || event.key === "Enter") {
              event.preventDefault();
              stopRecording();
            }
          }}
          onClick={() => {
            if (suppressNextClick) {
              suppressNextClick = false;
              return;
            }
            toggleRecording();
          }}
          onContextMenu={(event) => event.preventDefault()}
        >
          {label()}
        </button>
        <span class="visually-hidden" role="status" aria-live="polite">
          {statusMessage()}
        </span>
      </section>

      <section class="conversation" aria-label="Conversa" aria-live="polite">
        <For each={messages()}>
          {(message) => (
            <p class="conversation-message">
              <strong>
                {message.role === "assistant" ? "Assistente:" : "Utilizador:"}
              </strong>{" "}
              {message.text}
            </p>
          )}
        </For>
      </section>
    </div>
  );
}
