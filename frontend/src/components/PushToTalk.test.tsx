import { cleanup, fireEvent, render, waitFor } from "@solidjs/testing-library";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import PushToTalk from "./PushToTalk";

class FakeMediaRecorder extends EventTarget {
  static isTypeSupported = () => true;
  state: RecordingState = "inactive";
  mimeType = "audio/webm;codecs=opus";

  constructor(_stream: MediaStream, options?: MediaRecorderOptions) {
    super();
    if (options?.mimeType) this.mimeType = options.mimeType;
  }

  start() {
    this.state = "recording";
  }

  stop() {
    if (this.state !== "recording") throw new DOMException("Inactive recorder");
    this.state = "inactive";
    const dataEvent = new Event("dataavailable");
    Object.defineProperty(dataEvent, "data", {
      value: new Blob(["audio"], { type: this.mimeType }),
    });
    this.dispatchEvent(dataEvent);
    this.dispatchEvent(new Event("stop"));
  }
}

class FakeAudio extends EventTarget {
  pause = vi.fn();
  play = vi.fn().mockResolvedValue(undefined);

  constructor(_source: string) {
    super();
  }
}

const track = { stop: vi.fn() };
const stream = { getTracks: () => [track] } as unknown as MediaStream;
const getUserMedia = vi.fn().mockResolvedValue(stream);

beforeEach(() => {
  track.stop.mockClear();
  getUserMedia.mockClear();
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia },
  });
  vi.stubGlobal("MediaRecorder", FakeMediaRecorder);
  vi.stubGlobal("Audio", FakeAudio);
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      new Response(new Blob(["wav"], { type: "audio/wav" }), {
        status: 200,
        headers: {
          "X-User-Text": encodeURIComponent("Olá, assistente."),
          "X-Assistant-Text": encodeURIComponent("Olá! Como posso ajudar?"),
        },
      }),
    ),
  );
  vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:reply");
  vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => undefined);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("PushToTalk", () => {
  it("supports synthesized click activation and sends the recording", async () => {
    const view = render(() => <PushToTalk />);
    const button = view.getByRole("button", { name: "Prima para falar" });

    fireEvent.click(button);
    await waitFor(() =>
      expect(view.getByRole("button").textContent).toBe("Solte para enviar"),
    );
    fireEvent.click(button);

    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));
    const request = vi.mocked(fetch).mock.calls[0][1];
    expect(request?.headers).toMatchObject({ "X-Pocket-Assistant": "web" });
    await waitFor(() =>
      expect(view.getByRole("button").textContent).toBe("A reproduzir…"),
    );
    expect(track.stop).toHaveBeenCalled();
    const conversation = view.getByRole("region", { name: "Conversa" });
    expect(conversation.textContent).toContain("Utilizador: Olá, assistente.");
    expect(conversation.textContent).toContain(
      "Assistente: Olá! Como posso ajudar?",
    );
  });

  it("does not submit a recording after the component unmounts", async () => {
    const view = render(() => <PushToTalk />);
    fireEvent.click(view.getByRole("button"));
    await waitFor(() =>
      expect(view.getByRole("button").textContent).toBe("Solte para enviar"),
    );

    view.unmount();

    expect(fetch).not.toHaveBeenCalled();
    expect(track.stop).toHaveBeenCalled();
  });

  it("aborts processing and suppresses playback after unmount", async () => {
    let resolveBlob!: (blob: Blob) => void;
    const blob = new Promise<Blob>((resolve) => {
      resolveBlob = resolve;
    });
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      headers: new Headers(),
      blob: () => blob,
    } as Response);
    const view = render(() => <PushToTalk />);

    fireEvent.click(view.getByRole("button"));
    await waitFor(() =>
      expect(view.getByRole("button").textContent).toBe("Solte para enviar"),
    );
    fireEvent.click(view.getByRole("button"));
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));
    const signal = vi.mocked(fetch).mock.calls[0][1]?.signal;

    view.unmount();
    resolveBlob(new Blob(["wav"], { type: "audio/wav" }));
    await blob;

    expect(signal?.aborted).toBe(true);
    expect(URL.createObjectURL).not.toHaveBeenCalled();
  });
});
