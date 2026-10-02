export function createCameraAdapter({ mediaDevices = globalThis.navigator?.mediaDevices } = {}) {
  const listeners = new Set();
  const trackListeners = new Map();
  let stream = null;
  let closed = false;
  let snapshot = { status: "stopped", error: "", notice: "", stream: null };

  const handleDeviceChange = () => update({ notice: "Camera devices changed" });
  mediaDevices?.addEventListener?.("devicechange", handleDeviceChange);

  async function open() {
    ensureOpen();
    if (!mediaDevices?.getUserMedia) {
      const error = "Camera access is not supported in this browser";
      update({ status: "error", error, notice: "", stream: null });
      return { ok: false, error };
    }
    if (stream) return { ok: true, stream };

    update({ status: "opening", error: "", notice: "", stream: null });
    try {
      const acquiredStream = await mediaDevices.getUserMedia({ video: true, audio: false });
      if (closed) {
        for (const track of acquiredStream.getTracks()) track.stop();
        return { ok: false, error: "Camera request was cancelled" };
      }
      stream = acquiredStream;
      for (const track of stream.getTracks()) {
        const ended = () => handleUnexpectedEnd();
        track.addEventListener?.("ended", ended);
        trackListeners.set(track, ended);
      }
      update({ status: "open", error: "", notice: "", stream });
      return { ok: true, stream };
    } catch (cause) {
      if (closed) return { ok: false, error: "Camera request was cancelled" };
      const error = cause instanceof Error ? cause.message : String(cause);
      stream = null;
      update({ status: "error", error, notice: "", stream: null });
      return { ok: false, error };
    }
  }

  function stop() {
    ensureOpen();
    if (!stream) {
      if (snapshot.status !== "error") update({ status: "stopped", stream: null });
      return;
    }
    releaseStream();
    update({ status: "stopped", error: "", notice: "", stream: null });
  }

  function handleUnexpectedEnd() {
    if (!stream || closed) return;
    releaseStream();
    update({
      status: "error",
      error: "Camera stream ended unexpectedly",
      notice: "",
      stream: null,
    });
  }

  function releaseStream() {
    const activeStream = stream;
    stream = null;
    for (const track of activeStream?.getTracks?.() ?? []) {
      const ended = trackListeners.get(track);
      if (ended) track.removeEventListener?.("ended", ended);
      trackListeners.delete(track);
      track.stop();
    }
  }

  function subscribe(listener) {
    ensureOpen();
    if (typeof listener !== "function") throw new Error("Camera listener must be a function");
    listeners.add(listener);
    listener(getSnapshot());
    return () => listeners.delete(listener);
  }

  function getSnapshot() {
    return Object.freeze({ ...snapshot });
  }

  function close() {
    if (closed) return;
    if (stream) releaseStream();
    mediaDevices?.removeEventListener?.("devicechange", handleDeviceChange);
    closed = true;
    snapshot = { status: "closed", error: "", notice: "", stream: null };
    listeners.clear();
  }

  function update(changes) {
    snapshot = { ...snapshot, ...changes };
    const next = getSnapshot();
    for (const listener of listeners) listener(next);
  }

  function ensureOpen() {
    if (closed) throw new Error("Camera adapter was closed");
  }

  return Object.freeze({ open, stop, subscribe, getSnapshot, close });
}
