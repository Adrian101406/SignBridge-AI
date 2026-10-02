const DEMO_LABELS = ["HELLO", "YES", "NO"];


export function createMockRecognitionAdapter({
  schedule = (callback) => setInterval(callback, 1800),
  cancelSchedule = (id) => clearInterval(id),
} = {}) {
  const listeners = new Set();
  let timer = null;
  let index = 0;
  let running = false;
  let mode = "default";

  function start() {
    if (running) return;
    running = true;
    emit({ type: "status", status: "DEMO — NOT AI", demo: true });
    timer = schedule(() => {
      const label = DEMO_LABELS[index % DEMO_LABELS.length];
      index += 1;
      emit({ type: "observation", label, confidence: 0.93, stable: true, demo: true });
    });
  }

  function stop() {
    if (!running) return;
    running = false;
    if (timer !== null) cancelSchedule(timer);
    timer = null;
    emit({ type: "status", status: "Demo recognition stopped", demo: true });
  }

  function subscribe(listener) {
    if (typeof listener !== "function") throw new Error("Recognition listener must be a function");
    listeners.add(listener);
    return () => listeners.delete(listener);
  }

  function close() {
    stop();
    listeners.clear();
  }

  function emit(event) {
    const frozenEvent = Object.freeze({ ...event });
    for (const listener of listeners) listener(frozenEvent);
  }

  function setMode(nextMode) { mode = normalizeMode(nextMode); }
  function continueAfterDecision() {}

  return Object.freeze({ start, stop, setMode, continueAfterDecision, subscribe, close, isRunning: () => running, getMode: () => mode });
}


export function createBackendRecognitionAdapter({
  backend,
  getStream,
  MediaRecorderImpl = globalThis.MediaRecorder,
  clipMs = 5000,
  readyMs = 5000,
  now = () => globalThis.performance?.now?.() ?? Date.now(),
  scheduleTimeout = (callback, delay) => setTimeout(callback, delay),
  cancelTimeout = (id) => clearTimeout(id),
  scheduleInterval = (callback, delay) => setInterval(callback, delay),
  cancelInterval = (id) => clearInterval(id),
  progressIntervalMs = 200,
} = {}) {
  const listeners = new Set();
  let running = false;
  let recorder = null;
  let recordingTimer = null;
  let readyTimer = null;
  let progressTimer = null;
  let generation = 0;
  let captureNumber = 0;
  let mode = "default";
  let pendingContinuation = null;

  function emit(event) { for (const listener of listeners) listener(Object.freeze({ ...event })); }

  function start() {
    if (running) return;
    const stream = getStream?.();
    if (!stream) { emit({ type: "status", status: "Open the camera before starting recognition" }); return; }
    const mimeType = selectMimeType(MediaRecorderImpl);
    if (!MediaRecorderImpl || !mimeType) { emit({ type: "status", status: "No supported WebM recorder is available" }); return; }
    running = true;
    generation += 1;
    captureNumber = 0;
    capture(stream, mimeType, generation);
  }

  function capture(stream, mimeType, runGeneration) {
    if (!running) return;
    captureNumber += 1;
    const captureMode = mode;
    const chunks = [];
    recorder = new MediaRecorderImpl(stream, { mimeType });
    recorder.addEventListener("dataavailable", (event) => { if (event.data?.size) chunks.push(event.data); });
    recorder.addEventListener("stop", async () => {
      if (!running) return;
      clearRecordingTimer();
      clearProgressTimer();
      const processingStartedAt = now();
      emitProcessingStatus(processingStartedAt);
      progressTimer = scheduleInterval(() => emitProcessingStatus(processingStartedAt), progressIntervalMs);
      try {
        const result = await backend.recognize(new Blob(chunks, { type: mimeType.split(";", 1)[0] }), captureMode);
        if (!running || generation !== runGeneration) return;
        clearProgressTimer();
        pendingContinuation = { stream, mimeType, runGeneration };
        emit({
          type: "observation",
          label: result.gloss,
          confidence: result.confidence,
          stable: result.stable,
          recognizerType: result.recognizer_type || captureMode,
        });
      } catch (error) {
        if (!running || generation !== runGeneration) return;
        clearProgressTimer();
        emit({ type: "status", status: error.message });
        if (running && generation === runGeneration) prepareNextCapture(stream, mimeType, runGeneration);
      }
    }, { once: true });
    recorder.start();
    const recordingStartedAt = now();
    emitRecordingStatus(recordingStartedAt);
    progressTimer = scheduleInterval(() => emitRecordingStatus(recordingStartedAt), progressIntervalMs);
    recordingTimer = scheduleTimeout(() => recorder?.state === "recording" && recorder.stop(), clipMs);
  }

  function stop() {
    running = false;
    generation += 1;
    clearRecordingTimer();
    clearReadyTimer();
    clearProgressTimer();
    if (recorder?.state === "recording") recorder.stop();
    recorder = null;
    pendingContinuation = null;
    emit({ type: "status", status: "Recognition stopped" });
  }

  function setMode(nextMode) {
    mode = normalizeMode(nextMode);
  }

  function continueAfterDecision() {
    const continuation = pendingContinuation;
    if (!running || !continuation || continuation.runGeneration !== generation) return false;
    pendingContinuation = null;
    prepareNextCapture(continuation.stream, continuation.mimeType, continuation.runGeneration);
    return true;
  }

  function emitRecordingStatus(startedAt) {
    const prompt = captureNumber === 1 ? "Show your sign now" : "Show next sign now";
    const remainingMs = Math.max(0, clipMs - (now() - startedAt));
    emit({ type: "status", status: `${prompt} · Recording: ${formatSeconds(remainingMs)}s remaining` });
  }

  function emitProcessingStatus(startedAt) {
    const elapsedMs = Math.max(0, now() - startedAt);
    emit({ type: "status", status: `Processing sign: ${formatSeconds(elapsedMs)}s · Please wait` });
  }

  function prepareNextCapture(stream, mimeType, runGeneration) {
    const readyStartedAt = now();
    emitReadyStatus(readyStartedAt);
    progressTimer = scheduleInterval(() => emitReadyStatus(readyStartedAt), 1000);
    readyTimer = scheduleTimeout(() => {
      readyTimer = null;
      clearProgressTimer();
      if (running && generation === runGeneration) {
        emit({ type: "release" });
        capture(stream, mimeType, runGeneration);
      }
    }, readyMs);
  }

  function emitReadyStatus(startedAt) {
    const secondsRemaining = Math.ceil((readyMs - (now() - startedAt)) / 1000);
    if (secondsRemaining <= 0) return;
    emit({ type: "status", status: `Get ready for the next sign · Starting in ${secondsRemaining}s` });
  }

  function clearRecordingTimer() {
    if (recordingTimer !== null) cancelTimeout(recordingTimer);
    recordingTimer = null;
  }

  function clearReadyTimer() {
    if (readyTimer !== null) cancelTimeout(readyTimer);
    readyTimer = null;
  }

  function clearProgressTimer() {
    if (progressTimer !== null) cancelInterval(progressTimer);
    progressTimer = null;
  }

  async function recognizeFile(file, requestedMode = mode) {
    if (!(file instanceof Blob) || !["video/mp4", "video/webm"].includes(file.type)) {
      emit({ type: "status", status: "Choose an MP4 or WebM video" });
      return null;
    }
    if (running) stop();
    const runGeneration = ++generation;
    emit({ type: "status", status: "Processing uploaded video…" });
    try {
      const selectedMode = normalizeMode(requestedMode);
      const result = await backend.recognize(file, selectedMode);
      if (generation !== runGeneration) return null;
      emit({
        type: "observation",
        label: result.gloss,
        confidence: result.confidence,
        stable: result.stable,
        recognizerType: result.recognizer_type || selectedMode,
      });
      emit({ type: "status", status: "Uploaded video recognition complete" });
      return result;
    } catch (error) {
      if (generation !== runGeneration) return null;
      emit({ type: "status", status: error.message });
      return null;
    }
  }

  function subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); }
  function close() { stop(); listeners.clear(); }
  return Object.freeze({
    start,
    stop,
    setMode,
    continueAfterDecision,
    recognizeFile,
    subscribe,
    close,
    isRunning: () => running,
    getMode: () => mode,
  });
}


function normalizeMode(mode) {
  if (mode === "default" || mode === "number") return mode;
  throw new Error("Recognition mode must be default or number");
}


function selectMimeType(MediaRecorderImpl) {
  if (!MediaRecorderImpl) return "";
  const options = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
  if (typeof MediaRecorderImpl.isTypeSupported !== "function") return "video/webm";
  return options.find((type) => MediaRecorderImpl.isTypeSupported(type)) || "";
}


function formatSeconds(milliseconds) {
  return (milliseconds / 1000).toFixed(1);
}
