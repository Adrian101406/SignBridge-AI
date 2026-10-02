export function createMockAsrAdapter({
  schedule = (callback) => setTimeout(callback, 650),
  cancelSchedule = (id) => clearTimeout(id),
} = {}) {
  const listeners = new Set();
  let activeRequestId = "";
  let timer = null;
  let recording = false;
  let generation = 0;

  function start(requestId) {
    if (typeof requestId !== "string" || !requestId) throw new Error("ASR request ID is required");
    cancel();
    activeRequestId = requestId;
    recording = true;
    emit({ type: "status", requestId, status: "DEMO — NOT REAL ASR · Recording", demo: true });
  }

  function stop() {
    if (!recording || !activeRequestId) return;
    recording = false;
    const requestId = activeRequestId;
    const runGeneration = generation;
    emit({ type: "status", requestId, status: "DEMO — NOT REAL ASR · Processing", demo: true });
    timer = schedule(() => {
      if (generation !== runGeneration || activeRequestId !== requestId) return;
      timer = null;
      emit({
        type: "transcript",
        requestId,
        transcript: "Hello, this is a demonstration.",
        demo: true,
      });
    });
  }

  function cancel() {
    generation += 1;
    recording = false;
    activeRequestId = "";
    if (timer !== null) cancelSchedule(timer);
    timer = null;
  }

  function subscribe(listener) {
    if (typeof listener !== "function") throw new Error("ASR listener must be a function");
    listeners.add(listener);
    return () => listeners.delete(listener);
  }

  function close() {
    cancel();
    listeners.clear();
  }

  function emit(event) {
    const frozenEvent = Object.freeze({ ...event });
    for (const listener of listeners) listener(frozenEvent);
  }

  return Object.freeze({ start, stop, cancel, subscribe, close });
}


export function createBackendAsrAdapter({ backend, mediaDevices = navigator.mediaDevices, AudioContextImpl = globalThis.AudioContext || globalThis.webkitAudioContext } = {}) {
  const listeners = new Set();
  let requestId = "";
  let context;
  let processor;
  let source;
  let stream;
  let samples = [];
  let generation = 0;
  let stopRequested = false;
  const emit = (event) => { for (const listener of listeners) listener(Object.freeze({ ...event })); };

  async function start(id) {
    cancel();
    requestId = id;
    const run = generation;
    stopRequested = false;
    try {
      const acquired = await mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true }, video: false });
      if (generation !== run || requestId !== id) { acquired.getTracks().forEach((track) => track.stop()); return; }
      stream = acquired;
      context = new AudioContextImpl();
      source = context.createMediaStreamSource(stream);
      processor = context.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (event) => samples.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      source.connect(processor);
      processor.connect(context.destination);
      emit({ type: "status", requestId, status: "Recording…" });
      if (stopRequested) await stop();
    } catch (error) {
      cleanup();
      emit({ type: "error", requestId: id, error: error.message });
    }
  }

  async function stop() {
    if (!requestId) return;
    if (!context) { stopRequested = true; return; }
    const id = requestId;
    const run = generation;
    const rate = context.sampleRate;
    const captured = samples;
    cleanup();
    if (!captured.some((chunk) => chunk.length > 0)) {
      emit({ type: "error", requestId: id, error: "No microphone audio was captured" });
      return;
    }
    emit({ type: "status", requestId: id, status: "Transcribing and checking medical wording…" });
    try {
      const result = await backend.transcribe(encodeWav(captured, rate));
      if (generation !== run || requestId !== id) return;
      emit({ type: "transcript", requestId: id, transcript: result.transcript.original_text, warnings: result.transcript.warnings });
      emit({
        type: "proposal",
        requestId: id,
        proposedText: result.medical_proposal.suggested_text,
        warnings: [...result.medical_proposal.medical_flags, ...result.medical_proposal.changed_critical_terms],
      });
    } catch (error) {
      emit({ type: "error", requestId: id, error: error.message });
    }
  }

  function cleanup() {
    processor?.disconnect(); source?.disconnect(); stream?.getTracks?.().forEach((track) => track.stop());
    context?.close?.(); processor = null; source = null; stream = null; context = null; samples = [];
  }
  function cancel() { generation += 1; stopRequested = false; cleanup(); requestId = ""; }
  function subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); }
  function close() { cancel(); listeners.clear(); }
  return Object.freeze({ start, stop, cancel, subscribe, close });
}


function encodeWav(chunks, sourceRate) {
  const input = new Float32Array(chunks.reduce((sum, chunk) => sum + chunk.length, 0));
  let offset = 0;
  for (const chunk of chunks) { input.set(chunk, offset); offset += chunk.length; }
  const ratio = sourceRate / 16000;
  const output = new Float32Array(Math.floor(input.length / ratio));
  for (let i = 0; i < output.length; i += 1) output[i] = input[Math.floor(i * ratio)] || 0;
  const buffer = new ArrayBuffer(44 + output.length * 2);
  const view = new DataView(buffer);
  const write = (at, text) => [...text].forEach((char, index) => view.setUint8(at + index, char.charCodeAt(0)));
  write(0, "RIFF"); view.setUint32(4, 36 + output.length * 2, true); write(8, "WAVEfmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, 16000, true); view.setUint32(28, 32000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  write(36, "data"); view.setUint32(40, output.length * 2, true);
  output.forEach((sample, index) => view.setInt16(44 + index * 2, Math.max(-1, Math.min(1, sample)) * 0x7fff, true));
  return new Blob([buffer], { type: "audio/wav" });
}
