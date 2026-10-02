export function createDoctorState() {
  const listeners = new Set();
  let closed = false;
  let requestCounter = 0;
  let draftRevision = 0;
  let requestContext = null;
  let snapshot = {
    originalTranscript: "",
    draft: "",
    proposedText: "",
    warnings: [],
    latestPatientMessage: "",
    activeRequestId: "",
    status: "Ready to record",
    sending: false,
    error: "",
  };

  function beginAsr() {
    ensureOpen();
    requestCounter += 1;
    const id = `asr-${requestCounter}`;
    requestContext = {
      id,
      draftRevision,
      originalTranscript: snapshot.originalTranscript,
      draft: snapshot.draft,
    };
    update({ activeRequestId: id, status: "Recording…", error: "" });
    return Object.freeze({ id });
  }

  function receiveAsr(requestId, transcript) {
    ensureOpen();
    if (!isActive(requestId)) return false;
    const incomingTranscript = typeof transcript === "string" ? transcript : "";
    const originalTranscript = appendSegment(requestContext.originalTranscript, incomingTranscript);
    const changes = { originalTranscript, status: "ASR transcript ready for review", error: "" };
    if (draftRevision === requestContext.draftRevision) {
      changes.draft = appendSegment(requestContext.draft, incomingTranscript);
      draftRevision += 1;
    }
    update(changes);
    return true;
  }

  function failAsr(requestId, message) {
    ensureOpen();
    if (!isActive(requestId)) return false;
    update({ status: "ASR failed", error: String(message || "ASR failed") });
    return true;
  }

  function receiveProposal(requestId, proposedText, warnings = []) {
    ensureOpen();
    if (!isActive(requestId)) return false;
    update({
      proposedText: typeof proposedText === "string" ? proposedText : "",
      warnings: Array.isArray(warnings) ? warnings.map(String) : [],
    });
    return true;
  }

  function editDraft(text) {
    ensureOpen();
    draftRevision += 1;
    update({ draft: typeof text === "string" ? text : "", error: "" });
  }

  function cancel() {
    ensureOpen();
    requestContext = null;
    draftRevision += 1;
    update({
      originalTranscript: "",
      draft: "",
      proposedText: "",
      warnings: [],
      activeRequestId: "",
      status: "Draft cancelled",
      sending: false,
      error: "",
    });
  }

  function beginConfirm() {
    ensureOpen();
    const confirmedText = snapshot.draft.trim();
    if (!confirmedText) throw new Error("Doctor draft cannot be empty");
    update({ sending: true, status: "Sending confirmed message…", error: "" });
    return confirmedText;
  }

  function confirmFailed(message) {
    ensureOpen();
    update({ sending: false, status: "Message not delivered", error: String(message || "Message was not delivered") });
  }

  function confirmSucceeded() {
    ensureOpen();
    requestContext = null;
    draftRevision += 1;
    update({
      originalTranscript: "",
      draft: "",
      proposedText: "",
      warnings: [],
      activeRequestId: "",
      status: "Confirmed message sent",
      sending: false,
      error: "",
    });
  }

  function receivePatientMessage(message) {
    ensureOpen();
    if (!message || message.sender !== "patient" || typeof message.text !== "string") return false;
    update({ latestPatientMessage: message.text });
    return true;
  }

  function subscribe(listener) {
    ensureOpen();
    if (typeof listener !== "function") throw new Error("Doctor state listener must be a function");
    listeners.add(listener);
    listener(getSnapshot());
    return () => listeners.delete(listener);
  }

  function getSnapshot() {
    return Object.freeze({ ...snapshot, warnings: Object.freeze([...snapshot.warnings]) });
  }

  function close() {
    if (closed) return;
    closed = true;
    requestContext = null;
    listeners.clear();
  }

  function isActive(requestId) {
    return Boolean(requestContext && requestContext.id === requestId);
  }

  function update(changes) {
    snapshot = { ...snapshot, ...changes };
    const next = getSnapshot();
    for (const listener of listeners) listener(next);
  }

  function ensureOpen() {
    if (closed) throw new Error("Doctor state was closed");
  }

  return Object.freeze({
    beginAsr,
    receiveAsr,
    failAsr,
    receiveProposal,
    editDraft,
    cancel,
    beginConfirm,
    confirmFailed,
    confirmSucceeded,
    receivePatientMessage,
    getSnapshot,
    subscribe,
    close,
  });
}


function appendSegment(existing, incoming) {
  const before = String(existing || "").trim();
  const next = String(incoming || "").trim();
  if (!before) return next;
  if (!next) return before;
  return `${before}\n${next}`;
}
