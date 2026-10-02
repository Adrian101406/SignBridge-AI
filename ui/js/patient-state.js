export function createPatientState({ minimumConfidence = 0.8 } = {}) {
  if (!Number.isFinite(minimumConfidence) || minimumConfidence < 0 || minimumConfidence > 1) {
    throw new Error("Minimum confidence must be between zero and one");
  }

  const listeners = new Set();
  const history = [];
  let closed = false;
  let snapshot = {
    draft: "",
    currentLabel: "",
    confidence: null,
    status: "Recognition is stopped",
    sending: false,
    error: "",
    originalBim: "",
    proposalReady: false,
    medicalFlags: [],
    candidatePending: false,
    candidateDomain: "",
  };

  function observe({ label, confidence, recognizerType, recognizer_type: recognizerTypeSnake }) {
    ensureOpen();
    const cleanLabel = typeof label === "string" ? label.trim() : "";
    const cleanConfidence = Number.isFinite(confidence) ? confidence : null;
    const candidateDomain = String(recognizerType || recognizerTypeSnake || "").trim().toLowerCase();

    if (!cleanLabel) {
      update({
        currentLabel: "",
        confidence: cleanConfidence,
        candidatePending: false,
        candidateDomain: "",
        error: "",
        status: "Hold the sign steady",
      });
      return false;
    }
    update({
      currentLabel: cleanLabel,
      confidence: cleanConfidence,
      candidatePending: true,
      candidateDomain,
      error: "",
      status: `Check ${cleanLabel}, then confirm or try again`,
    });
    return true;
  }

  function confirmCandidate() {
    ensureOpen();
    if (!snapshot.candidatePending || !snapshot.currentLabel) return false;
    history.push(snapshot.draft);
    const prefix = snapshot.draft.trimEnd();
    const confirmedLabel = snapshot.currentLabel;
    update({
      draft: prefix ? `${prefix} ${confirmedLabel}` : confirmedLabel,
      currentLabel: "",
      confidence: null,
      candidatePending: false,
      candidateDomain: "",
      status: `${confirmedLabel} added to your sentence`,
      originalBim: "",
      proposalReady: false,
      medicalFlags: [],
    });
    return true;
  }

  function retryCandidate() {
    ensureOpen();
    if (!snapshot.candidatePending) return false;
    update({
      currentLabel: "",
      confidence: null,
      candidatePending: false,
      candidateDomain: "",
      status: "Get ready to try the sign again",
      error: "",
    });
    return true;
  }

  function release() {
    ensureOpen();
    update({
      currentLabel: "",
      confidence: null,
      candidatePending: false,
      candidateDomain: "",
      status: "Ready for the next sign",
    });
  }

  function setDraft(text) {
    ensureOpen();
    history.length = 0;
    update({ draft: typeof text === "string" ? text : "", error: "" });
  }

  function undoLastWord() {
    ensureOpen();
    if (history.length === 0) return false;
    update({ draft: history.pop(), error: "" });
    return true;
  }

  function clearDraft() {
    ensureOpen();
    history.length = 0;
    update({
      draft: "",
      currentLabel: "",
      confidence: null,
      candidatePending: false,
      candidateDomain: "",
      error: "",
      sending: false,
      originalBim: "",
      proposalReady: false,
      medicalFlags: [],
    });
  }

  function markSending() {
    ensureOpen();
    if (!snapshot.draft.trim()) throw new Error("Patient draft cannot be empty");
    update({ sending: true, error: "" });
  }

  function markSendFailed(message) {
    ensureOpen();
    update({ sending: false, error: String(message || "Message was not delivered") });
  }

  function cancelPending() {
    ensureOpen();
    update({ sending: false, error: "" });
  }

  function applyProposal(sourceText, suggestedText, medicalFlags = []) {
    ensureOpen();
    const source = typeof sourceText === "string" ? sourceText.trim() : "";
    const suggestion = typeof suggestedText === "string" ? suggestedText.trim() : "";
    if (!source || !suggestion) throw new Error("Medical proposal requires source and suggested text");
    history.length = 0;
    update({
      draft: suggestion,
      originalBim: source,
      proposalReady: true,
      medicalFlags: Array.isArray(medicalFlags) ? medicalFlags.map(String) : [],
      sending: false,
      error: "",
      status: "Medical sentence ready for review",
    });
  }

  function markSendSucceeded() {
    ensureOpen();
    history.length = 0;
    update({
      draft: "",
      currentLabel: "",
      confidence: null,
      candidatePending: false,
      candidateDomain: "",
      sending: false,
      error: "",
      originalBim: "",
      proposalReady: false,
      medicalFlags: [],
      status: "Message sent to the doctor",
    });
  }

  function subscribe(listener) {
    ensureOpen();
    if (typeof listener !== "function") throw new Error("Patient state listener must be a function");
    listeners.add(listener);
    listener(getSnapshot());
    return () => listeners.delete(listener);
  }

  function getSnapshot() {
    return Object.freeze({ ...snapshot });
  }

  function close() {
    if (closed) return;
    closed = true;
    listeners.clear();
    history.length = 0;
  }

  function update(changes) {
    snapshot = { ...snapshot, ...changes };
    const next = getSnapshot();
    for (const listener of listeners) listener(next);
  }

  function ensureOpen() {
    if (closed) throw new Error("Patient state was closed");
  }

  return Object.freeze({
    observe,
    confirmCandidate,
    retryCandidate,
    release,
    setDraft,
    undoLastWord,
    clearDraft,
    markSending,
    markSendFailed,
    cancelPending,
    applyProposal,
    markSendSucceeded,
    getSnapshot,
    subscribe,
    close,
  });
}
