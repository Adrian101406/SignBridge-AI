import { createBackendAsrAdapter } from "./adapters/asr-adapter.js";
import { createBackendClient, createBackendSession } from "./adapters/backend-client.js?v=3";
import { createDoctorState } from "./doctor-state.js?v=3";


function utcMinute(value) {
  return new Date(value).toISOString().slice(0, 16).replace("T", " ");
}

export function formatConversationHistory(messages, savedAt = new Date()) {
  const lines = messages.map((message) => {
    const label = message.sender === "doctor" ? "Doctor" : "Patient";
    return `[${utcMinute(message.timestamp)} UTC] ${label}: ${message.text}`;
  });
  return [
    "SignBridge AI Conversation History",
    `Saved: ${utcMinute(savedAt)} UTC`,
    "",
    ...lines,
    "",
  ].join("\n");
}

function conversationHistoryFilename(value) {
  return `signbridge-conversation-${new Date(value).toISOString().slice(0, 16).replace("T", "-").replace(":", "")}.txt`;
}

function downloadTextFile(filename, text) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export function mountDoctorPage({
  root,
  state,
  session,
  asr,
  confirmClear = (message) => globalThis.confirm(message),
  saveTextFile = downloadTextFile,
  now = () => new Date(),
}) {
  const elements = collectElements(root);
  const controller = new AbortController();
  const subscriptions = [];
  let recording = false;
  let processing = false;
  let disposed = false;

  subscriptions.push(state.subscribe(renderState));
  subscriptions.push(session.subscribe(receiveConfirmedMessage));
  if (session.subscribeConversationCleared) subscriptions.push(session.subscribeConversationCleared(resetConversation));
  if (session.subscribeStatus) subscriptions.push(session.subscribeStatus(renderConnection));
  subscriptions.push(asr.subscribe(handleAsrEvent));

  elements.recordStart.addEventListener("click", startRecording, { signal: controller.signal });
  elements.recordStop.addEventListener("click", stopRecording, { signal: controller.signal });
  elements.draft.addEventListener("input", () => state.editDraft(elements.draft.value), { signal: controller.signal });
  elements.cancel.addEventListener("click", cancelDraft, { signal: controller.signal });
  elements.confirm.addEventListener("click", confirmDraft, { signal: controller.signal });
  elements.saveHistory.addEventListener("click", saveHistory, { signal: controller.signal });
  elements.clearConversation.addEventListener("click", clearConversation, { signal: controller.signal });

  function startRecording() {
    const request = state.beginAsr();
    recording = true;
    processing = false;
    asr.start(request.id);
    renderRecordingControls();
  }

  function stopRecording() {
    if (!recording) return;
    recording = false;
    processing = true;
    renderRecordingControls();
    asr.stop();
  }

  function cancelDraft() {
    recording = false;
    processing = false;
    asr.cancel();
    state.cancel();
    renderRecordingControls();
  }

  async function confirmDraft() {
    let confirmedText;
    try {
      confirmedText = state.beginConfirm();
    } catch (error) {
      elements.recordingStatus.textContent = error.message;
      return;
    }
    setSessionStatus("Sending…", "neutral");
    try {
      const confirmed = await session.sendConfirmed(confirmedText);
      if (disposed) return;
      appendMessage(confirmed, "Doctor");
      state.confirmSucceeded();
      setSessionStatus("Patient connected", "success");
    } catch (error) {
      if (disposed) return;
      state.confirmFailed(error.message);
      setSessionStatus("Patient not connected", "error");
      elements.draft.focus();
    }
  }

  function handleAsrEvent(event) {
    if (event.type === "status") {
      elements.recordingStatus.textContent = event.status;
      elements.microphoneStatus.textContent = event.status;
      elements.microphoneStatus.dataset.tone = "neutral";
      if (event.status.includes("medical wording")) setMedicalStatus("Checking…", "neutral");
      return;
    }
    if (event.type === "transcript") {
      state.receiveAsr(event.requestId, event.transcript);
      processing = false;
      renderRecordingControls();
      elements.microphoneStatus.textContent = "Transcript ready";
      elements.microphoneStatus.dataset.tone = "success";
      return;
    }
    if (event.type === "proposal") {
      state.receiveProposal(event.requestId, event.proposedText, event.warnings);
      processing = false;
      renderRecordingControls();
      setMedicalStatus("Connected", "success");
      return;
    }
    if (event.type === "error") {
      state.failAsr(event.requestId, event.error);
      processing = false;
      renderRecordingControls();
      elements.microphoneStatus.textContent = "Microphone error";
      elements.microphoneStatus.dataset.tone = "error";
      setMedicalStatus("No result", "error");
    }
  }

  function receiveConfirmedMessage(message) {
    if (!state.receivePatientMessage(message)) return;
    appendMessage(message, "Patient");
    setSessionStatus("Patient connected", "success");
  }

  function resetConversation() {
    elements.conversation.replaceChildren();
    const empty = document.createElement("li");
    empty.className = "empty-state";
    empty.textContent = "The confirmed conversation will appear here.";
    elements.conversation.append(empty);
    renderHistoryActions();
  }

  function saveHistory() {
    const messages = session.getHistory?.() || [];
    if (messages.length === 0) return;
    const savedAt = now();
    saveTextFile(conversationHistoryFilename(savedAt), formatConversationHistory(messages, savedAt));
  }

  async function clearConversation() {
    if (!confirmClear("Clear the confirmed conversation on both displays? Save history first if you need a copy.")) return;
    elements.clearConversation.disabled = true;
    try {
      await session.clearConversation();
      if (disposed) return;
      resetConversation();
    } catch (error) {
      if (disposed) return;
      setSessionStatus(error.message, "error");
      renderHistoryActions();
    }
  }

  function renderHistoryActions() {
    const empty = (session.getHistory?.() || []).length === 0;
    elements.saveHistory.disabled = empty;
    elements.clearConversation.disabled = empty;
  }

  function renderState(snapshot) {
    if (elements.original.value !== snapshot.originalTranscript) elements.original.value = snapshot.originalTranscript;
    if (elements.draft.value !== snapshot.draft) elements.draft.value = snapshot.draft;
    elements.recordingStatus.textContent = snapshot.error || snapshot.status;
    elements.confirm.disabled = !snapshot.draft.trim() || snapshot.sending;
    elements.cancel.disabled = snapshot.sending;

    elements.medicalReview.replaceChildren();
    if (!snapshot.proposedText && snapshot.warnings.length === 0) {
      elements.medicalReview.textContent = "No medical suggestion is available. Your draft will not be changed automatically.";
    } else {
      if (snapshot.proposedText) {
        const proposal = document.createElement("p");
        proposal.textContent = snapshot.proposedText;
        elements.medicalReview.append(proposal);
      }
      for (const warningText of snapshot.warnings) {
        const warning = document.createElement("p");
        warning.className = "review-warning";
        warning.textContent = warningText;
        elements.medicalReview.append(warning);
      }
    }
  }

  function renderRecordingControls() {
    elements.recordStart.disabled = recording;
    elements.recordStop.disabled = !recording;
    elements.voiceActions.hidden = processing;
    elements.audioVisualizer.classList.toggle("is-recording", recording);
  }

  function appendMessage(message, label) {
    elements.conversation.querySelector(".empty-state")?.remove();
    const item = document.createElement("li");
    item.className = `message-bubble message-${message.sender}`;
    const sender = document.createElement("strong");
    sender.textContent = label;
    const text = document.createElement("span");
    text.textContent = message.text;
    item.append(sender, text);
    elements.conversation.append(item);
    elements.conversation.scrollTop = elements.conversation.scrollHeight;
    renderHistoryActions();
  }

  function setSessionStatus(text, tone) {
    elements.sessionStatus.textContent = text;
    elements.sessionStatus.dataset.tone = tone;
  }

  function setMedicalStatus(text, tone) {
    elements.medicalStatus.textContent = text;
    elements.medicalStatus.dataset.tone = tone;
  }

  function renderConnection(status) {
    if (status === "connected") setSessionStatus("Patient connected", "success");
    else if (status === "error") setSessionStatus("Connection error", "error");
    else setSessionStatus("Waiting for patient", "neutral");
  }

  function close() {
    if (disposed) return;
    disposed = true;
    controller.abort();
    for (const unsubscribe of subscriptions) unsubscribe?.();
    asr.close();
    session.close();
    state.close();
  }

  setMedicalStatus("Ready", "neutral");
  renderRecordingControls();
  renderHistoryActions();
  return Object.freeze({ close });
}


function collectElements(root) {
  const byId = (id) => {
    const element = root.querySelector(`#${id}`);
    if (!element) throw new Error(`Doctor UI is missing #${id}`);
    return element;
  };
  const bySelector = (selector) => {
    const element = root.querySelector(selector);
    if (!element) throw new Error(`Doctor UI is missing ${selector}`);
    return element;
  };
  return {
    microphoneStatus: byId("microphone-status-chip"),
    sessionStatus: byId("doctor-session-status-chip"),
    conversation: byId("conversation-log"),
    saveHistory: byId("save-conversation-history"),
    clearConversation: byId("clear-conversation"),
    recordStart: byId("record-start"),
    recordStop: byId("record-stop"),
    voiceActions: bySelector(".voice-actions"),
    recordingStatus: byId("recording-status"),
    audioVisualizer: byId("audio-visualizer"),
    original: byId("original-transcript"),
    draft: byId("doctor-draft"),
    medicalReview: byId("medical-review"),
    medicalStatus: byId("medical-status-chip"),
    cancel: byId("cancel-doctor-draft"),
    confirm: byId("confirm-doctor-message"),
  };
}


if (document.documentElement.dataset.page === "doctor") {
  const backend = createBackendClient();
  const page = mountDoctorPage({
    root: document,
    state: createDoctorState(),
    session: createBackendSession({ role: "doctor", backend }),
    asr: createBackendAsrAdapter({ backend }),
  });
  addEventListener("pagehide", () => page.close(), { once: true });
}
