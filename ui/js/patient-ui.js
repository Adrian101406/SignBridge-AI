import { createCameraAdapter } from "./adapters/camera-adapter.js";
import { createBackendClient, createBackendSession } from "./adapters/backend-client.js?v=4";
import { createBackendRecognitionAdapter } from "./adapters/recognition-adapter.js?v=11";
import { createBrowserTtsAdapter } from "./adapters/tts-adapter.js";
import { createPatientState } from "./patient-state.js?v=3";
import { createTouchKeyboard } from "./touch-keyboard.js?v=4";

const MAX_VIDEO_UPLOAD_BYTES = 100 * 1024 * 1024;


export function mountPatientPage({ root, state, session, recognition, camera, tts, mediaUrls = globalThis.URL }) {
  const elements = collectElements(root);
  const controller = new AbortController();
  const subscriptions = [];
  const touchKeyboard = createTouchKeyboard({
    target: elements.draft,
    container: elements.touchKeyboard,
    onRequestClose: hideTouchKeyboard,
  });
  elements.touchKeyboard.hidden = true;
  elements.keyboardToggle.hidden = false;
  let recognitionRunning = false;
  let demoRecognition = false;
  let latestDoctorReply = "";
  let uploadedVideo = null;
  let uploadedVideoUrl = "";
  let proposalGeneration = 0;
  let recognitionMode = "default";
  let disposed = false;

  subscriptions.push(state.subscribe(renderState));
  subscriptions.push(session.subscribe(receiveConfirmedMessage));
  if (session.subscribeConversationCleared) subscriptions.push(session.subscribeConversationCleared(resetConversation));
  if (session.subscribeStatus) subscriptions.push(session.subscribeStatus(renderConnection));
  subscriptions.push(recognition.subscribe(handleRecognitionEvent));
  subscriptions.push(camera.subscribe(renderCamera));

  elements.draft.addEventListener("input", () => {
    proposalGeneration += 1;
    state.setDraft(elements.draft.value);
  }, { signal: controller.signal });
  elements.draft.addEventListener("focus", showTouchKeyboard, { signal: controller.signal });
  elements.draft.addEventListener("click", showTouchKeyboard, { signal: controller.signal });
  elements.keyboardToggle.addEventListener("click", () => {
    showTouchKeyboard();
    elements.draft.focus();
  }, { signal: controller.signal });
  elements.draft.ownerDocument.addEventListener("pointerdown", handleOutsidePointerDown, { signal: controller.signal });
  elements.draft.ownerDocument.addEventListener("click", handleOutsidePointerDown, { signal: controller.signal });
  elements.undo.addEventListener("click", () => { proposalGeneration += 1; state.undoLastWord(); }, { signal: controller.signal });
  elements.clear.addEventListener("click", () => { proposalGeneration += 1; state.clearDraft(); }, { signal: controller.signal });
  elements.directSend.addEventListener("click", directSendDraft, { signal: controller.signal });
  elements.send.addEventListener("click", sendDraft, { signal: controller.signal });
  elements.cameraToggle.addEventListener("click", toggleCamera, { signal: controller.signal });
  elements.videoUploadButton.addEventListener("click", () => elements.videoUpload.click(), { signal: controller.signal });
  elements.videoUpload.addEventListener("change", selectUploadedVideo, { signal: controller.signal });
  elements.recognitionStart.addEventListener("click", startRecognition, { signal: controller.signal });
  elements.recognitionStop.addEventListener("click", stopRecognition, { signal: controller.signal });
  elements.numberMode.addEventListener("click", toggleNumberMode, { signal: controller.signal });
  elements.confirmRecognition.addEventListener("click", confirmRecognition, { signal: controller.signal });
  elements.retryRecognition.addEventListener("click", retryRecognition, { signal: controller.signal });
  elements.playReply.addEventListener("click", playReply, { signal: controller.signal });
  elements.playAvatar.addEventListener("click", playAvatarVideo, { signal: controller.signal });

  async function sendDraft() {
    if (recognitionRunning) haltRecognition();
    const snapshot = state.getSnapshot();
    const confirmedText = snapshot.draft.trim();
    if (!confirmedText) return;
    if (!snapshot.proposalReady) {
      const runGeneration = ++proposalGeneration;
      state.markSending();
      setGenerationStatus("Generating medical sentence…", "neutral");
      setSessionStatus("Generating medical sentence…", "neutral");
      try {
        const proposal = await session.proposePatient(confirmedText);
        if (disposed) return;
        if (runGeneration !== proposalGeneration || state.getSnapshot().draft.trim() !== confirmedText) {
          state.cancelPending();
          setGenerationStatus("", "neutral");
          setSessionStatus("Sentence changed — generate again", "neutral");
          return;
        }
        state.applyProposal(proposal.source_text, proposal.suggested_text, proposal.medical_flags);
        setGenerationStatus("", "neutral");
        const changedTerms = proposal.changed_critical_terms || [];
        elements.recognitionStatus.textContent = changedTerms.length
          ? `Review changed medical terms: ${changedTerms.join(", ")}`
          : "Person 2 suggestion ready — review before sending";
        setSessionStatus("Review required", "neutral");
      } catch (error) {
        if (disposed) return;
        state.markSendFailed(error.message);
        setGenerationStatus("Could not generate sentence. Please try again.", "error");
        setSessionStatus("Medical intelligence unavailable", "error");
      }
      return;
    }
    await deliverConfirmed(confirmedText);
  }

  async function directSendDraft() {
    const snapshot = state.getSnapshot();
    const confirmedText = snapshot.draft.trim();
    if (!confirmedText || snapshot.sending || snapshot.proposalReady) return;
    proposalGeneration += 1;
    await deliverConfirmed(confirmedText);
  }

  async function deliverConfirmed(confirmedText) {
    state.markSending();
    setSessionStatus("Sending…", "neutral");
    try {
      const confirmed = await session.sendConfirmed(confirmedText);
      if (disposed) return;
      appendMessage(confirmed, "You");
      state.markSendSucceeded();
      hideTouchKeyboard();
      setSessionStatus("Doctor connected", "success");
    } catch (error) {
      if (disposed) return;
      state.markSendFailed(error.message);
      setSessionStatus("Doctor not connected", "error");
      elements.draft.focus();
    }
  }

  async function toggleCamera() {
    const current = camera.getSnapshot();
    if (current.status === "open" || current.status === "opening") {
      stopRecognition();
      camera.stop();
      return;
    }
    recognition.stop();
    recognitionRunning = false;
    renderRecognitionControls();
    clearUploadedVideo();
    await camera.open();
  }

  async function startRecognition() {
    recognition.setMode?.(recognitionMode);
    recognitionRunning = true;
    renderRecognitionControls();
    if (uploadedVideo) {
      await recognition.recognizeFile(uploadedVideo, recognitionMode);
      recognitionRunning = false;
      renderRecognitionControls();
      return;
    }
    recognition.start();
  }

  function stopRecognition() {
    haltRecognition();
    const snapshot = state.getSnapshot();
    if (snapshot.draft.trim() && !snapshot.proposalReady && !snapshot.sending) void sendDraft();
  }

  function haltRecognition() {
    recognitionRunning = false;
    recognition.stop();
    state.release();
    renderRecognitionControls();
  }

  function toggleNumberMode() {
    setRecognitionMode(recognitionMode === "number" ? "default" : "number");
  }

  function setRecognitionMode(nextMode) {
    recognitionMode = nextMode === "number" ? "number" : "default";
    recognition.setMode?.(recognitionMode);
    renderRecognitionMode();
  }

  function confirmRecognition() {
    const candidate = state.getSnapshot();
    if (!state.confirmCandidate()) return;
    proposalGeneration += 1;
    if (candidate.candidateDomain === "number" || recognitionMode === "number") {
      setRecognitionMode("default");
    }
    recognition.continueAfterDecision?.();
  }

  function retryRecognition() {
    const candidate = state.getSnapshot();
    if (!state.retryCandidate()) return;
    if (candidate.candidateDomain === "number") setRecognitionMode("number");
    recognition.continueAfterDecision?.();
  }

  function handleRecognitionEvent(event) {
    demoRecognition = event.demo === true;
    if (event.type === "observation") {
      state.observe(event);
    } else if (event.type === "release") {
      state.release();
    } else if (event.type === "status") {
      elements.recognitionStatus.textContent = event.status;
    }
  }

  function receiveConfirmedMessage(message) {
    if (message.sender !== "doctor") return;
    latestDoctorReply = message.text;
    if (expectsNumberAnswer(message.text)) setRecognitionMode("number");
    if (isFeverDurationQuestion(message.text)) void playAvatarVideo();
    elements.playReply.disabled = false;
    appendMessage(message, "Doctor");
    setSessionStatus("Doctor connected", "success");
  }

  async function playReply() {
    if (!latestDoctorReply) return;
    const result = await tts.play(latestDoctorReply);
    if (!result.ok) elements.recognitionStatus.textContent = result.error;
  }

  async function playAvatarVideo() {
    elements.avatarVideo.hidden = false;
    elements.avatarEmpty.hidden = true;
    elements.playAvatar.disabled = false;
    elements.avatarStatus.textContent = "Playing the fever duration question.";
    elements.avatarVideo.currentTime = 0;
    try {
      await elements.avatarVideo.play();
    } catch {
      elements.avatarStatus.textContent = "Fever duration avatar ready — tap Replay avatar.";
    }
  }

  function renderState(snapshot) {
    if (elements.draft.value !== snapshot.draft) elements.draft.value = snapshot.draft;
    elements.currentDetection.textContent = snapshot.currentLabel || "—";
    elements.confidence.textContent = Number.isFinite(snapshot.confidence)
      ? `${Math.round(snapshot.confidence * 100)}%`
      : "—";
    const prefix = demoRecognition ? "DEMO — NOT AI · " : "";
    elements.recognitionStatus.textContent = snapshot.error || `${prefix}${snapshot.status}`;
    elements.send.disabled = !snapshot.draft.trim() || snapshot.sending;
    elements.send.textContent = snapshot.proposalReady ? "Confirm & send" : "Generate sentence";
    elements.directSend.hidden = snapshot.proposalReady;
    elements.directSend.disabled = !snapshot.draft.trim() || snapshot.sending || snapshot.proposalReady;
    elements.originalBim.hidden = !snapshot.originalBim;
    const reviewFlags = snapshot.medicalFlags?.length ? ` · Review: ${snapshot.medicalFlags.join("; ")}` : "";
    elements.originalBim.textContent = snapshot.originalBim ? `Original BIM: ${snapshot.originalBim}${reviewFlags}` : "";
    elements.undo.disabled = snapshot.sending;
    elements.clear.disabled = !snapshot.draft || snapshot.sending;
    elements.decisionActions.hidden = !snapshot.candidatePending;
    elements.recognitionActions.hidden = snapshot.candidatePending;
    elements.confirmRecognition.disabled = !snapshot.candidatePending;
    elements.retryRecognition.disabled = !snapshot.candidatePending;
  }

  function showTouchKeyboard() {
    elements.touchKeyboard.hidden = false;
    elements.keyboardToggle.hidden = true;
    elements.keyboardToggle.setAttribute("aria-expanded", "true");
  }

  function hideTouchKeyboard() {
    elements.touchKeyboard.hidden = true;
    elements.keyboardToggle.hidden = false;
    elements.keyboardToggle.setAttribute("aria-expanded", "false");
  }

  function handleOutsidePointerDown(event) {
    if (event.target === elements.draft || event.target === elements.keyboardToggle || elements.touchKeyboard.contains(event.target)) return;
    if (event.type === "click" && elements.draft.ownerDocument.activeElement === elements.draft) return;
    hideTouchKeyboard();
  }

  function renderCamera(snapshot) {
    const isOpen = snapshot.status === "open";
    const isUploaded = Boolean(uploadedVideo);
    elements.cameraStage.classList.toggle("is-active", isOpen || isUploaded);
    elements.cameraToggle.textContent = isOpen || snapshot.status === "opening" ? "Close camera" : "Open camera";
    elements.cameraToggle.disabled = snapshot.status === "opening";
    elements.cameraStatus.textContent = isUploaded
      ? "Video Ready"
      : snapshot.status === "opening" ? "Camera Opening" : `Camera ${isOpen ? "Active" : "Off"}`;
    elements.cameraStatus.dataset.tone = snapshot.status === "error" ? "error" : isOpen || isUploaded ? "success" : "neutral";
    if (snapshot.stream) elements.cameraPreview.srcObject = snapshot.stream;
    if (!snapshot.stream && elements.cameraPreview.srcObject) elements.cameraPreview.srcObject = null;
    if (snapshot.error) elements.recognitionStatus.textContent = snapshot.error;
    else if (snapshot.notice) elements.recognitionStatus.textContent = snapshot.notice;
  }

  function selectUploadedVideo() {
    const file = elements.videoUpload.files?.[0];
    if (!file) return;
    if (!["video/mp4", "video/webm"].includes(file.type)) {
      elements.recognitionStatus.textContent = "Choose an MP4 or WebM video";
      elements.videoUpload.value = "";
      return;
    }
    if (file.size > MAX_VIDEO_UPLOAD_BYTES) {
      elements.recognitionStatus.textContent = "Video must be 100 MiB or smaller";
      elements.videoUpload.value = "";
      return;
    }
    recognition.stop();
    recognitionRunning = false;
    camera.stop();
    clearUploadedVideo();
    uploadedVideo = file;
    uploadedVideoUrl = mediaUrls.createObjectURL(file);
    elements.cameraPreview.srcObject = null;
    elements.cameraPreview.src = uploadedVideoUrl;
    elements.cameraPreview.controls = true;
    elements.cameraStage.classList.add("is-active");
    elements.cameraStatus.textContent = "Video Ready";
    elements.cameraStatus.dataset.tone = "success";
    elements.recognitionStatus.textContent = "Uploaded video ready for recognition";
    renderRecognitionControls();
  }

  function clearUploadedVideo() {
    if (uploadedVideoUrl) mediaUrls.revokeObjectURL(uploadedVideoUrl);
    uploadedVideo = null;
    uploadedVideoUrl = "";
    elements.videoUpload.value = "";
    elements.cameraPreview.controls = false;
    elements.cameraPreview.removeAttribute("src");
  }

  function renderRecognitionControls() {
    elements.recognitionStart.disabled = recognitionRunning;
    elements.recognitionStop.disabled = !recognitionRunning;
  }

  function renderRecognitionMode() {
    const numberMode = recognitionMode === "number";
    elements.recognitionMode.textContent = numberMode ? "Number" : "Medical + General";
    elements.numberMode.textContent = numberMode ? "Use words" : "Number";
    elements.numberMode.setAttribute("aria-pressed", String(numberMode));
    elements.numberMode.dataset.active = String(numberMode);
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
  }

  function resetConversation() {
    elements.conversation.replaceChildren();
    const empty = document.createElement("li");
    empty.className = "empty-state";
    empty.textContent = "Your conversation with the doctor will appear here.";
    elements.conversation.append(empty);
  }

  function setSessionStatus(text, tone) {
    elements.sessionStatus.textContent = text;
    elements.sessionStatus.dataset.tone = tone;
  }

  function setGenerationStatus(text, tone) {
    elements.generationStatus.textContent = text;
    elements.generationStatus.dataset.tone = tone;
    elements.generationStatus.hidden = !text;
  }

  function renderConnection(status) {
    if (status === "connected") setSessionStatus("Doctor connected", "success");
    else if (status === "error") setSessionStatus("Connection error", "error");
    else setSessionStatus("Waiting for doctor", "neutral");
  }

  function close() {
    if (disposed) return;
    disposed = true;
    controller.abort();
    clearUploadedVideo();
    for (const unsubscribe of subscriptions) unsubscribe?.();
    recognition.close();
    touchKeyboard.close();
    camera.close();
    session.close();
    state.close();
  }

  renderRecognitionControls();
  renderRecognitionMode();
  return Object.freeze({ close });
}


function collectElements(root) {
  const byId = (id) => {
    const element = root.querySelector(`#${id}`);
    if (!element) throw new Error(`Patient UI is missing #${id}`);
    return element;
  };
  return {
    cameraPreview: byId("camera-preview"),
    cameraStage: root.querySelector(".camera-stage"),
    cameraToggle: byId("camera-toggle"),
    videoUploadButton: byId("video-upload-button"),
    videoUpload: byId("video-upload"),
    cameraStatus: byId("camera-status-chip"),
    sessionStatus: byId("session-status-chip"),
    recognitionStart: byId("recognition-start"),
    recognitionStop: byId("recognition-stop"),
    numberMode: byId("recognition-number-mode"),
    recognitionMode: byId("recognition-mode"),
    decisionActions: byId("recognition-decision-actions"),
    recognitionActions: root.querySelector(".recognition-actions"),
    confirmRecognition: byId("confirm-recognition"),
    retryRecognition: byId("retry-recognition"),
    currentDetection: byId("current-detection"),
    confidence: byId("recognition-confidence"),
    recognitionStatus: byId("recognition-status"),
    draft: byId("patient-draft"),
    originalBim: byId("patient-original-bim"),
    generationStatus: byId("patient-generation-status"),
    touchKeyboard: byId("touch-keyboard"),
    keyboardToggle: byId("keyboard-toggle"),
    undo: byId("undo-word"),
    clear: byId("clear-draft"),
    directSend: byId("direct-send-patient-message"),
    send: byId("send-patient-message"),
    conversation: byId("conversation-log"),
    avatarVideo: byId("avatar-video"),
    avatarEmpty: byId("avatar-empty"),
    avatarStatus: byId("avatar-status"),
    playAvatar: byId("play-avatar-video"),
    playReply: byId("play-doctor-reply"),
  };
}


function expectsNumberAnswer(text) {
  const value = String(text || "");
  return /\b(how many|how long|age|old|days?|weeks?|months?|times?|dose|dosage|hours?|minutes?|temperature|rate|score)\b|多少|几天|几次|年龄|多大|多久|剂量|体温|评分|\b(berapa|hari|minggu|bulan|umur|kali|dos|jam|suhu)\b/i.test(value);
}


function isFeverDurationQuestion(text) {
  const value = String(text || "").toLowerCase();
  const asksDuration = /\bhow\s+many\s+days?\b|\bberapa\s+hari\b|多少天|几天|幾天/.test(value);
  const mentionsFever = /\bfever\b|\bdemam\b|发烧|發燒/.test(value);
  return asksDuration && mentionsFever;
}


if (document.documentElement.dataset.page === "patient") {
  const backend = createBackendClient();
  const camera = createCameraAdapter();
  const page = mountPatientPage({
    root: document,
    state: createPatientState(),
    session: createBackendSession({ role: "patient", backend }),
    recognition: createBackendRecognitionAdapter({ backend, getStream: () => camera.getSnapshot().stream }),
    camera,
    tts: createBrowserTtsAdapter(),
  });
  addEventListener("pagehide", () => page.close(), { once: true });
}
