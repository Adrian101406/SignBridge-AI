function messageId() {
  return globalThis.crypto?.randomUUID?.() ?? `msg-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function toUiMessage(payload) {
  return Object.freeze({
    type: "confirmed-message",
    id: payload.message_id,
    sender: payload.sender,
    text: payload.final_text,
    timestamp: payload.confirmed_at,
  });
}

export function createBackendClient({ baseUrl = "", fetchImpl = globalThis.fetch, WebSocketImpl = globalThis.WebSocket } = {}) {
  async function request(path, options) {
    const response = await fetchImpl(`${baseUrl}${path}`, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = payload.detail;
      throw new Error(detail?.message || detail?.code || `Request failed (${response.status})`);
    }
    return payload;
  }

  return Object.freeze({
    recognize: (blob, mode = "default") => request("/api/patient/clips", {
      method: "POST",
      headers: { "Content-Type": blob.type || "video/webm", "X-Recognition-Mode": mode },
      body: blob,
    }),
    transcribe: (wav) => request("/api/doctor/asr", { method: "POST", headers: { "Content-Type": "audio/wav" }, body: wav }),
    proposePatient: (sourceText) => request("/api/patient/mcie", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_text: sourceText }),
    }),
    clearConversation: () => request("/api/session/clear", { method: "POST" }),
    confirm: async (role, text, id = messageId()) => toUiMessage(await request(`/api/${role}/confirm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ final_text: text, message_id: id }),
    })),
    connect(onEvent, onStatus = () => {}) {
      const scheme = location.protocol === "https:" ? "wss" : "ws";
      let socket;
      let reconnectTimer;
      let closed = false;
      const open = () => {
        if (closed) return;
        socket = new WebSocketImpl(`${scheme}://${location.host}/ws/session`);
        socket.addEventListener("open", () => onStatus("connected"));
        socket.addEventListener("close", () => {
          onStatus("waiting");
          if (!closed) reconnectTimer = setTimeout(open, 1000);
        });
        socket.addEventListener("error", () => onStatus("error"));
        socket.addEventListener("message", (event) => {
          const data = JSON.parse(event.data);
          if (data.event === "message.confirmed") onEvent(toUiMessage(data.payload));
          else onEvent(data);
        });
      };
      open();
      return () => { closed = true; clearTimeout(reconnectTimer); socket?.close(); };
    },
  });
}

export function createBackendSession({ role, backend }) {
  const listeners = new Set();
  const statusListeners = new Set();
  const clearListeners = new Set();
  const history = [];
  const incomingHistory = [];
  const seen = new Set();
  let status = "waiting";
  const disconnect = backend.connect((event) => {
    if (event?.event === "session.snapshot") {
      const hadHistory = history.length > 0;
      history.length = 0;
      incomingHistory.length = 0;
      seen.clear();
      for (const payload of event.payload?.messages || []) receive(toUiMessage(payload));
      if (hadHistory && history.length === 0) notifyCleared();
      return;
    }
    if (event?.event === "conversation.cleared") {
      history.length = 0;
      incomingHistory.length = 0;
      seen.clear();
      notifyCleared();
      return;
    }
    receive(event);
  }, (next) => {
    status = next;
    for (const listener of statusListeners) listener(next);
  });
  function receive(event) {
    if (event?.type !== "confirmed-message" || seen.has(event.id)) return;
    seen.add(event.id);
    history.push(event);
    if (event.sender === role) return;
    incomingHistory.push(event);
    for (const listener of listeners) listener(event);
  }
  function notifyCleared() {
    for (const listener of clearListeners) listener();
  }
  return Object.freeze({
    async sendConfirmed(text) {
      const confirmed = await backend.confirm(role, text);
      receive(confirmed);
      return confirmed;
    },
    proposePatient: (text) => backend.proposePatient(text),
    clearConversation: () => backend.clearConversation(),
    getHistory: () => history.slice(),
    subscribe(listener) {
      listeners.add(listener);
      for (const message of incomingHistory) listener(message);
      return () => listeners.delete(listener);
    },
    subscribeConversationCleared(listener) {
      clearListeners.add(listener);
      return () => clearListeners.delete(listener);
    },
    subscribeStatus(listener) {
      statusListeners.add(listener);
      listener(status);
      return () => statusListeners.delete(listener);
    },
    getConnectionState: () => status,
    close() { disconnect(); listeners.clear(); statusListeners.clear(); clearListeners.clear(); status = "closed"; },
  });
}
