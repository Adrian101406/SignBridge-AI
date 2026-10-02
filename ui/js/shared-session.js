const CHANNEL_NAME = "signbridge-session-v1";
const VALID_ROLES = new Set(["doctor", "patient"]);


export function createSessionChannel({
  role,
  channelFactory = (name) => new BroadcastChannel(name),
  now = () => new Date(),
  timeoutMs = 1500,
} = {}) {
  if (!VALID_ROLES.has(role)) {
    throw new Error("Session role must be doctor or patient");
  }
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    throw new Error("Session timeout must be greater than zero");
  }

  const channel = channelFactory(CHANNEL_NAME);
  const listeners = new Set();
  const pending = new Map();
  let closed = false;
  let connectionState = "waiting";

  function onMessage(event) {
    const data = event?.data;
    if (isAcknowledgement(data)) {
      const request = pending.get(data.messageId);
      if (!request) return;
      clearTimeout(request.timer);
      pending.delete(data.messageId);
      connectionState = "connected";
      request.resolve(request.message);
      return;
    }

    if (!isConfirmedMessage(data) || data.sender === role) return;
    connectionState = "connected";
    const message = Object.freeze({ ...data });
    for (const listener of listeners) {
      listener(message);
    }
    channel.postMessage({ type: "ack", messageId: message.id });
  }

  channel.addEventListener("message", onMessage);

  function sendConfirmed(text) {
    if (closed) return Promise.reject(new Error("Session channel was closed"));
    const confirmedText = typeof text === "string" ? text.trim() : "";
    if (!confirmedText) return Promise.reject(new Error("Confirmed text cannot be empty"));

    const timestamp = now();
    if (!(timestamp instanceof Date) || Number.isNaN(timestamp.getTime())) {
      return Promise.reject(new Error("Session clock returned an invalid date"));
    }

    const message = Object.freeze({
      type: "confirmed-message",
      id: createMessageId(),
      sender: role,
      text: confirmedText,
      timestamp: timestamp.toISOString(),
    });

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        pending.delete(message.id);
        connectionState = "waiting";
        reject(new Error("Delivery confirmation was not received"));
      }, timeoutMs);
      pending.set(message.id, { message, resolve, reject, timer });
      channel.postMessage(message);
    });
  }

  function subscribe(listener) {
    if (typeof listener !== "function") {
      throw new Error("Session listener must be a function");
    }
    if (closed) {
      throw new Error("Session channel was closed");
    }
    listeners.add(listener);
    return () => listeners.delete(listener);
  }

  function close() {
    if (closed) return;
    closed = true;
    channel.removeEventListener("message", onMessage);
    channel.close();
    listeners.clear();
    for (const request of pending.values()) {
      clearTimeout(request.timer);
      request.reject(new Error("Session channel was closed"));
    }
    pending.clear();
    connectionState = "closed";
  }

  return Object.freeze({
    sendConfirmed,
    subscribe,
    getConnectionState: () => connectionState,
    close,
  });
}


function createMessageId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `msg-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}


function isAcknowledgement(data) {
  return data?.type === "ack" && typeof data.messageId === "string" && data.messageId.length > 0;
}


function isConfirmedMessage(data) {
  return data?.type === "confirmed-message"
    && typeof data.id === "string"
    && data.id.length > 0
    && VALID_ROLES.has(data.sender)
    && typeof data.text === "string"
    && data.text.trim().length > 0
    && typeof data.timestamp === "string"
    && !Number.isNaN(Date.parse(data.timestamp));
}
