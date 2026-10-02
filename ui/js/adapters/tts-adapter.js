export function createUnavailableTtsAdapter() {
  return Object.freeze({
    async play() {
      return { ok: false, error: "TTS is not connected" };
    },
  });
}


export function createBrowserTtsAdapter({ speechSynthesisImpl = globalThis.speechSynthesis, UtteranceImpl = globalThis.SpeechSynthesisUtterance } = {}) {
  return Object.freeze({
    async play(text) {
      if (!speechSynthesisImpl || !UtteranceImpl) return { ok: false, error: "Windows browser TTS is unavailable" };
      if (typeof text !== "string" || !text.trim()) return { ok: false, error: "There is no confirmed reply to play" };
      const voices = speechSynthesisImpl.getVoices?.() || [];
      if (voices.length === 0) return { ok: false, error: "No local Windows speech voice is installed" };
      speechSynthesisImpl.cancel();
      const utterance = new UtteranceImpl(text);
      utterance.voice = voices.find((voice) => /^(en-MY|ms-MY)$/i.test(voice.lang))
        || voices.find((voice) => /^(en|ms)/i.test(voice.lang))
        || voices[0];
      utterance.lang = utterance.voice.lang;
      speechSynthesisImpl.speak(utterance);
      return { ok: true };
    },
  });
}
