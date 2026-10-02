const LETTER_ROWS = [
  ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],
  ["a", "s", "d", "f", "g", "h", "j", "k", "l"],
  ["Shift", "z", "x", "c", "v", "b", "n", "m", "Backspace"],
  ["123", "Space", ".", "?"],
];

const NUMBER_ROWS = [
  ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
  ["-", "/", ":", ";", "(", ")", "$", "&", "@"],
  [".", ",", "?", "!", "'", '"', "Backspace"],
  ["ABC", "Space"],
];


export function createTouchKeyboard({ target, container, onRequestClose = () => {} }) {
  if (!target || !container) throw new Error("Touch keyboard target and container are required");
  const controller = new AbortController();
  let mode = "letters";
  let shifted = false;

  function render() {
    container.replaceChildren();
    container.classList.add("touch-keyboard");
    container.setAttribute("aria-label", "On-screen keyboard");
    const toolbar = document.createElement("div");
    toolbar.className = "keyboard-toolbar";
    const title = document.createElement("strong");
    title.textContent = "Touch keyboard";
    const closeButton = document.createElement("button");
    closeButton.type = "button";
    closeButton.className = "keyboard-close";
    closeButton.dataset.key = "Close";
    closeButton.textContent = "×";
    closeButton.setAttribute("aria-label", "Close keyboard");
    toolbar.append(title, closeButton);
    container.append(toolbar);
    const rows = mode === "letters" ? LETTER_ROWS : NUMBER_ROWS;
    for (const keys of rows) {
      const row = document.createElement("div");
      row.className = "keyboard-row";
      for (const key of keys) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "keyboard-key";
        button.dataset.key = key;
        button.textContent = keyLabel(key, shifted);
        button.setAttribute("aria-label", key);
        if (key === "Space") button.classList.add("keyboard-key-space");
        if (key === "Backspace") button.classList.add("keyboard-key-wide");
        if (key === "Shift") {
          button.classList.add("keyboard-key-wide", "keyboard-key-shift");
          button.setAttribute("aria-pressed", String(shifted));
        }
        row.append(button);
      }
      container.append(row);
    }
  }

  container.addEventListener("pointerdown", (event) => event.preventDefault(), { signal: controller.signal });
  container.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-key]");
    if (!button) return;
    const key = button.dataset.key;
    if (key === "Close") {
      onRequestClose();
      return;
    }
    if (key === "Shift") {
      shifted = !shifted;
      render();
      return;
    }
    if (key === "123" || key === "ABC") {
      mode = key === "123" ? "numbers" : "letters";
      shifted = false;
      render();
      return;
    }
    if (key === "Backspace") {
      backspace(target);
    } else {
      const text = key === "Space" ? " " : shifted && isLetter(key) ? key.toUpperCase() : key;
      insertAtSelection(target, text);
    }
    target.dispatchEvent(new Event("input", { bubbles: true }));
    target.focus();
    if (shifted && isLetter(key)) {
      shifted = false;
      render();
    }
  }, { signal: controller.signal });

  render();
  return Object.freeze({ close: () => controller.abort() });
}


function keyLabel(key, shifted) {
  if (key === "Backspace") return "⌫";
  if (key === "Space") return "Space";
  if (key === "Shift") return "⇧";
  return shifted && isLetter(key) ? key.toUpperCase() : key;
}


function isLetter(key) {
  return /^[a-z]$/.test(key);
}


function insertAtSelection(target, text) {
  const start = Number.isInteger(target.selectionStart) ? target.selectionStart : target.value.length;
  const end = Number.isInteger(target.selectionEnd) ? target.selectionEnd : start;
  target.setRangeText(text, start, end, "end");
}


function backspace(target) {
  const start = Number.isInteger(target.selectionStart) ? target.selectionStart : target.value.length;
  const end = Number.isInteger(target.selectionEnd) ? target.selectionEnd : start;
  if (start !== end) {
    target.setRangeText("", start, end, "end");
  } else if (start > 0) {
    target.setRangeText("", start - 1, start, "end");
  }
}
