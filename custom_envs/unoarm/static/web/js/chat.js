let messagesEl = null;
let sendBtnEl = null;
let inputEl = null;

export function addMessage(role, text) {
  if (!messagesEl) return;
  const item = document.createElement("div");
  item.className = `msg ${role}`;
  item.textContent = text;
  messagesEl.appendChild(item);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

export function setBusy(isBusy) {
  if (sendBtnEl) sendBtnEl.disabled = isBusy;
  if (inputEl) inputEl.disabled = isBusy;
}

export function setupChatHandlers({
  form,
  input,
  sendBtn,
  stopBtn,
  resetBtn,
  postJSON,
  onSystemMessage,
}) {
  messagesEl = document.getElementById("messages");
  sendBtnEl = sendBtn;
  inputEl = input;

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    addMessage("user", text);
    setBusy(true);
    try {
      const data = await postJSON("/api/message", { text });
      if (data.reply) addMessage("assistant", data.reply);
    } catch (error) {
      const msg = String(error.message || error);
      addMessage("system", msg);
      onSystemMessage?.(msg);
    } finally {
      setBusy(false);
      input.focus();
    }
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  stopBtn?.addEventListener("click", async () => {
    try {
      await postJSON("/api/stop");
    } catch (error) {
      const msg = String(error.message || error);
      addMessage("system", msg);
      onSystemMessage?.(msg);
    }
  });

  resetBtn?.addEventListener("click", async () => {
    try {
      await postJSON("/api/reset");
    } catch (error) {
      const msg = String(error.message || error);
      addMessage("system", msg);
      onSystemMessage?.(msg);
    }
  });
}
