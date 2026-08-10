/** Web settings panel: load / save via /api/settings */

const FIELDS = [
  "checkpoint",
  "vlm_model_name",
  "device",
  "llm_model",
  "api_key",
  "api_base_url",
  "api_key_env",
  "llm_timeout",
  "scene",
  "max_steps",
  "n_action_steps",
  "action_ema",
  "max_action_delta",
  "reach_success_threshold",
  "speed",
  "bridge_base_url",
  "bridge_arms",
  "bridge_result_timeout_sec",
  "bridge_http_timeout_sec",
];

const CHECKBOX_FIELDS = [
  "terminate_on_success",
  "remove_sword",
  "remove_shield",
  "enable_execution_point",
  "bridge_enabled",
  "bridge_execute",
];

export function setupSettingsHandlers({ postJSON, fetchJSON, onSystemMessage }) {
  const form = document.getElementById("settingsForm");
  const statusEl = document.getElementById("settingsStatus");
  const saveBtn = document.getElementById("settingsSaveBtn");
  const reloadBtn = document.getElementById("settingsReloadBtn");
  const addTaskBtn = document.getElementById("addAllowedTaskBtn");

  function createTaskInput(value = "") {
    const row = document.createElement("div");
    row.className = "task-row";

    const input = document.createElement("input");
    input.className = "design-input task-input";
    input.type = "text";
    input.spellcheck = false;
    input.placeholder = "Reach the sword handle";
    input.value = value;

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "task-remove-btn";
    removeBtn.textContent = "删除";
    removeBtn.addEventListener("click", () => {
      row.remove();
      ensureTaskInput();
    });

    row.append(input, removeBtn);
    return row;
  }

  function renderAllowedTasks(tasks) {
    const listEl = document.getElementById("set_allowed_tasks");
    if (!listEl) return;
    listEl.replaceChildren();
    const cleanTasks = tasks.map((task) => String(task || "").trim()).filter(Boolean);
    for (const task of cleanTasks.length ? cleanTasks : [""]) {
      listEl.append(createTaskInput(task));
    }
  }

  function ensureTaskInput() {
    const listEl = document.getElementById("set_allowed_tasks");
    if (listEl && listEl.querySelectorAll(".task-input").length === 0) {
      listEl.append(createTaskInput());
    }
  }

  function collectAllowedTasks() {
    const listEl = document.getElementById("set_allowed_tasks");
    if (!listEl) return [];
    return Array.from(listEl.querySelectorAll(".task-input"))
      .map((input) => input.value.trim())
      .filter(Boolean);
  }

  async function loadSettings() {
    try {
      const data = await fetchJSON("/api/settings");
      fillForm(data);
      updatePolicyBadge(data);
      if (statusEl) statusEl.textContent = data.policy_ready ? "策略已加载" : (data.policy_error || "策略未加载");
    } catch (error) {
      onSystemMessage(`加载设置失败: ${error.message || error}`);
    }
  }

  function fillForm(data) {
    for (const key of FIELDS) {
      const el = document.getElementById(`set_${key}`);
      if (!el) continue;
      if (el.type === "checkbox") {
        el.checked = Boolean(data[key]);
      } else {
        el.value = data[key] ?? "";
      }
    }
    for (const key of CHECKBOX_FIELDS) {
      const el = document.getElementById(`set_${key}`);
      if (el) el.checked = Boolean(data[key]);
    }
    const list = Array.isArray(data.allowed_tasks) ? data.allowed_tasks : [];
    renderAllowedTasks(list);
    syncSceneOptionsVisibility();
  }

  function syncSceneOptionsVisibility() {
    const sceneEl = document.getElementById("set_scene");
    const swordOpts = document.getElementById("swordSceneOptions");
    if (!swordOpts) return;
    const scene = sceneEl?.value || "reach_sword";
    swordOpts.style.display = scene === "reach_sword" ? "" : "none";
  }

  document.getElementById("set_scene")?.addEventListener("change", syncSceneOptionsVisibility);

  function collectPatch() {
    const patch = {};
    for (const key of FIELDS) {
      const el = document.getElementById(`set_${key}`);
      if (!el) continue;
      if (el.type === "number") {
        patch[key] = el.value === "" ? null : Number(el.value);
      } else {
        patch[key] = el.value;
      }
    }
    for (const key of CHECKBOX_FIELDS) {
      const el = document.getElementById(`set_${key}`);
      if (el) patch[key] = el.checked;
    }
    patch.allowed_tasks = collectAllowedTasks();
    return patch;
  }

  function updatePolicyBadge(data) {
    const badge = document.getElementById("statusPolicy");
    if (!badge) return;
    const v = badge.querySelector("span:last-child");
    const b = badge.querySelector("b");
    if (b) b.textContent = "策略";
    if (data.policy_ready) {
      badge.dataset.state = "good";
      if (v) v.textContent = "已加载";
    } else {
      badge.dataset.state = "warn";
      if (v) v.textContent = "未加载";
    }
  }

  saveBtn?.addEventListener("click", async () => {
    if (statusEl) statusEl.textContent = "保存中…";
    try {
      const data = await postJSON("/api/settings", collectPatch());
      fillForm(data);
      updatePolicyBadge(data);
      if (statusEl) {
        statusEl.textContent = data.policy_ready
          ? "已保存，策略已加载"
          : `已保存。${data.policy_error || "策略未加载"}`;
      }
      onSystemMessage("设置已保存");
    } catch (error) {
      if (statusEl) statusEl.textContent = String(error.message || error);
      onSystemMessage(`保存设置失败: ${error.message || error}`);
    }
  });

  reloadBtn?.addEventListener("click", () => loadSettings());
  addTaskBtn?.addEventListener("click", () => {
    const listEl = document.getElementById("set_allowed_tasks");
    if (!listEl) return;
    const row = createTaskInput();
    listEl.append(row);
    row.querySelector(".task-input")?.focus();
  });

  form?.addEventListener("submit", (event) => {
    event.preventDefault();
    saveBtn?.click();
  });

  return { loadSettings, updatePolicyBadge };
}

export function setSettingsVisible(visible) {
  document.getElementById("settingsPanel")?.classList.toggle("active", visible);
  document.getElementById("modeSettingsBtn")?.classList.toggle("active", visible);
}
