/** Target pose tab: sword (right) + shield (left) handle/euler via /api/settings */

const DEFAULT_SWORD_HANDLE = [-0.35, -0.55, 1.0];
const DEFAULT_SWORD_EULER = [0, 0, 0];
const DEFAULT_SHIELD_HANDLE = [0.35, -0.46, 1.05];
const DEFAULT_SHIELD_EULER = [0, 0, 0];
const DEFAULT_EXECUTION_POINT = [-0.15, -0.35, 1.15];

function num(id, fallback = 0) {
  const el = document.getElementById(id);
  if (!el || el.value === "") return fallback;
  const v = Number(el.value);
  return Number.isFinite(v) ? v : fallback;
}

function setTriple(prefix, xyz, defaults) {
  const ids = [`${prefix}_hx`, `${prefix}_hy`, `${prefix}_hz`];
  for (let i = 0; i < 3; i++) {
    const el = document.getElementById(ids[i]);
    if (el) el.value = xyz?.[i] ?? defaults[i];
  }
}

function setEuler(prefix, euler, defaults) {
  const map = [
    [`${prefix}_yaw`, 0],
    [`${prefix}_pitch`, 1],
    [`${prefix}_roll`, 2],
  ];
  for (const [id, idx] of map) {
    const el = document.getElementById(id);
    if (el) el.value = euler?.[idx] ?? defaults[idx];
  }
}

function fmtVec3(v, digits = 3) {
  return `[${Number(v[0]).toFixed(digits)}, ${Number(v[1]).toFixed(digits)}, ${Number(v[2]).toFixed(digits)}]`;
}

function fmtEuler(v) {
  return `[${Number(v[0]).toFixed(1)}°, ${Number(v[1]).toFixed(1)}°, ${Number(v[2]).toFixed(1)}°]`;
}

export function setupSwordHandlers({ postJSON, fetchJSON, onSystemMessage }) {
  const form = document.getElementById("swordForm");
  const statusEl = document.getElementById("swordStatus");
  const applyBtn = document.getElementById("swordApplyBtn");
  const reloadBtn = document.getElementById("swordReloadBtn");
  const resetBtn = document.getElementById("swordResetBtn");

  function fillForm(data) {
    const swordHandle = Array.isArray(data?.sword_handle_pos)
      ? data.sword_handle_pos
      : DEFAULT_SWORD_HANDLE;
    const swordEuler = Array.isArray(data?.sword_euler_deg)
      ? data.sword_euler_deg
      : DEFAULT_SWORD_EULER;
    const shieldHandle = Array.isArray(data?.shield_handle_pos)
      ? data.shield_handle_pos
      : DEFAULT_SHIELD_HANDLE;
    const shieldEuler = Array.isArray(data?.shield_euler_deg)
      ? data.shield_euler_deg
      : DEFAULT_SHIELD_EULER;
    const execPos = Array.isArray(data?.execution_point_pos)
      ? data.execution_point_pos
      : DEFAULT_EXECUTION_POINT;

    setTriple("sword", swordHandle, DEFAULT_SWORD_HANDLE);
    setEuler("sword", swordEuler, DEFAULT_SWORD_EULER);
    setTriple("shield", shieldHandle, DEFAULT_SHIELD_HANDLE);
    setEuler("shield", shieldEuler, DEFAULT_SHIELD_EULER);
    setTriple("exec", execPos, DEFAULT_EXECUTION_POINT);
  }

  function collectPatch() {
    return {
      sword_handle_pos: [
        num("sword_hx", DEFAULT_SWORD_HANDLE[0]),
        num("sword_hy", DEFAULT_SWORD_HANDLE[1]),
        num("sword_hz", DEFAULT_SWORD_HANDLE[2]),
      ],
      sword_euler_deg: [
        num("sword_yaw", 0),
        num("sword_pitch", 0),
        num("sword_roll", 0),
      ],
      shield_handle_pos: [
        num("shield_hx", DEFAULT_SHIELD_HANDLE[0]),
        num("shield_hy", DEFAULT_SHIELD_HANDLE[1]),
        num("shield_hz", DEFAULT_SHIELD_HANDLE[2]),
      ],
      shield_euler_deg: [
        num("shield_yaw", 0),
        num("shield_pitch", 0),
        num("shield_roll", 0),
      ],
      execution_point_pos: [
        num("exec_hx", DEFAULT_EXECUTION_POINT[0]),
        num("exec_hy", DEFAULT_EXECUTION_POINT[1]),
        num("exec_hz", DEFAULT_EXECUTION_POINT[2]),
      ],
      scene: "reach_sword",
    };
  }

  async function loadCurrent() {
    if (statusEl) statusEl.textContent = "读取当前位姿…";
    try {
      const data = await fetchJSON("/api/settings");
      fillForm(data);
      if (statusEl) {
        const sh = data.sword_handle_pos || DEFAULT_SWORD_HANDLE;
        const se = data.sword_euler_deg || DEFAULT_SWORD_EULER;
        const hh = data.shield_handle_pos || DEFAULT_SHIELD_HANDLE;
        const he = data.shield_euler_deg || DEFAULT_SHIELD_EULER;
        const ep = data.execution_point_pos || DEFAULT_EXECUTION_POINT;
        const execOn = data.enable_execution_point ? "开" : "关";
        statusEl.textContent =
          `剑 ${fmtVec3(sh)} ${fmtEuler(se)}\n` +
          `盾 ${fmtVec3(hh)} ${fmtEuler(he)}\n` +
          `执行点[${execOn}] ${fmtVec3(ep)}`;
      }
      return data;
    } catch (error) {
      if (statusEl) statusEl.textContent = String(error.message || error);
      onSystemMessage(`读取目标位姿失败: ${error.message || error}`);
      throw error;
    }
  }

  async function apply() {
    if (statusEl) statusEl.textContent = "应用中…";
    try {
      const data = await postJSON("/api/settings", collectPatch());
      fillForm(data);
      if (statusEl) statusEl.textContent = "已应用并重载环境（剑+盾+执行点）";
      onSystemMessage("目标位姿（剑+盾+执行点）已更新并重载");
      return data;
    } catch (error) {
      if (statusEl) statusEl.textContent = String(error.message || error);
      onSystemMessage(`应用目标位姿失败: ${error.message || error}`);
      throw error;
    }
  }

  applyBtn?.addEventListener("click", async (event) => {
    event.preventDefault();
    await apply().catch(() => {});
  });

  reloadBtn?.addEventListener("click", () => {
    loadCurrent().catch(() => {});
  });

  resetBtn?.addEventListener("click", () => {
    fillForm({
      sword_handle_pos: DEFAULT_SWORD_HANDLE,
      sword_euler_deg: DEFAULT_SWORD_EULER,
      shield_handle_pos: DEFAULT_SHIELD_HANDLE,
      shield_euler_deg: DEFAULT_SHIELD_EULER,
      execution_point_pos: DEFAULT_EXECUTION_POINT,
    });
    if (statusEl) statusEl.textContent = "已填入默认值（尚未应用，点「应用并重载」生效）";
  });

  form?.addEventListener("submit", (event) => {
    event.preventDefault();
    applyBtn?.click();
  });

  return { loadCurrent, apply, fillForm, setVisible: setSwordVisible };
}

export function setSwordVisible(visible) {
  document.getElementById("swordPanel")?.classList.toggle("active", visible);
  document.getElementById("modeSwordBtn")?.classList.toggle("active", visible);
}
