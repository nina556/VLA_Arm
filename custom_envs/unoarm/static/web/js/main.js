import { RobotScene, bindSceneToolbar } from "./scene.js?v=tableplace1";
import { addMessage, setBusy, setupChatHandlers } from "./chat.js?v=dual6";
import {
  CONTROL_JOINTS,
  setupDesignHandlers,
  setDesignVisible,
  refreshDesignList,
  applyJointStateToDesign,
  applyJointLimits,
  switchMode,
} from "./design.js?v=dual6";
import { setupSettingsHandlers, setSettingsVisible } from "./settings.js?v=tableplace1";
import { setupSwordHandlers, setSwordVisible } from "./sword.js?v=dual6";
import { setupValidateHandlers, setValidateVisible } from "./validate.js?v=dual6";
import { setupReachIkHandlers, setReachIkVisible } from "./reach_ik.js?v=tableplace7";

const workspace = document.getElementById("workspace");
const obsSidebarToggleBtn = document.getElementById("obsSidebarToggleBtn");
const obsSidebarCloseBtn = document.getElementById("obsSidebarCloseBtn");
const camTop = document.getElementById("camTop");
const camLeftWrist = document.getElementById("camLeftWrist");
const camRightWrist = document.getElementById("camRightWrist");
const OBS_CAM_ELS = {
  top: camTop,
  left_wrist: camLeftWrist,
  right_wrist: camRightWrist,
};
const OBS_PREVIEW_KEY = "unoarm.obsPreviewOpen";

function readObsPreviewPref() {
  try {
    const raw = localStorage.getItem(OBS_PREVIEW_KEY);
    if (raw == null) return true;
    return raw === "1" || raw === "true";
  } catch {
    return true;
  }
}

let obsPreviewOpen = readObsPreviewPref();

function applyObsCameras(cameras) {
  if (!obsPreviewOpen) return;
  if (!cameras || typeof cameras !== "object") return;
  for (const [name, el] of Object.entries(OBS_CAM_ELS)) {
    const b64 = cameras[name];
    if (el && typeof b64 === "string" && b64.length > 0) {
      el.src = `data:image/jpeg;base64,${b64}`;
    }
  }
}

function setObsPreviewOpen(open, { syncServer = true } = {}) {
  obsPreviewOpen = Boolean(open);
  workspace?.classList.toggle("obs-open", obsPreviewOpen);
  obsSidebarToggleBtn?.classList.toggle("active", obsPreviewOpen);
  obsSidebarToggleBtn?.setAttribute("aria-pressed", obsPreviewOpen ? "true" : "false");
  try {
    localStorage.setItem(OBS_PREVIEW_KEY, obsPreviewOpen ? "1" : "0");
  } catch {
    /* ignore */
  }
  if (syncServer) {
    wsSend({ type: "obs_preview", enabled: obsPreviewOpen });
  }
  // Sidebar width change needs a canvas resize.
  requestAnimationFrame(() => {
    window.dispatchEvent(new Event("resize"));
  });
}
const logs = document.getElementById("logs");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const stopBtn = document.getElementById("stopBtn");
const resetBtn = document.getElementById("resetBtn");
const modeChatBtn = document.getElementById("modeChatBtn");
const modeDesignBtn = document.getElementById("modeDesignBtn");
const modeSwordBtn = document.getElementById("modeSwordBtn");
const modeReachIkBtn = document.getElementById("modeReachIkBtn");
const modeValidateBtn = document.getElementById("modeValidateBtn");
const modeSettingsBtn = document.getElementById("modeSettingsBtn");

let ws = null;
let lastLogCount = 0;
let currentAppMode = "chat";
/** @type {null | "settings" | "sword" | "validate"} */
let overlayPanel = null;
/** Last known scene name from snapshot (pose_ack has no scene). */
let lastSceneName = "";
/** Ignore stale pose_ack while sliding. */
let lastPoseAckSeq = 0;
/** Prefer fresh HTTP reach updates over slightly stale WS snapshots. */
let lastHttpReachAt = 0;

function setStatus(id, label, value, state) {
  const node = document.getElementById(id);
  if (!node) return;
  node.dataset.state = state || "neutral";
  const b = node.querySelector("b");
  const v = node.querySelector(".status-value") || node.querySelector("span:not(.status-dot)");
  if (b) b.textContent = label;
  if (v) v.textContent = value;
}

async function postJSON(url, body = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    const message = Array.isArray(detail)
      ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
      : detail || data.message || response.statusText;
    throw new Error(message);
  }
  return data;
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: { Accept: "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    const message = Array.isArray(detail)
      ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
      : detail || data.message || response.statusText;
    throw new Error(message);
  }
  return data;
}

function wsSend(message) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(message));
  }
}

function onSystemMessage(text) {
  addMessage("system", text);
}

const scene = new RobotScene(document.getElementById("sceneCanvas"), {
  onSystemMessage: (text) => addMessage("system", text),
});

bindSceneToolbar(scene);

setupChatHandlers({
  form,
  input,
  sendBtn,
  stopBtn,
  resetBtn,
  postJSON,
  onSystemMessage,
});

setupDesignHandlers({
  wsSend,
  postJSON,
  fetchJSON,
  onSystemMessage,
  onPoseLocalApply: (pose) => {
    scene.applyJointState({
      name: CONTROL_JOINTS,
      position: pose,
    });
  },
  onReachUpdate: (payload) => {
    lastSceneName = lastSceneName || "reach_sword";
    lastHttpReachAt = performance.now();
    if (payload && typeof payload === "object" && !Array.isArray(payload)) {
      updateReachUI(payload);
    } else {
      // legacy: (distance, success, attached) via rest not available; ignore
    }
  },
});

const settingsApi = setupSettingsHandlers({
  postJSON,
  fetchJSON,
  onSystemMessage,
});

const swordApi = setupSwordHandlers({
  postJSON,
  fetchJSON,
  onSystemMessage,
});

const validateApi = setupValidateHandlers({
  postJSON,
  fetchJSON,
  onSystemMessage,
});

const reachIkApi = setupReachIkHandlers({
  postJSON,
  fetchJSON,
  onSystemMessage,
});

function hideOverlays() {
  overlayPanel = null;
  setSettingsVisible(false);
  setSwordVisible(false);
  setValidateVisible(false);
  setReachIkVisible(false);
  reachIkApi.cleanup?.();
}

function showSettingsPanel() {
  document.getElementById("chatPanel")?.classList.remove("active");
  document.getElementById("designPanel")?.classList.remove("active");
  setSwordVisible(false);
  setValidateVisible(false);
  setReachIkVisible(false);
  reachIkApi.cleanup?.();
  overlayPanel = "settings";
  setSettingsVisible(true);
  document.getElementById("modeChatBtn")?.classList.remove("active");
  document.getElementById("modeDesignBtn")?.classList.remove("active");
  document.getElementById("modeSwordBtn")?.classList.remove("active");
  document.getElementById("modeValidateBtn")?.classList.remove("active");
  document.getElementById("modeReachIkBtn")?.classList.remove("active");
  setBusy(true);
  settingsApi.loadSettings();
}

function showSwordPanel() {
  document.getElementById("chatPanel")?.classList.remove("active");
  document.getElementById("designPanel")?.classList.remove("active");
  setSettingsVisible(false);
  setValidateVisible(false);
  setReachIkVisible(false);
  reachIkApi.cleanup?.();
  overlayPanel = "sword";
  setSwordVisible(true);
  document.getElementById("modeChatBtn")?.classList.remove("active");
  document.getElementById("modeDesignBtn")?.classList.remove("active");
  document.getElementById("modeSettingsBtn")?.classList.remove("active");
  document.getElementById("modeValidateBtn")?.classList.remove("active");
  document.getElementById("modeReachIkBtn")?.classList.remove("active");
  setBusy(true);
  swordApi.loadCurrent().catch(() => {});
}

function showValidatePanel() {
  document.getElementById("chatPanel")?.classList.remove("active");
  document.getElementById("designPanel")?.classList.remove("active");
  setSettingsVisible(false);
  setSwordVisible(false);
  setReachIkVisible(false);
  reachIkApi.cleanup?.();
  overlayPanel = "validate";
  setValidateVisible(true);
  document.getElementById("modeChatBtn")?.classList.remove("active");
  document.getElementById("modeDesignBtn")?.classList.remove("active");
  document.getElementById("modeSwordBtn")?.classList.remove("active");
  document.getElementById("modeSettingsBtn")?.classList.remove("active");
  document.getElementById("modeReachIkBtn")?.classList.remove("active");
  setBusy(true);
  validateApi.loadCurrent().catch(() => {});
}

function showReachIkPanel() {
  document.getElementById("chatPanel")?.classList.remove("active");
  document.getElementById("designPanel")?.classList.remove("active");
  setSettingsVisible(false);
  setSwordVisible(false);
  setValidateVisible(false);
  overlayPanel = "reach_ik";
  setReachIkVisible(true);
  document.getElementById("modeChatBtn")?.classList.remove("active");
  document.getElementById("modeDesignBtn")?.classList.remove("active");
  document.getElementById("modeSwordBtn")?.classList.remove("active");
  document.getElementById("modeSettingsBtn")?.classList.remove("active");
  document.getElementById("modeValidateBtn")?.classList.remove("active");
  setBusy(true);
  reachIkApi.loadCurrent().catch(() => {});
}

function leaveOverlayPanel() {
  hideOverlays();
  setDesignVisible(currentAppMode === "design");
  if (currentAppMode === "chat") {
    setBusy(false);
    input.focus();
  } else {
    setBusy(true);
  }
}

function clearReachUI() {
  setStatus("statusReach", "到位", "--", "neutral");
  scene.setGraspFx({ left: false, right: false });
  scene.setGraspAttached({ left: false, right: false });
  window.__reachSuccessAnnounced = false;
  window.__graspAttachedAnnounced = false;
  const reachEl = document.getElementById("designReachStatus");
  if (reachEl) {
    reachEl.dataset.state = "idle";
    reachEl.textContent =
      armSide === "left"
        ? "reach：左臂靠近盾牌红点并闭合夹爪可吸附"
        : "reach：右臂靠近剑柄红点并闭合夹爪可吸附";
  }
}

/** @type {"left"|"right"} */
let armSide = "right";

function setArmSide(side) {
  armSide = side === "left" ? "left" : "right";
  document.getElementById("armSideLeftBtn")?.classList.toggle("active", armSide === "left");
  document.getElementById("armSideRightBtn")?.classList.toggle("active", armSide === "right");
  window.dispatchEvent(new CustomEvent("unoarm:arm-side", { detail: { side: armSide } }));
}

function pickReachFields(data) {
  if (armSide === "left") {
    return {
      distance: data.left_reach_distance ?? data.reach_distance,
      success: data.left_success ?? data.success,
      attached: data.left_attached,
    };
  }
  return {
    distance: data.right_reach_distance ?? data.reach_distance,
    success: data.right_success ?? data.success,
    attached: data.right_attached ?? data.attached,
  };
}

function updateReachUI(dataOrDistance, success, attached) {
  // Support legacy (distance, success, attached) and snapshot-style object.
  let distM;
  let ok;
  let stuck;
  let leftAttached = false;
  let rightAttached = false;
  let execEnabled = false;
  let execPassed = false;
  let execDistM = null;
  let episodeOk = false;
  if (dataOrDistance && typeof dataOrDistance === "object") {
    const picked = pickReachFields(dataOrDistance);
    distM = Number(picked.distance);
    ok = Boolean(picked.success);
    stuck = Boolean(picked.attached);
    leftAttached = Boolean(dataOrDistance.left_attached);
    rightAttached = Boolean(dataOrDistance.right_attached ?? dataOrDistance.attached);
    execEnabled = Boolean(dataOrDistance.execution_point_enabled);
    execPassed = Boolean(dataOrDistance.execution_point_passed);
    execDistM = dataOrDistance.execution_point_distance;
    episodeOk = Boolean(dataOrDistance.success);
  } else {
    distM = Number(dataOrDistance);
    ok = Boolean(success);
    stuck = Boolean(attached);
    episodeOk = ok;
    if (armSide === "left") leftAttached = stuck;
    else rightAttached = stuck;
  }
  if (!Number.isFinite(distM)) return;
  const distCm = (distM * 100).toFixed(1);
  const label = armSide === "left" ? "左→盾" : "右→剑";
  const statusGood = stuck || ok || (execEnabled && episodeOk);
  setStatus(
    "statusReach",
    "到位",
    execEnabled && stuck
      ? execPassed || episodeOk
        ? `${label} 执行点✓`
        : `${label} 吸附→绿点`
      : stuck
        ? `${label} 吸附 ${distCm} cm`
        : ok
          ? `${label} ${distCm} cm`
          : `${label} ${distCm} cm`,
    statusGood ? "good" : "warn"
  );
  scene.setGraspAttached({ left: leftAttached, right: rightAttached });
  scene.setGraspFx({ left: leftAttached, right: rightAttached });
  const reachEl = document.getElementById("designReachStatus");
  if (!reachEl) return;
  if (stuck && execEnabled && armSide === "right") {
    const ed = Number(execDistM);
    const edCm = Number.isFinite(ed) ? (ed * 100).toFixed(1) : "--";
    if (execPassed || episodeOk) {
      reachEl.dataset.state = "success";
      reachEl.textContent = `已抓取并掠过执行点 · 剑柄距绿点 ${edCm} cm`;
      if (!window.__executionPointAnnounced) {
        window.__executionPointAnnounced = true;
        onSystemMessage(`执行点已通过（剑柄距绿点 ${edCm} cm）`);
      }
    } else {
      window.__executionPointAnnounced = false;
      reachEl.dataset.state = "near";
      reachEl.textContent = `已吸附剑柄 · 移向执行点（绿点） ${edCm} cm`;
    }
  } else if (stuck) {
    reachEl.dataset.state = "success";
    reachEl.textContent =
      armSide === "left"
        ? `左臂已吸附盾牌 · ${distCm} cm — 张开 Left_Gripper 释放`
        : `右臂已吸附剑柄 · ${distCm} cm — 张开 Right_Gripper 释放`;
    if (!window.__graspAttachedAnnounced) {
      window.__graspAttachedAnnounced = true;
      onSystemMessage(
        armSide === "left"
          ? `盾牌已吸附到左爪（${distCm} cm）`
          : `剑柄已吸附到右爪（${distCm} cm）`
      );
    }
  } else if (ok) {
    window.__graspAttachedAnnounced = false;
    window.__executionPointAnnounced = false;
    reachEl.dataset.state = "success";
    reachEl.textContent =
      armSide === "left"
        ? `左臂到位 ${distCm} cm — 闭合 Left_Gripper 可吸附盾牌`
        : `右臂到位 ${distCm} cm — 闭合 Right_Gripper 可吸附剑柄`;
  } else {
    window.__reachSuccessAnnounced = false;
    window.__graspAttachedAnnounced = false;
    window.__executionPointAnnounced = false;
    reachEl.dataset.state = distM < 0.15 ? "near" : "idle";
    reachEl.textContent =
      armSide === "left"
        ? `左臂距盾牌红点 ${distCm} cm`
        : `右臂距剑柄红点 ${distCm} cm`;
  }
}

function handlePoseAck(data) {
  if (!data.ok) {
    onSystemMessage(data.error || "姿态未生效（请先切到动作设计）");
    return;
  }
  const seq = Number(data.seq);
  if (Number.isFinite(seq)) {
    if (seq < lastPoseAckSeq) return;
    lastPoseAckSeq = seq;
  }
  // 设计模式由滑条本地驱动 3D；此处只更新到位距离。
  const sceneName = lastSceneName || "reach_sword";
  if (sceneName === "reach_sword" && data.reach_distance != null) {
    updateReachUI(data);
  }
}

function handleSnapshot(data) {
  if (data.cameras) applyObsCameras(data.cameras);
  if (data.joint_limits) applyJointLimits(data.joint_limits);

  // Design sliders own the 3D robot only in the design panel. Overlays
  // (Reach-IK replay, validate, …) and server-driven statuses must apply
  // joint_state, otherwise MuJoCo cameras move while the Three.js arm freezes.
  const designDriving =
    currentAppMode === "design" &&
    !overlayPanel &&
    data.status !== "preview" &&
    data.status !== "running" &&
    data.status !== "routing" &&
    data.status !== "replay";

  if (data.joint_state) {
    // 动作设计拖滑条时不要用 snapshot 覆盖本地姿态（会与 onPoseLocalApply 打架）。
    if (!designDriving) {
      scene.applyJointState(data.joint_state);
    }
    if (currentAppMode === "design" && data.status === "preview") {
      applyJointStateToDesign(data.joint_state);
    }
  }

  const running =
    data.status === "running" || data.status === "routing" || data.status === "replay";
  setStatus(
    "statusRollout",
    "执行",
    data.status === "running"
      ? "执行中"
      : data.status === "routing"
        ? "路由中"
        : data.status === "replay"
          ? "回放中"
          : "空闲",
    running ? "warn" : "neutral"
  );
  setStatus("statusTask", "任务", data.current_task || "--", data.current_task ? "good" : "neutral");
  setStatus("statusStep", "步数", `${data.step || 0} / ${data.max_steps || 0}`, "neutral");

  if (typeof data.policy_ready === "boolean") {
    setStatus(
      "statusPolicy",
      "策略",
      data.policy_ready ? "已加载" : "未加载",
      data.policy_ready ? "good" : "warn"
    );
  }

  if (data.scene) {
    lastSceneName = data.scene.name || lastSceneName;
    scene.applySceneConfig(data.scene);
  }

  const sceneName = data.scene?.name || lastSceneName;
  const skipStaleReach =
    designDriving && lastHttpReachAt > 0 && performance.now() - lastHttpReachAt < 250;
  if (sceneName === "reach_sword") {
    // Always sync adsorb/FX from snapshot (even when distance UI is briefly skipped).
    scene.setGraspAttached({
      left: Boolean(data.left_attached),
      right: Boolean(data.right_attached ?? data.attached),
    });
    if (data.reach_distance != null && !skipStaleReach) {
      updateReachUI(data);
    }
  } else if (sceneName === "table_place") {
    scene.setGraspAttached({
      left: false,
      right: Boolean(data.right_attached ?? data.peg_attached ?? data.attached),
    });
    if (data.place_distance != null && !skipStaleReach) {
      const fit = data.place_fit != null ? Number(data.place_fit) : null;
      const distCm = Number(data.place_distance) * 100;
      const ok = Boolean(data.success);
      setStatus(
        "statusReach",
        "放置",
        fit != null && Number.isFinite(fit)
          ? `${distCm.toFixed(1)}cm / fit=${fit.toFixed(2)}`
          : `${distCm.toFixed(1)}cm`,
        ok ? "good" : "neutral"
      );
    }
  } else if (data.scene && data.scene.name && data.scene.name !== "reach_sword") {
    clearReachUI();
  }

  if (overlayPanel) {
    setBusy(true);
  } else if (currentAppMode === "chat") {
    setBusy(Boolean(data.input_locked));
  } else {
    setBusy(true);
  }

  if (data.mode && data.mode !== currentAppMode && !overlayPanel) {
    currentAppMode = data.mode;
    setDesignVisible(data.mode === "design");
  }

  if (Array.isArray(data.logs) && data.logs.length !== lastLogCount) {
    lastLogCount = data.logs.length;
    logs.textContent = data.logs.map((row) => row.text).join("\n");
    logs.scrollTop = logs.scrollHeight;
  }

  if (data.validate) {
    validateApi.applyValidateSnapshot(data.validate);
  }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    setStatus("statusConn", "连接", "在线", "good");
    wsSend({ type: "obs_preview", enabled: obsPreviewOpen });
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "pose_ack") {
        handlePoseAck(data);
        return;
      }
      if (data.type === "obs_preview_ack") {
        return;
      }
      handleSnapshot(data);
    } catch (error) {
      console.error("WS parse error", error);
    }
  };

  ws.onclose = () => {
    setStatus("statusConn", "连接", "离线", "bad");
    setStatus("statusRollout", "执行", "离线", "bad");
    setTimeout(connect, 900);
  };
}

async function handleModeSwitch(mode) {
  if (mode === "settings") {
    showSettingsPanel();
    return;
  }
  if (mode === "sword") {
    showSwordPanel();
    return;
  }
  if (mode === "validate") {
    showValidatePanel();
    return;
  }
  if (mode === "reach_ik") {
    showReachIkPanel();
    return;
  }
  if (overlayPanel) {
    leaveOverlayPanel();
  }
  if (mode === currentAppMode) return;
  try {
    await switchMode(mode);
    currentAppMode = mode;
    if (mode === "chat") {
      setBusy(false);
      input.focus();
    } else {
      setBusy(true);
    }
  } catch (error) {
    onSystemMessage(String(error.message || error));
  }
}

modeChatBtn?.addEventListener("click", () => handleModeSwitch("chat"));
modeDesignBtn?.addEventListener("click", () => handleModeSwitch("design"));
modeSwordBtn?.addEventListener("click", () => handleModeSwitch("sword"));
modeReachIkBtn?.addEventListener("click", () => handleModeSwitch("reach_ik"));
modeValidateBtn?.addEventListener("click", () => handleModeSwitch("validate"));
modeSettingsBtn?.addEventListener("click", () => handleModeSwitch("settings"));

obsSidebarToggleBtn?.addEventListener("click", () => {
  setObsPreviewOpen(!obsPreviewOpen);
});
obsSidebarCloseBtn?.addEventListener("click", () => {
  setObsPreviewOpen(false);
});
document.getElementById("armSideLeftBtn")?.addEventListener("click", () => setArmSide("left"));
document.getElementById("armSideRightBtn")?.addEventListener("click", () => setArmSide("right"));

// Apply saved preference before first paint of WS frames.
setObsPreviewOpen(obsPreviewOpen, { syncServer: false });
setArmSide("right");

connect();
settingsApi.loadSettings();
refreshDesignList();
