export const CONTROL_JOINTS = [
  "Left_Joint1",
  "Left_Joint2",
  "Left_Joint3",
  "Left_Joint4",
  "Left_Joint5",
  "Left_Joint6",
  "Left_Joint7",
  "Left_Gripper_Joint",
  "Right_Joint1",
  "Right_Joint2",
  "Right_Joint3",
  "Right_Joint4",
  "Right_Joint5",
  "Right_Joint6",
  "Right_Joint7",
  "Right_Gripper_Joint",
];

// Fallback until server sends real MuJoCo jnt_range (raw radians, not normalized).
const FALLBACK_LIMITS = CONTROL_JOINTS.map(() => [-3.14, 3.14]);
const NUDGE_STEP = 0.02;

let wsSend = () => {};
let postJSON = async () => ({});
let fetchJSON = async () => ({});
let onSystemMessage = () => {};
/** @type {(pose: number[]) => void} */
let onPoseLocalApply = () => {};
/** @type {(distance: number, success: boolean) => void} */
let onReachUpdate = () => {};

let currentMode = "chat";
let currentPose = new Array(CONTROL_JOINTS.length).fill(0);
let jointLimits = FALLBACK_LIMITS.map((pair) => [...pair]);
let keyframes = {};
let playlist = [];
let projectName = "";
let projectTask = "";

let selectedJointIndex = CONTROL_JOINTS.findIndex((name) => name.startsWith("Right_"));
let activeArmSide = "right";
let selectedKeyframe = null;
let selectedPlaylistIndex = -1;
let selectedDesignName = null;
let suppressPoseSend = false;
let generatePollTimer = null;
let poseSeq = 0;
let pendingPoseSend = null;
let poseSendInFlight = false;

let els = {};

function jointMin(index) {
  return jointLimits[index]?.[0] ?? -3.14;
}

function jointMax(index) {
  return jointLimits[index]?.[1] ?? 3.14;
}

function clamp(value, index = 0) {
  return Math.min(jointMax(index), Math.max(jointMin(index), value));
}

function poseDictToVector(poseDict) {
  if (Array.isArray(poseDict)) {
    return poseDict.slice(0, CONTROL_JOINTS.length).map((v, i) => clamp(Number(v) || 0, i));
  }
  return CONTROL_JOINTS.map((name, i) => clamp(Number(poseDict?.[name]) || 0, i));
}

function vectorToPoseDict(vector) {
  const dict = {};
  CONTROL_JOINTS.forEach((name, i) => {
    dict[name] = vector[i];
  });
  return dict;
}

function formatJointValue(value) {
  return Number(value).toFixed(3);
}

function updateSelectedJointLabel() {
  if (!els.selectedJointLabel) return;
  const name = CONTROL_JOINTS[selectedJointIndex];
  const value = currentPose[selectedJointIndex];
  const lo = jointMin(selectedJointIndex);
  const hi = jointMax(selectedJointIndex);
  els.selectedJointLabel.textContent =
    `${name} = ${formatJointValue(value)}  [${formatJointValue(lo)}, ${formatJointValue(hi)}] rad`;
}

function visibleJointIndices() {
  const prefix = activeArmSide === "left" ? "Left_" : "Right_";
  return CONTROL_JOINTS
    .map((name, index) => ({ name, index }))
    .filter((item) => item.name.startsWith(prefix))
    .map((item) => item.index);
}

function ensureSelectedJointVisible() {
  const visible = visibleJointIndices();
  if (!visible.includes(selectedJointIndex)) {
    selectedJointIndex = visible[0] ?? 0;
  }
}

function highlightJointRows() {
  els.jointSliders?.querySelectorAll(".joint-row").forEach((row) => {
    row.classList.toggle("selected", Number(row.dataset.index) === selectedJointIndex);
  });
}

function updateSliderValues() {
  suppressPoseSend = true;
  ensureSelectedJointVisible();
  els.jointSliders?.querySelectorAll(".joint-row").forEach((row) => {
    const index = Number(row.dataset.index);
    const slider = row.querySelector('input[type="range"]');
    const valueEl = row.querySelector(".joint-value");
    const target = Number(currentPose[index]) || 0;
    if (slider) {
      const lo = jointMin(index);
      const hi = jointMax(index);
      if (Number(slider.min) !== lo) slider.min = String(lo);
      if (Number(slider.max) !== hi) slider.max = String(hi);
      // Browsers may leave the thumb visually stuck when .value is already equal to
      // the target (e.g. after 归零). Nudge then set to force a repaint.
      const mid = lo + (hi - lo) * 0.5;
      const nudge = Math.abs(target - mid) < 1e-6 ? (target <= lo + 1e-6 ? hi : lo) : mid;
      slider.valueAsNumber = nudge;
      slider.valueAsNumber = target;
    }
    if (valueEl) valueEl.textContent = formatJointValue(target);
  });
  suppressPoseSend = false;
  updateSelectedJointLabel();
  highlightJointRows();
}

/** Apply MuJoCo raw joint limits from server (shape: [[low, high], ...] x 16). */
export function applyJointLimits(limits) {
  if (!Array.isArray(limits) || limits.length !== CONTROL_JOINTS.length) return;
  const next = limits.map((pair, i) => {
    const lo = Number(pair?.[0]);
    const hi = Number(pair?.[1]);
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo >= hi) {
      return [...FALLBACK_LIMITS[i]];
    }
    return [lo, hi];
  });
  const same =
    next.length === jointLimits.length &&
    next.every((pair, i) => pair[0] === jointLimits[i][0] && pair[1] === jointLimits[i][1]);
  if (same) return;
  jointLimits = next;
  currentPose = currentPose.map((v, i) => clamp(v, i));
  updateSliderValues();
}

function isDesignUiActive() {
  return Boolean(document.getElementById("designPanel")?.classList.contains("active"));
}

async function flushPoseSend() {
  if (poseSendInFlight || suppressPoseSend) return;
  if (!(currentMode === "design" || isDesignUiActive())) return;
  if (!pendingPoseSend) return;
  const { pose, seq } = pendingPoseSend;
  pendingPoseSend = null;
  poseSendInFlight = true;
  try {
    const data = await postJSON("/api/design/pose", { pose });
    if (!(seq < poseSeq && pendingPoseSend) && data.reach_distance != null) {
      onReachUpdate(data);
    }
  } catch (error) {
    onSystemMessage(`姿态同步失败: ${error.message || error}`);
  } finally {
    poseSendInFlight = false;
    if (pendingPoseSend) {
      void flushPoseSend();
    }
  }
}

function sendCurrentPose() {
  if (suppressPoseSend) return;
  if (!(currentMode === "design" || isDesignUiActive())) return;
  const pose = [...currentPose];
  onPoseLocalApply(pose);
  poseSeq += 1;
  pendingPoseSend = { pose, seq: poseSeq };
  void flushPoseSend();
}

function setPoseFromVector(vector, { send = true } = {}) {
  currentPose = poseDictToVector(vector);
  updateSliderValues();
  if (send) sendCurrentPose();
}

async function syncProjectToServer() {
  if (els.designName) {
    const n = els.designName.value.trim();
    if (n) projectName = n;
  }
  if (els.taskInput) {
    projectTask = els.taskInput.value.trim();
  }
  await postJSON("/api/design/project", {
    name: projectName || "untitled",
    task: projectTask,
    keyframes,
    playlist,
  });
}

async function loadProjectFromServer() {
  const data = await fetchJSON("/api/design/project");
  projectName = data.name || "";
  projectTask = data.task || "";
  keyframes = data.keyframes || {};
  playlist = Array.isArray(data.playlist) ? [...data.playlist] : [];
  if (els.taskInput) els.taskInput.value = projectTask;
  if (els.designName) els.designName.value = projectName;
  renderKeyframeList();
  renderPlaylist();
}

function buildJointSliders() {
  if (!els.jointSliders) return;
  els.jointSliders.innerHTML = "";

  ensureSelectedJointVisible();
  visibleJointIndices().forEach((index) => {
    const name = CONTROL_JOINTS[index];
    const row = document.createElement("div");
    row.className = "joint-row";
    row.dataset.index = String(index);

    const label = document.createElement("label");
    label.textContent = name;

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = String(jointMin(index));
    slider.max = String(jointMax(index));
    slider.step = "0.01";
    slider.value = String(currentPose[index]);

    const valueEl = document.createElement("span");
    valueEl.className = "joint-value";
    valueEl.textContent = formatJointValue(currentPose[index]);

    slider.addEventListener("input", () => {
      currentPose[index] = clamp(Number(slider.value), index);
      valueEl.textContent = formatJointValue(currentPose[index]);
      if (index === selectedJointIndex) updateSelectedJointLabel();
      sendCurrentPose();
    });

    row.addEventListener("click", (event) => {
      if (event.target === slider) return;
      selectedJointIndex = index;
      highlightJointRows();
      updateSelectedJointLabel();
    });

    row.append(label, slider, valueEl);
    els.jointSliders.appendChild(row);
  });

  highlightJointRows();
  updateSelectedJointLabel();
}

function renderKeyframeList() {
  if (!els.keyframeList) return;
  els.keyframeList.innerHTML = "";
  const names = Object.keys(keyframes).sort();
  if (!names.length) {
    const empty = document.createElement("div");
    empty.className = "design-list-item";
    empty.style.cursor = "default";
    empty.textContent = "暂无关键帧";
    els.keyframeList.appendChild(empty);
    return;
  }

  names.forEach((name) => {
    const item = document.createElement("div");
    item.className = "design-list-item";
    if (name === selectedKeyframe) item.classList.add("selected");

    const label = document.createElement("span");
    label.textContent = name;

    const actions = document.createElement("div");
    actions.className = "item-actions";

    const loadBtn = document.createElement("button");
    loadBtn.type = "button";
    loadBtn.textContent = "加载";
    loadBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      selectedKeyframe = name;
      setPoseFromVector(keyframes[name]);
      renderKeyframeList();
    });

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.textContent = "删除";
    delBtn.addEventListener("click", async (event) => {
      event.stopPropagation();
      try {
        await postJSON("/api/design/keyframes", { action: "delete", name });
        delete keyframes[name];
        playlist = playlist.filter((key) => key !== name);
        if (selectedKeyframe === name) selectedKeyframe = null;
        await postJSON("/api/design/playlist", { playlist });
        renderKeyframeList();
        renderPlaylist();
      } catch (error) {
        onSystemMessage(String(error.message || error));
      }
    });

    item.addEventListener("click", () => {
      selectedKeyframe = name;
      renderKeyframeList();
    });

    actions.append(loadBtn, delBtn);
    item.append(label, actions);
    els.keyframeList.appendChild(item);
  });
}

function renderPlaylist() {
  if (!els.playlist) return;
  els.playlist.innerHTML = "";
  if (!playlist.length) {
    const empty = document.createElement("div");
    empty.className = "design-list-item";
    empty.style.cursor = "default";
    empty.textContent = "播放列表为空";
    els.playlist.appendChild(empty);
    return;
  }

  playlist.forEach((name, index) => {
    const item = document.createElement("div");
    item.className = "design-list-item";
    if (index === selectedPlaylistIndex) item.classList.add("selected");

    const label = document.createElement("span");
    label.textContent = `${index + 1}. ${name}`;

    item.addEventListener("click", () => {
      selectedPlaylistIndex = index;
      renderPlaylist();
    });

    item.append(label);
    els.playlist.appendChild(item);
  });
}

function renderDesignList(items) {
  if (!els.designList) return;
  els.designList.innerHTML = "";
  if (!items?.length) {
    const empty = document.createElement("div");
    empty.className = "design-list-item";
    empty.style.cursor = "default";
    empty.textContent = "暂无已保存动作";
    els.designList.appendChild(empty);
    return;
  }

  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "design-list-item";
    if (item.name === selectedDesignName) row.classList.add("selected");

    const label = document.createElement("span");
    label.textContent = `${item.name} — ${item.task || ""}`;

    const actions = document.createElement("div");
    actions.className = "item-actions";

    const loadBtn = document.createElement("button");
    loadBtn.type = "button";
    loadBtn.textContent = "打开";
    loadBtn.addEventListener("click", async (event) => {
      event.stopPropagation();
      try {
        const data = await postJSON("/api/design/load", { name: item.name });
        projectName = data.name || item.name;
        projectTask = data.task || item.task || "";
        keyframes = data.keyframes || {};
        playlist = Array.isArray(data.playlist) ? [...data.playlist] : [];
        selectedDesignName = item.name;
        if (els.taskInput) els.taskInput.value = projectTask;
        if (els.designName) els.designName.value = projectName;
        renderKeyframeList();
        renderPlaylist();
        renderDesignList(items);
        if (playlist.length && keyframes[playlist[0]]) {
          setPoseFromVector(keyframes[playlist[0]]);
        }
      } catch (error) {
        onSystemMessage(String(error.message || error));
      }
    });

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.textContent = "删除";
    delBtn.addEventListener("click", async (event) => {
      event.stopPropagation();
      if (!confirm(`删除已保存动作「${item.name}」？`)) return;
      try {
        await fetchJSON(`/api/design/delete?name=${encodeURIComponent(item.name)}`, {
          method: "DELETE",
        });
        if (selectedDesignName === item.name) selectedDesignName = null;
        await refreshDesignList();
      } catch (error) {
        onSystemMessage(String(error.message || error));
      }
    });

    row.addEventListener("click", () => {
      selectedDesignName = item.name;
      renderDesignList(items);
    });

    actions.append(loadBtn, delBtn);
    row.append(label, actions);
    els.designList.appendChild(row);
  });
}

export async function refreshDesignList() {
  try {
    const data = await fetchJSON("/api/design/list");
    const items = Array.isArray(data) ? data : data.items || data.designs || [];
    renderDesignList(items);
  } catch (error) {
    onSystemMessage(String(error.message || error));
  }
}

export function setDesignVisible(visible) {
  const chatPanel = document.getElementById("chatPanel");
  const designPanel = document.getElementById("designPanel");
  document.getElementById("settingsPanel")?.classList.remove("active");
  document.getElementById("swordPanel")?.classList.remove("active");
  document.getElementById("validatePanel")?.classList.remove("active");
  document.getElementById("modeSettingsBtn")?.classList.remove("active");
  document.getElementById("modeSwordBtn")?.classList.remove("active");
  document.getElementById("modeValidateBtn")?.classList.remove("active");
  chatPanel?.classList.toggle("active", !visible);
  designPanel?.classList.toggle("active", visible);

  document.getElementById("modeChatBtn")?.classList.toggle("active", !visible);
  document.getElementById("modeDesignBtn")?.classList.toggle("active", visible);

  currentMode = visible ? "design" : "chat";
}

export function applyJointStateToDesign(jointState) {
  if (currentMode !== "design" || !jointState?.position) return;
  const vector = poseDictToVector(
    jointState.name?.length
      ? Object.fromEntries(jointState.name.map((name, i) => [name, jointState.position[i]]))
      : jointState.position
  );
  setPoseFromVector(vector, { send: false });
}

function nudgeSelectedJoint(delta) {
  currentPose[selectedJointIndex] = clamp(
    currentPose[selectedJointIndex] + delta,
    selectedJointIndex
  );
  updateSliderValues();
  sendCurrentPose();
}

function cycleJoint(direction) {
  const visible = visibleJointIndices();
  if (!visible.length) return;
  const currentVisibleIndex = Math.max(0, visible.indexOf(selectedJointIndex));
  selectedJointIndex = visible[(currentVisibleIndex + direction + visible.length) % visible.length];
  highlightJointRows();
  updateSelectedJointLabel();
  els.jointSliders
    ?.querySelector(`.joint-row[data-index="${selectedJointIndex}"]`)
    ?.scrollIntoView({ block: "nearest" });
}

function setActiveArmSide(side) {
  activeArmSide = side === "left" ? "left" : "right";
  ensureSelectedJointVisible();
  buildJointSliders();
}

async function savePlaylist() {
  await postJSON("/api/design/playlist", { playlist });
  renderPlaylist();
}

async function pollGenerateStatus() {
  try {
    const data = await fetchJSON("/api/design/generate/status");
    const lines = [];
    if (data.status) lines.push(`status: ${data.status}`);
    if (data.progress != null) lines.push(`progress: ${data.progress}`);
    if (Array.isArray(data.logs)) lines.push(...data.logs.map((row) => row.text || row));
    else if (data.message) lines.push(data.message);
    if (els.generateStatus) els.generateStatus.textContent = lines.join("\n");

    if (data.status === "running") {
      generatePollTimer = window.setTimeout(pollGenerateStatus, 1000);
    } else {
      generatePollTimer = null;
    }
  } catch (error) {
    if (els.generateStatus) els.generateStatus.textContent = String(error.message || error);
    generatePollTimer = null;
  }
}

export function setupDesignHandlers({
  wsSend: wsSendFn,
  postJSON: postJSONFn,
  fetchJSON: fetchJSONFn,
  onSystemMessage: onSystemMessageFn,
  onPoseLocalApply: onPoseLocalApplyFn,
  onReachUpdate: onReachUpdateFn,
}) {
  wsSend = wsSendFn;
  postJSON = postJSONFn;
  fetchJSON = fetchJSONFn;
  onSystemMessage = onSystemMessageFn || (() => {});
  if (typeof onPoseLocalApplyFn === "function") {
    onPoseLocalApply = onPoseLocalApplyFn;
  }
  if (typeof onReachUpdateFn === "function") {
    onReachUpdate = onReachUpdateFn;
  }
  els = {
    jointSliders: document.getElementById("jointSliders"),
    selectedJointLabel: document.getElementById("selectedJointLabel"),
    keyframeList: document.getElementById("keyframeList"),
    playlist: document.getElementById("playlist"),
    taskInput: document.getElementById("taskInput"),
    designName: document.getElementById("designName"),
    designList: document.getElementById("designList"),
    generateStatus: document.getElementById("generateStatus"),
    zeroPoseBtn: document.getElementById("zeroPoseBtn"),
    saveKeyframeBtn: document.getElementById("saveKeyframeBtn"),
    addToPlaylistBtn: document.getElementById("addToPlaylistBtn"),
    moveUpBtn: document.getElementById("moveUpBtn"),
    moveDownBtn: document.getElementById("moveDownBtn"),
    removePlaylistBtn: document.getElementById("removePlaylistBtn"),
    previewBtn: document.getElementById("previewBtn"),
    stopPreviewBtn: document.getElementById("stopPreviewBtn"),
    saveDesignBtn: document.getElementById("saveDesignBtn"),
    generateBtn: document.getElementById("generateBtn"),
    genEpisodes: document.getElementById("genEpisodes"),
    genSegmentSteps: document.getElementById("genSegmentSteps"),
    genHoldSteps: document.getElementById("genHoldSteps"),
    genPoseJitter: document.getElementById("genPoseJitter"),
    genTrimIdle: document.getElementById("genTrimIdle"),
    genOverwrite: document.getElementById("genOverwrite"),
  };

  buildJointSliders();
  window.addEventListener("robot:arm-side", (event) => {
    setActiveArmSide(event.detail?.side);
  });

  els.zeroPoseBtn?.addEventListener("click", () => {
    setPoseFromVector(new Array(CONTROL_JOINTS.length).fill(0));
  });

  els.saveKeyframeBtn?.addEventListener("click", async () => {
    const name = prompt("关键帧名称", selectedKeyframe || "pose_1");
    if (!name?.trim()) return;
    const trimmed = name.trim();
    try {
      const pose = vectorToPoseDict(currentPose);
      await postJSON("/api/design/keyframes", { action: "upsert", name: trimmed, pose });
      keyframes[trimmed] = pose;
      selectedKeyframe = trimmed;
      // 方便预览：播放列表为空时自动加入该关键帧。
      if (!playlist.length) {
        playlist.push(trimmed);
      }
      await syncProjectToServer();
      renderKeyframeList();
      renderPlaylist();
    } catch (error) {
      onSystemMessage(String(error.message || error));
    }
  });

  els.addToPlaylistBtn?.addEventListener("click", async () => {
    if (!selectedKeyframe) {
      onSystemMessage("请先在关键帧库中选中一个关键帧");
      return;
    }
    playlist.push(selectedKeyframe);
    try {
      await savePlaylist();
    } catch (error) {
      playlist.pop();
      onSystemMessage(String(error.message || error));
    }
  });

  els.moveUpBtn?.addEventListener("click", async () => {
    if (selectedPlaylistIndex <= 0) return;
    const tmp = playlist[selectedPlaylistIndex - 1];
    playlist[selectedPlaylistIndex - 1] = playlist[selectedPlaylistIndex];
    playlist[selectedPlaylistIndex] = tmp;
    selectedPlaylistIndex -= 1;
    try {
      await savePlaylist();
    } catch (error) {
      onSystemMessage(String(error.message || error));
    }
  });

  els.moveDownBtn?.addEventListener("click", async () => {
    if (selectedPlaylistIndex < 0 || selectedPlaylistIndex >= playlist.length - 1) return;
    const tmp = playlist[selectedPlaylistIndex + 1];
    playlist[selectedPlaylistIndex + 1] = playlist[selectedPlaylistIndex];
    playlist[selectedPlaylistIndex] = tmp;
    selectedPlaylistIndex += 1;
    try {
      await savePlaylist();
    } catch (error) {
      onSystemMessage(String(error.message || error));
    }
  });

  els.removePlaylistBtn?.addEventListener("click", async () => {
    if (selectedPlaylistIndex < 0) return;
    playlist.splice(selectedPlaylistIndex, 1);
    selectedPlaylistIndex = Math.min(selectedPlaylistIndex, playlist.length - 1);
    try {
      await savePlaylist();
    } catch (error) {
      onSystemMessage(String(error.message || error));
    }
  });

  els.previewBtn?.addEventListener("click", async () => {
    try {
      await syncProjectToServer();
      if (!playlist.length) {
        onSystemMessage("播放列表为空：请先把关键帧加入播放列表，再点预览");
        return;
      }
      const segmentSteps = Number(els.genSegmentSteps?.value) || 20;
      const result = await postJSON("/api/design/preview/start", { segment_steps: segmentSteps });
      onSystemMessage(`预览开始（${result.frames || "?"} 帧）`);
    } catch (error) {
      onSystemMessage(String(error.message || error));
    }
  });

  els.stopPreviewBtn?.addEventListener("click", async () => {
    try {
      await postJSON("/api/design/preview/stop");
    } catch (error) {
      onSystemMessage(String(error.message || error));
    }
  });

  els.saveDesignBtn?.addEventListener("click", async () => {
    const name = els.designName?.value.trim();
    const task = els.taskInput?.value.trim();
    if (!name || !task) {
      onSystemMessage("请填写动作名称和英文任务描述");
      return;
    }
    try {
      projectName = name;
      projectTask = task;
      await syncProjectToServer();
      await postJSON("/api/design/save", { name, task });
      selectedDesignName = name;
      await refreshDesignList();
      onSystemMessage(`已保存动作：${name}`);
    } catch (error) {
      onSystemMessage(String(error.message || error));
    }
  });

  els.generateBtn?.addEventListener("click", async () => {
    const name = selectedDesignName || els.designName?.value.trim();
    if (!name) {
      onSystemMessage("请先选择或填写要生成的动作名称");
      return;
    }
    try {
      await postJSON("/api/design/generate", {
        name,
        episodes: Number(els.genEpisodes?.value ?? 30),
        segment_steps: Number(els.genSegmentSteps?.value ?? 20),
        hold_steps: Number(els.genHoldSteps?.value ?? 2),
        pose_jitter_std: Number(els.genPoseJitter?.value ?? 0.05),
        trim_leading_idle: els.genTrimIdle?.checked !== false,
        idle_action_tolerance: 1e-4,
        midpoint_noise_std: 0,
        hold_noise_std: 0,
        overwrite: Boolean(els.genOverwrite?.checked),
      });
      if (generatePollTimer) clearTimeout(generatePollTimer);
      pollGenerateStatus();
    } catch (error) {
      onSystemMessage(String(error.message || error));
    }
  });

  document.addEventListener("keydown", (event) => {
    if (currentMode !== "design") return;
    const tag = event.target?.tagName?.toLowerCase();
    if (tag === "input" || tag === "textarea") return;

    if (event.key === "ArrowLeft") {
      event.preventDefault();
      nudgeSelectedJoint(-NUDGE_STEP);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      nudgeSelectedJoint(NUDGE_STEP);
    } else if (event.key === "[") {
      event.preventDefault();
      cycleJoint(-1);
    } else if (event.key === "]") {
      event.preventDefault();
      cycleJoint(1);
    }
  });

  loadProjectFromServer()
    .then(async () => {
      try {
        const poseData = await fetchJSON("/api/design/pose");
        if (poseData.limits) applyJointLimits(poseData.limits);
        if (poseData.pose) {
          setPoseFromVector(poseData.pose, { send: false });
        } else if (Array.isArray(poseData.position)) {
          setPoseFromVector(poseData.position, { send: false });
        }
      } catch (_err) {
        /* pose endpoint may not exist yet */
      }
      await refreshDesignList();
    })
    .catch((error) => {
      onSystemMessage(String(error.message || error));
    });
}

export async function switchMode(mode) {
  await postJSON("/api/mode", { mode });
  setDesignVisible(mode === "design");
  if (mode === "design") {
    await loadProjectFromServer();
    try {
      const data = await fetchJSON("/api/design/pose");
      if (Array.isArray(data.pose) && data.pose.length === CONTROL_JOINTS.length) {
        setPoseFromVector(data.pose, { send: false });
      }
      if (data.reach_distance != null) {
        onReachUpdate(data);
      }
    } catch (error) {
      onSystemMessage(`同步当前姿态失败: ${error.message || error}`);
    }
    sendCurrentPose();
  }
}
