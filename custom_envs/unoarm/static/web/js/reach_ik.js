/** Reach-IK generate + 3D replay tab */

const EXAMPLE_TARGETS_JSON = `{
  "task": "Grasp the sword handle",
  "targets": [
    {"xyz": [-0.35, -0.55, 1.00]},
    {"xyz": [-0.30, -0.50, 1.05]},
    {"xyz": [-0.25, -0.55, 1.02]}
  ]
}
`;

function num(id, fallback = 0) {
  const el = document.getElementById(id);
  if (!el || el.value === "") return fallback;
  const v = Number(el.value);
  return Number.isFinite(v) ? v : fallback;
}

function currentMode() {
  const bbox = document.getElementById("rik_mode_bbox");
  return bbox?.checked ? "bbox" : "json";
}

function syncModeBlocks() {
  const mode = currentMode();
  const jsonBlock = document.getElementById("rikJsonBlock");
  const bboxBlock = document.getElementById("rikBboxBlock");
  if (jsonBlock) jsonBlock.hidden = mode !== "json";
  if (bboxBlock) bboxBlock.hidden = mode !== "bbox";
}

function syncReachIkSceneUi(scene) {
  const isTable = scene === "table_place";
  const sword = document.getElementById("rikSwordBlocks");
  const table = document.getElementById("rikTablePlaceBlock");
  const hintSword = document.getElementById("rikHintSword");
  const hintTable = document.getElementById("rikHintTable");
  if (sword) sword.hidden = isTable;
  if (table) table.hidden = !isTable;
  if (hintSword) hintSword.hidden = isTable;
  if (hintTable) hintTable.hidden = !isTable;
  const out = document.getElementById("rik_output_name");
  if (out && isTable && (!out.value || out.value === "reach_ik_web")) {
    out.value = "table_place_ik_web";
  }
  if (out && !isTable && out.value === "table_place_ik_web") {
    out.value = "reach_ik_web";
  }
  if (!isTable) syncModeBlocks();
}

export function setupReachIkHandlers({ postJSON, fetchJSON, onSystemMessage }) {
  const genStatusEl = document.getElementById("rikGenerateStatus");
  const replayStatusEl = document.getElementById("rikReplayStatus");
  const datasetSelect = document.getElementById("rik_dataset_select");
  const genBtn = document.getElementById("rikGenerateBtn");
  const playBtn = document.getElementById("rikPlayBtn");
  const pauseBtn = document.getElementById("rikPauseBtn");
  const stopBtn = document.getElementById("rikStopBtn");

  let currentScene = "reach_sword";

  async function refreshSceneUi() {
    try {
      const settings = await fetchJSON("/api/settings");
      currentScene = settings?.scene || "reach_sword";
      syncReachIkSceneUi(currentScene);
      if (currentScene === "table_place" && genStatusEl && genStatusEl.textContent.includes("下一阶段")) {
        genStatusEl.textContent = "桌面抓放 Reach-IK 已就绪：配置参数后点击「开始生成」。";
      }
    } catch {
      /* ignore */
    }
  }
  refreshSceneUi();

  let pollGen = null;
  let pollReplay = null;
  let paused = false;

  function setGenUi(running) {
    if (genBtn) genBtn.disabled = running;
  }

  function setReplayUi(state) {
    const playing = state === "playing";
    const isPaused = state === "paused";
    if (playBtn) playBtn.disabled = playing;
    if (pauseBtn) pauseBtn.disabled = !(playing || isPaused);
    if (stopBtn) stopBtn.disabled = !(playing || isPaused || state === "done");
    if (pauseBtn) pauseBtn.textContent = isPaused ? "继续" : "暂停";
  }

  function renderGenStatus(data) {
    if (!data || !genStatusEl) return;
    const logs = Array.isArray(data.logs) ? data.logs.slice(-30).join("\n") : "";
    genStatusEl.textContent =
      `状态 ${data.state || "--"} · ${data.message || ""}\n` +
      (data.output_root ? `输出 ${data.output_root}\n` : "") +
      (logs ? `\n${logs}` : "");
    setGenUi(data.state === "running");
  }

  function renderReplayStatus(data) {
    if (!data || !replayStatusEl) return;
    replayStatusEl.textContent =
      `状态 ${data.state || "--"} · ${data.message || ""}\n` +
      `episode ${(data.episode ?? 0) + 1}/${data.n_episodes ?? "--"} · ` +
      `frame ${data.frame ?? 0}/${data.n_frames ?? 0}\n` +
      (data.peg_xy
        ? `peg_xy [${data.peg_xy.map((x) => Number(x).toFixed(3)).join(", ")}]\n`
        : "") +
      (data.handle ? `handle [${data.handle.map((x) => Number(x).toFixed(3)).join(", ")}]\n` : "") +
      (data.root ? `root ${data.root}` : "");
    setReplayUi(data.state || "idle");
    paused = Boolean(data.paused);
  }

  async function refreshDatasets() {
    try {
      const data = await fetchJSON("/api/reach-ik/datasets");
      const items = Array.isArray(data.datasets) ? data.datasets : [];
      if (!datasetSelect) return;
      const prev = datasetSelect.value;
      datasetSelect.innerHTML = "";
      if (!items.length) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "(暂无数据集)";
        datasetSelect.appendChild(opt);
        return;
      }
      for (const item of items) {
        const opt = document.createElement("option");
        opt.value = item.root;
        const n = item.n_episodes != null ? item.n_episodes : "?";
        opt.textContent = `${item.name} · ${n} ep`;
        datasetSelect.appendChild(opt);
      }
      if (prev && [...datasetSelect.options].some((o) => o.value === prev)) {
        datasetSelect.value = prev;
      }
      await onDatasetChange();
    } catch (error) {
      onSystemMessage(`刷新 Reach-IK 数据集失败: ${error.message || error}`);
    }
  }

  async function onDatasetChange() {
    const root = datasetSelect?.value;
    if (!root) return;
    try {
      const info = await fetchJSON(`/api/reach-ik/datasets/info?root=${encodeURIComponent(root)}`);
      const n = Number(info.n_episodes || 0);
      const epInput = document.getElementById("rik_episode");
      if (epInput) {
        epInput.max = String(Math.max(0, n - 1));
        if (Number(epInput.value) >= n) epInput.value = "0";
      }
      const meta = info.meta || {};
      const skipped = meta.skipped_targets != null ? meta.skipped_targets : null;
      const success = meta.success_targets != null ? meta.success_targets : null;
      if (replayStatusEl) {
        let extra = "";
        if (success != null || skipped != null) {
          extra = `\nsuccess_targets=${success ?? "?"} skipped=${skipped ?? "?"}`;
        }
        replayStatusEl.textContent = `已选 ${root}\nepisodes=${n}${extra}\n播放将从所选 Episode 顺序播到末尾`;
      }
    } catch (error) {
      if (replayStatusEl) replayStatusEl.textContent = String(error.message || error);
    }
  }

  function startGenPoll() {
    stopGenPoll();
    pollGen = setInterval(async () => {
      try {
        const data = await fetchJSON("/api/reach-ik/generate/status");
        renderGenStatus(data);
        if (data.state === "done" || data.state === "failed") {
          stopGenPoll();
          if (data.state === "done") {
            onSystemMessage(`Reach-IK 生成完成: ${data.output_root || ""}`);
            refreshDatasets();
          }
        }
      } catch {
        /* ignore transient */
      }
    }, 800);
  }

  function stopGenPoll() {
    if (pollGen) {
      clearInterval(pollGen);
      pollGen = null;
    }
  }

  function startReplayPoll() {
    stopReplayPoll();
    pollReplay = setInterval(async () => {
      try {
        const data = await fetchJSON("/api/reach-ik/replay/status");
        renderReplayStatus(data);
        if (data.state === "done" || data.state === "idle" || data.state === "error") {
          if (data.state !== "paused") stopReplayPoll();
        }
      } catch {
        /* ignore */
      }
    }, 400);
  }

  function stopReplayPoll() {
    if (pollReplay) {
      clearInterval(pollReplay);
      pollReplay = null;
    }
  }

  document.querySelectorAll('input[name="rik_mode"]').forEach((el) => {
    el.addEventListener("change", syncModeBlocks);
  });
  syncModeBlocks();

  document.getElementById("rikLoadExampleBtn")?.addEventListener("click", () => {
    const ta = document.getElementById("rik_targets_json");
    if (ta) ta.value = EXAMPLE_TARGETS_JSON.trim();
  });

  // Prefill example once if empty
  const ta0 = document.getElementById("rik_targets_json");
  if (ta0 && !ta0.value.trim()) ta0.value = EXAMPLE_TARGETS_JSON.trim();

  genBtn?.addEventListener("click", async () => {
    let payload;
    if (currentScene === "table_place") {
      payload = {
        mode: "table_place",
        num_episodes: Math.max(1, Math.round(num("rik_tp_num_episodes", 20))),
        approach_offset_y_min: num("rik_tp_approach_min", 0.04),
        approach_offset_y_max: num("rik_tp_approach_max", 0.10),
        lift_z_min: num("rik_tp_lift_min", 0.905),
        lift_z_max: num("rik_tp_lift_max", 0.985),
        place_z_min: num("rik_tp_place_min", 0.895),
        place_z_max: num("rik_tp_place_max", 0.975),
        table_clearance_m: Math.max(0, num("rik_tp_clearance", 0.01)),
        settle_steps: Math.max(1, Math.round(num("rik_tp_settle", 30))),
        segment_steps: Math.max(1, Math.round(num("rik_segment", 20))),
        hold_steps: Math.max(0, Math.round(num("rik_hold", 2))),
        seed: Math.round(num("rik_seed", 0)),
        sample_peg_xy: Boolean(document.getElementById("rik_tp_sample_peg")?.checked),
        append: Boolean(document.getElementById("rik_tp_append")?.checked),
        peg_x_min: num("rik_tp_peg_xmin", -0.3),
        peg_x_max: num("rik_tp_peg_xmax", 0.08),
        peg_y_min: num("rik_tp_peg_ymin", -0.7),
        peg_y_max: num("rik_tp_peg_ymax", -0.5),
        peg_edge_margin: Math.max(0, num("rik_tp_peg_margin", 0.05)),
        episodes_per_peg: Math.max(1, Math.round(num("rik_tp_episodes_per_peg", 2))),
        output_name:
          document.getElementById("rik_output_name")?.value?.trim() || "table_place_ik_web",
        overwrite: Boolean(document.getElementById("rik_overwrite")?.checked),
        task: "Pick up the cylinder and place it on the target circle",
      };
      if (payload.append) payload.overwrite = false;
    } else {
      const mode = currentMode();
      payload = {
        mode,
        episodes_per_target: Math.max(1, Math.round(num("rik_episodes", 1))),
        segment_steps: Math.max(1, Math.round(num("rik_segment", 20))),
        hold_steps: Math.max(0, Math.round(num("rik_hold", 2))),
        seed: Math.round(num("rik_seed", 0)),
        output_name: document.getElementById("rik_output_name")?.value?.trim() || "reach_ik_web",
        overwrite: Boolean(document.getElementById("rik_overwrite")?.checked),
        task: "Grasp the sword handle",
      };
      if (mode === "json") {
        payload.targets_json_text = document.getElementById("rik_targets_json")?.value || "";
      } else {
        payload.bbox_min = [num("rik_bmin_x"), num("rik_bmin_y"), num("rik_bmin_z")];
        payload.bbox_max = [num("rik_bmax_x"), num("rik_bmax_y"), num("rik_bmax_z")];
        payload.num_targets = Math.max(1, Math.round(num("rik_num_targets", 20)));
      }
    }
    try {
      setGenUi(true);
      if (genStatusEl) genStatusEl.textContent = "提交生成任务…";
      const data = await postJSON("/api/reach-ik/generate", payload);
      renderGenStatus({ ...data, logs: [] });
      startGenPoll();
      onSystemMessage("Reach-IK 生成已开始");
    } catch (error) {
      setGenUi(false);
      if (genStatusEl) genStatusEl.textContent = String(error.message || error);
      onSystemMessage(`Reach-IK 生成失败: ${error.message || error}`);
    }
  });

  document.getElementById("rikRefreshDatasetsBtn")?.addEventListener("click", () => {
    refreshDatasets();
  });
  datasetSelect?.addEventListener("change", () => onDatasetChange().catch(() => {}));

  playBtn?.addEventListener("click", async () => {
    const root = datasetSelect?.value;
    if (!root) {
      onSystemMessage("请先选择数据集");
      return;
    }
    try {
      paused = false;
      const data = await postJSON("/api/reach-ik/replay/start", {
        root,
        episode: Math.max(0, Math.round(num("rik_episode", 0))),
        speed: Math.max(0.1, num("rik_speed", 1.0)),
      });
      renderReplayStatus(data);
      startReplayPoll();
      onSystemMessage("Reach-IK 回放开始");
    } catch (error) {
      if (replayStatusEl) replayStatusEl.textContent = String(error.message || error);
      onSystemMessage(`回放失败: ${error.message || error}`);
    }
  });

  pauseBtn?.addEventListener("click", async () => {
    try {
      const next = !paused;
      const data = await postJSON("/api/reach-ik/replay/pause", { paused: next });
      renderReplayStatus(data);
      if (!next) startReplayPoll();
    } catch (error) {
      onSystemMessage(`暂停失败: ${error.message || error}`);
    }
  });

  stopBtn?.addEventListener("click", async () => {
    try {
      const data = await postJSON("/api/reach-ik/replay/stop", {});
      renderReplayStatus(data);
      stopReplayPoll();
    } catch (error) {
      onSystemMessage(`停止回放失败: ${error.message || error}`);
    }
  });

  async function loadCurrent() {
    syncModeBlocks();
    await refreshSceneUi();
    await refreshDatasets();
    try {
      const g = await fetchJSON("/api/reach-ik/generate/status");
      renderGenStatus(g);
      if (g.state === "running") startGenPoll();
      const r = await fetchJSON("/api/reach-ik/replay/status");
      renderReplayStatus(r);
      if (r.state === "playing" || r.state === "paused") startReplayPoll();
    } catch {
      /* ignore */
    }
  }

  function cleanup() {
    stopGenPoll();
    stopReplayPoll();
    postJSON("/api/reach-ik/replay/stop", {}).catch(() => {});
  }

  return { loadCurrent, cleanup, refreshDatasets, setVisible: setReachIkVisible, refreshSceneUi };
}

export function setReachIkVisible(visible) {
  document.getElementById("reachIkPanel")?.classList.toggle("active", visible);
  document.getElementById("modeReachIkBtn")?.classList.toggle("active", visible);
}
