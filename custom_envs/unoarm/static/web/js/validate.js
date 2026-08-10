/** Post-training reach validation tab (sword + shield jitter). */

function num(id, fallback) {
  const el = document.getElementById(id);
  if (!el || el.value === "") return fallback;
  const v = Number(el.value);
  return Number.isFinite(v) ? v : fallback;
}

function fmtHandle(v) {
  if (!Array.isArray(v)) return "--";
  return v.map((x) => Number(x).toFixed(3)).join(", ");
}

export function setupValidateHandlers({ postJSON, fetchJSON, onSystemMessage }) {
  const startBtn = document.getElementById("validateStartBtn");
  const stopBtn = document.getElementById("validateStopBtn");
  const statusEl = document.getElementById("validateStatus");
  const resultsEl = document.getElementById("validateResults");
  const policyEl = document.getElementById("validatePolicyHint");

  async function refreshMeta() {
    try {
      const settings = await fetchJSON("/api/settings");
      if (policyEl) {
        const ckpt = settings.checkpoint || "(未配置)";
        const ready = settings.policy_ready ? "已加载" : "未加载";
        policyEl.textContent = `策略：${ready} · ${ckpt}`;
      }
      const taskInput = document.getElementById("validate_task");
      if (taskInput && !taskInput.value) {
        taskInput.value = "Grasp the shield with the left arm and the sword with the right arm";
      }
    } catch (error) {
      if (policyEl) policyEl.textContent = String(error.message || error);
    }
  }

  function renderStatus(data) {
    if (!data) return;
    const rate =
      data.success_rate != null ? `${(100 * Number(data.success_rate)).toFixed(0)}%` : "--";
    const handle = fmtHandle(data.current_handle);
    const shield = fmtHandle(data.current_shield_handle);
    if (statusEl) {
      statusEl.textContent =
        `状态 ${data.state || "--"} · ${data.message || ""}\n` +
        `进度 ${data.episode || 0}/${data.episodes || 0} · 成功 ${data.successes || 0} · 成功率 ${rate}\n` +
        `当前剑红点 [${handle}]\n` +
        `当前盾红点 [${shield}]`;
    }
    if (resultsEl && Array.isArray(data.results)) {
      const lines = data.results.map((row) => {
        const dist =
          row.reach_distance != null ? `${(100 * Number(row.reach_distance)).toFixed(1)}cm` : "--";
        const h = Array.isArray(row.handle)
          ? row.handle.map((x) => Number(x).toFixed(2)).join(",")
          : "";
        const sh = Array.isArray(row.shield_handle)
          ? row.shield_handle.map((x) => Number(x).toFixed(2)).join(",")
          : "";
        return (
          `#${row.episode} ${row.success ? "OK" : "FAIL"} steps=${row.steps} dist=${dist} ` +
          `sword=[${h}] shield=[${sh}]`
        );
      });
      resultsEl.textContent = lines.length ? lines.join("\n") : "尚无结果";
    }
    const running = data.state === "running";
    if (startBtn) startBtn.disabled = running;
    if (stopBtn) stopBtn.disabled = !running;
  }

  async function start() {
    const payload = {
      episodes: Math.max(1, Math.min(200, Math.round(num("validate_episodes", 10)))),
      max_steps: Math.max(1, Math.round(num("validate_max_steps", 150))),
      seed: Math.round(num("validate_seed", 0)),
      task: document.getElementById("validate_task")?.value?.trim() || null,
      handle_jitter: [
        Math.max(0, num("validate_jx", 0.15)),
        Math.max(0, num("validate_jy", 0.12)),
        Math.max(0, num("validate_jz", 0.08)),
      ],
      randomize_euler: Boolean(document.getElementById("validate_rand_euler")?.checked),
      shield_handle_jitter: [
        Math.max(0, num("validate_shield_jx", 0.15)),
        Math.max(0, num("validate_shield_jy", 0.12)),
        Math.max(0, num("validate_shield_jz", 0.08)),
      ],
      randomize_shield_euler: Boolean(
        document.getElementById("validate_rand_shield_euler")?.checked
      ),
    };
    try {
      const data = await postJSON("/api/validate/start", payload);
      renderStatus(data);
      onSystemMessage(`开始验证：${payload.episodes} 局（剑+盾红点随机）`);
    } catch (error) {
      onSystemMessage(`验证启动失败: ${error.message || error}`);
      if (statusEl) statusEl.textContent = String(error.message || error);
    }
  }

  async function stop() {
    try {
      const data = await postJSON("/api/validate/stop", {});
      renderStatus(data);
      onSystemMessage("已请求停止验证");
    } catch (error) {
      onSystemMessage(`停止验证失败: ${error.message || error}`);
    }
  }

  startBtn?.addEventListener("click", (event) => {
    event.preventDefault();
    start();
  });
  stopBtn?.addEventListener("click", (event) => {
    event.preventDefault();
    stop();
  });

  return {
    loadCurrent: refreshMeta,
    applyValidateSnapshot: renderStatus,
  };
}

export function setValidateVisible(visible) {
  document.getElementById("validatePanel")?.classList.toggle("active", visible);
  document.getElementById("modeValidateBtn")?.classList.toggle("active", visible);
}
