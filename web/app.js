/* Kabushiki Multi-Pattern Parallel Dashboard (live FastAPI + static GitHub Pages viewer) */

const SERIES_COLORS = ["#22c55e", "#3b82f6", "#f59e0b", "#ec4899", "#a855f7", "#14b8a6", "#f97316", "#eab308"];

const state = {
  apiAvailable: false,
  githubConnected: false,
  githubConfig: { owner: "", repo: "", token: "", branch: "main" },
  encryptedMode: false,
  encryptedEnvelope: null,
  selectedPatternId: null,
  dashboard: null,
  lastCompareResults: [],
};

const SESSION_KEY = "kabushiki_unlock_pass";
const GH_SESSION_KEY = "kabushiki_gh_edit_session";
const GH_LOCAL_KEY = "kabushiki_gh_edit_local";

function canEditPatterns() {
  return state.apiAvailable || state.githubConnected;
}

function detectDefaultGithubRepo() {
  const host = window.location.hostname || "";
  const pathParts = (window.location.pathname || "").split("/").filter(Boolean);
  let owner = "shuyabuuchi-spec";
  let repo = "kabushiki";
  if (host.endsWith(".github.io")) {
    owner = host.replace(/\.github\.io$/, "");
    if (pathParts.length > 0) repo = pathParts[0];
  }
  return { owner, repo };
}

function utf8ToBase64(str) {
  const bytes = new TextEncoder().encode(str);
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

function base64ToUtf8(b64) {
  const clean = String(b64 || "").replace(/\s+/g, "");
  return new TextDecoder().decode(b64ToBytes(clean));
}

function slugifyPatternIdJS(raw, existingIds) {
  let cleaned = String(raw || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  if (!cleaned) cleaned = "pattern";
  if (!cleaned.startsWith("p_")) cleaned = `p_${cleaned}`;
  cleaned = cleaned.slice(0, 40);
  const used = new Set(existingIds || []);
  if (!used.has(cleaned)) return cleaned;
  let idx = 2;
  while (used.has(`${cleaned}_${idx}`)) idx++;
  return `${cleaned}_${idx}`;
}

function loadSavedGithubConfig() {
  try {
    const raw = sessionStorage.getItem(GH_SESSION_KEY) || localStorage.getItem(GH_LOCAL_KEY);
    if (!raw) return false;
    const parsed = JSON.parse(raw);
    if (parsed && parsed.owner && parsed.repo && parsed.token) {
      state.githubConfig = { owner: parsed.owner, repo: parsed.repo, token: parsed.token, branch: "main" };
      state.githubConnected = true;
      return true;
    }
  } catch {
    // ignore corrupted storage
  }
  return false;
}

async function ghApiFetch(path, options = {}) {
  const { owner, repo, token } = state.githubConfig;
  const url = `https://api.github.com/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}${path}`;
  const resp = await fetch(url, {
    ...options,
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${token}`,
      "X-GitHub-Api-Version": "2022-11-28",
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  });
  return resp;
}

async function ghLoadRepoConfig() {
  // Try data/user_config.json first, fallback to config/default_config.json
  let sha = null;
  let cfgObj = null;
  const rUser = await ghApiFetch(`/contents/data/user_config.json?ref=${state.githubConfig.branch}`);
  if (rUser.ok) {
    const j = await rUser.json();
    sha = j.sha;
    cfgObj = JSON.parse(base64ToUtf8(j.content));
  }
  const rDef = await ghApiFetch(`/contents/config/default_config.json?ref=${state.githubConfig.branch}`);
  if (!rDef.ok && !cfgObj) {
    throw new Error(`リポジトリ設定の取得に失敗しました (HTTP ${rDef.status})`);
  }
  const defObj = rDef.ok ? JSON.parse(base64ToUtf8((await rDef.json()).content)) : {};
  const merged = { ...defObj, ...(cfgObj || {}) };
  if (!merged.patterns || !Array.isArray(merged.patterns)) merged.patterns = defObj.patterns || [];
  if (!merged.watchlists) merged.watchlists = defObj.watchlists || {};
  return { cfg: merged, sha };
}

async function ghSaveRepoConfig(cfgObj, sha, commitMessage) {
  const body = {
    message: commitMessage,
    content: utf8ToBase64(JSON.stringify(cfgObj, null, 2) + "\n"),
    branch: state.githubConfig.branch,
  };
  if (sha) body.sha = sha;
  const resp = await ghApiFetch("/contents/data/user_config.json", {
    method: "PUT",
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.message || `GitHub保存失敗 (HTTP ${resp.status})`);
  }
  return await resp.json();
}

async function ghDispatchMonitorWorkflow(inputs = {}) {
  const resp = await ghApiFetch("/actions/workflows/monitor_and_simulate.yml/dispatches", {
    method: "POST",
    body: JSON.stringify({
      ref: state.githubConfig.branch,
      inputs: {
        force_run: "true",
        reset_pattern: inputs.reset_pattern || "",
        collect_ml: inputs.collect_ml ? "true" : "false",
      },
    }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.message || `Actions起動失敗 (HTTP ${resp.status})。トークンに Actions: Read and write 権限があるか確認してください。`);
  }
  return true;
}

function syncActionButtonsForMode() {
  const editable = canEditPatterns();
  const isStatic = !state.apiAvailable;
  const ghBtn = document.getElementById("btnGithubConnect");
  if (ghBtn) {
    ghBtn.classList.toggle("hidden", !isStatic);
    ghBtn.textContent = state.githubConnected ? "☁ GitHub連携中" : "☁ GitHub編集連携";
  }

  ["btnAddPattern", "btnAddPatternTop", "btnEditSelected", "btnResetSelected", "btnCatchup", "btnCollectMl"].forEach(
    (id) => {
      const el = document.getElementById(id);
      if (el) el.disabled = !editable;
    }
  );
  // Real-time interactive backtest requires local Python server
  const btBtn = document.getElementById("btnRunBacktest");
  if (btBtn) btBtn.disabled = !state.apiAvailable;

  if (isStatic && state.githubConnected) {
    const connBadge = document.getElementById("connectionBadge");
    connBadge.textContent = state.encryptedMode
      ? "🔒 暗号化＋☁GitHub編集連携中"
      : "☁ GitHub編集連携中";
    connBadge.className = "pill pill-ok";
  }
}

function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function decryptDashboardEnvelope(envelope, passphrase) {
  if (!window.crypto || !window.crypto.subtle) {
    throw new Error("このブラウザは Web Crypto API (HTTPS) に対応していません。");
  }
  const enc = new TextEncoder();
  const salt = b64ToBytes(envelope.salt);
  const iv = b64ToBytes(envelope.iv);
  const ciphertext = b64ToBytes(envelope.ciphertext);
  const iterations = Number(envelope.iterations || 310000);

  const keyMaterial = await window.crypto.subtle.importKey(
    "raw",
    enc.encode(passphrase),
    "PBKDF2",
    false,
    ["deriveKey"]
  );
  const aesKey = await window.crypto.subtle.deriveKey(
    { name: "PBKDF2", salt, iterations, hash: "SHA-256" },
    keyMaterial,
    { name: "AES-GCM", length: 256 },
    false,
    ["decrypt"]
  );
  const plainBuffer = await window.crypto.subtle.decrypt(
    { name: "AES-GCM", iv },
    aesKey,
    ciphertext
  );
  const jsonText = new TextDecoder().decode(plainBuffer);
  return JSON.parse(jsonText);
}

function setStaticButtonsDisabled() {
  loadSavedGithubConfig();
  syncActionButtonsForMode();
}

async function tryUnlockWithPassphrase(passphrase) {
  if (!state.encryptedEnvelope) return false;
  const decrypted = await decryptDashboardEnvelope(state.encryptedEnvelope, passphrase);
  state.dashboard = decrypted;
  sessionStorage.setItem(SESSION_KEY, passphrase);
  document.getElementById("loginOverlay").classList.add("hidden");
  document.getElementById("loginError").classList.add("hidden");
  document.getElementById("btnLogout").classList.remove("hidden");
  const connBadge = document.getElementById("connectionBadge");
  connBadge.textContent = "🔒 暗号化ビューア（認証済み）";
  connBadge.className = "pill pill-ok";
  setStaticButtonsDisabled();
  renderDashboard();
  return true;
}

function handleLogout() {
  sessionStorage.removeItem(SESSION_KEY);
  state.dashboard = null;
  document.getElementById("btnLogout").classList.add("hidden");
  document.getElementById("loginPassphrase").value = "";
  document.getElementById("loginError").classList.add("hidden");
  document.getElementById("loginOverlay").classList.remove("hidden");
  document.getElementById("loginPassphrase").focus();
}

const fmtJPY = (n) => "¥" + Math.round(Number(n || 0)).toLocaleString("ja-JP");
const fmtPrice = (n) => "¥" + Number(n || 0).toLocaleString("ja-JP", { maximumFractionDigits: 1 });
const fmtSignedJPY = (n) => {
  const num = Number(n || 0);
  return `${num > 0 ? "+" : ""}¥${Math.round(num).toLocaleString("ja-JP")}`;
};
const fmtPct = (n) => {
  const num = Number(n || 0);
  return `${num > 0 ? "+" : ""}${num.toFixed(2)}%`;
};
const fmtTime = (iso) => {
  if (!iso) return "--";
  try {
    return new Date(iso).toLocaleString("ja-JP", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      timeZone: "Asia/Tokyo",
    });
  } catch {
    return iso;
  }
};

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function showToast(msg, isError = false) {
  const el = document.getElementById("toastArea");
  el.textContent = msg;
  el.classList.remove("hidden", "error");
  if (isError) el.classList.add("error");
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => el.classList.add("hidden"), 6500);
}

function getPatternList() {
  const dash = state.dashboard;
  if (!dash) return [];
  if (dash.patterns && dash.pattern_order) {
    return dash.pattern_order.map((id) => dash.patterns[id]).filter(Boolean);
  }
  if (dash.modes) {
    return Object.values(dash.modes);
  }
  return [];
}

function getSelectedPatternData() {
  const list = getPatternList();
  if (list.length === 0) return null;
  if (state.selectedPatternId) {
    const found = list.find((p) => (p.pattern_id || p.mode) === state.selectedPatternId);
    if (found) return found;
  }
  state.selectedPatternId = list[0].pattern_id || list[0].mode;
  return list[0];
}

function renderEquitySVG(containerId, points, initialCapital = 100000) {
  const box = document.getElementById(containerId);
  if (!points || points.length === 0) {
    box.innerHTML = `<div class="empty-state">資産推移データはまだありません。スキャンまたはバックテストを実行してください。</div>`;
    return;
  }

  const width = 760;
  const height = 250;
  const pad = { top: 20, right: 20, bottom: 32, left: 72 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const values = points.map((p) => Number(p.total_equity));
  const minVal = Math.min(...values, initialCapital) * 0.998;
  const maxVal = Math.max(...values, initialCapital) * 1.002;
  const range = Math.max(maxVal - minVal, 100);

  const xAt = (i) => (points.length === 1 ? pad.left + plotW / 2 : pad.left + (i / (points.length - 1)) * plotW);
  const yAt = (v) => pad.top + plotH - ((v - minVal) / range) * plotH;

  const lineCoords = points.map((p, i) => `${xAt(i).toFixed(1)},${yAt(Number(p.total_equity)).toFixed(1)}`);
  const areaPath =
    `M ${xAt(0).toFixed(1)},${(pad.top + plotH).toFixed(1)} ` +
    points.map((p, i) => `L ${xAt(i).toFixed(1)},${yAt(Number(p.total_equity)).toFixed(1)}`).join(" ") +
    ` L ${xAt(points.length - 1).toFixed(1)},${(pad.top + plotH).toFixed(1)} Z`;

  const lastVal = values[values.length - 1];
  const strokeColor = lastVal >= initialCapital ? "#22c55e" : "#ef4444";
  const gradId = `eqGrad_${containerId}`;
  const baseY = yAt(initialCapital);

  let gridSvg = "";
  for (let i = 0; i <= 4; i++) {
    const val = minVal + (range * i) / 4;
    const y = yAt(val);
    gridSvg += `
      <line x1="${pad.left}" y1="${y.toFixed(1)}" x2="${width - pad.right}" y2="${y.toFixed(1)}" stroke="#1e2d4a" stroke-width="1" />
      <text x="${pad.left - 8}" y="${(y + 4).toFixed(1)}" fill="#8ea0be" font-size="11" text-anchor="end">¥${Math.round(val).toLocaleString("ja-JP")}</text>
    `;
  }

  const labelIndices = Array.from(new Set([0, Math.floor(points.length / 2), points.length - 1]));
  const xLabels = labelIndices
    .map((idx) => {
      const anchor = idx === 0 ? "start" : idx === points.length - 1 ? "end" : "middle";
      return `<text x="${xAt(idx).toFixed(1)}" y="${height - 8}" fill="#8ea0be" font-size="11" text-anchor="${anchor}">${escapeHtml(fmtTime(points[idx].timestamp))}</text>`;
    })
    .join("");

  box.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="資産推移チャート">
      <defs>
        <linearGradient id="${gradId}" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="${strokeColor}" stop-opacity="0.32" />
          <stop offset="100%" stop-color="${strokeColor}" stop-opacity="0.01" />
        </linearGradient>
      </defs>
      ${gridSvg}
      <line x1="${pad.left}" y1="${baseY.toFixed(1)}" x2="${width - pad.right}" y2="${baseY.toFixed(1)}"
            stroke="#60a5fa" stroke-width="1.2" stroke-dasharray="5,4" />
      <text x="${width - pad.right}" y="${(baseY - 5).toFixed(1)}" fill="#93c5fd" font-size="10" text-anchor="end">元手 ${fmtJPY(initialCapital)}</text>
      <path d="${areaPath}" fill="url(#${gradId})" />
      <polyline points="${lineCoords.join(" ")}" fill="none" stroke="${strokeColor}" stroke-width="2.4" stroke-linejoin="round" />
      <circle cx="${xAt(points.length - 1).toFixed(1)}" cy="${yAt(lastVal).toFixed(1)}" r="4.5" fill="${strokeColor}" stroke="#fff" stroke-width="1.5" />
      ${xLabels}
    </svg>
  `;
}

function renderMultiCompareSVG(containerId, results) {
  const box = document.getElementById(containerId);
  if (!results || results.length === 0) {
    box.innerHTML = `<div class="empty-state">比較データがありません。</div>`;
    return;
  }

  const width = 920;
  const height = 250;
  const pad = { top: 18, right: 20, bottom: 30, left: 58 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  let minRet = 0;
  let maxRet = 0;
  let minTs = Infinity;
  let maxTs = -Infinity;

  results.forEach((r) => {
    (r.equity_curve || []).forEach((pt) => {
      const ret = Number(pt.return_pct || 0);
      const t = new Date(pt.timestamp).getTime();
      if (ret < minRet) minRet = ret;
      if (ret > maxRet) maxRet = ret;
      if (t < minTs) minTs = t;
      if (t > maxTs) maxTs = t;
    });
  });

  minRet = Math.min(minRet - 0.5, -0.5);
  maxRet = Math.max(maxRet + 0.5, 0.5);
  const retRange = Math.max(maxRet - minRet, 1.0);
  const tsRange = Math.max(maxTs - minTs, 1);

  const xAt = (ts) => pad.left + ((ts - minTs) / tsRange) * plotW;
  const yAt = (ret) => pad.top + plotH - ((ret - minRet) / retRange) * plotH;

  let gridSvg = "";
  for (let i = 0; i <= 4; i++) {
    const v = minRet + (retRange * i) / 4;
    const y = yAt(v);
    gridSvg += `
      <line x1="${pad.left}" y1="${y.toFixed(1)}" x2="${width - pad.right}" y2="${y.toFixed(1)}" stroke="#1e2d4a" stroke-width="1" />
      <text x="${pad.left - 8}" y="${(y + 4).toFixed(1)}" fill="#8ea0be" font-size="11" text-anchor="end">${v >= 0 ? "+" : ""}${v.toFixed(1)}%</text>
    `;
  }

  const zeroY = yAt(0);
  const legendItems = [];
  let polylinesSvg = "";

  results.forEach((r, idx) => {
    const color = SERIES_COLORS[idx % SERIES_COLORS.length];
    const pts = (r.equity_curve || [])
      .map((pt) => `${xAt(new Date(pt.timestamp).getTime()).toFixed(1)},${yAt(Number(pt.return_pct || 0)).toFixed(1)}`)
      .join(" ");
    if (pts) {
      polylinesSvg += `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2.2" stroke-linejoin="round" />`;
    }
    legendItems.push(
      `<span class="legend-item"><span class="legend-swatch" style="background:${color}"></span>${escapeHtml(r.pattern_name)} (${fmtPct(r.return_pct)})</span>`
    );
  });

  const startLabel = fmtTime(new Date(minTs).toISOString());
  const endLabel = fmtTime(new Date(maxTs).toISOString());

  box.innerHTML = `
    <div class="chart-legend">${legendItems.join("")}</div>
    <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="全パターン利回り比較チャート">
      ${gridSvg}
      <line x1="${pad.left}" y1="${zeroY.toFixed(1)}" x2="${width - pad.right}" y2="${zeroY.toFixed(1)}" stroke="#60a5fa" stroke-width="1.2" stroke-dasharray="5,4" />
      ${polylinesSvg}
      <text x="${pad.left}" y="${height - 8}" fill="#8ea0be" font-size="11" text-anchor="start">${escapeHtml(startLabel)}</text>
      <text x="${width - pad.right}" y="${height - 8}" fill="#8ea0be" font-size="11" text-anchor="end">${escapeHtml(endLabel)}</text>
    </svg>
  `;
}

function renderPatternsTable() {
  const tbody = document.getElementById("patternsTableBody");
  const list = getPatternList();
  const selected = getSelectedPatternData();
  const selId = selected ? selected.pattern_id || selected.mode : null;

  if (list.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" class="empty-cell">パターンが登録されていません。</td></tr>`;
    return;
  }

  // Update backtest target dropdown
  const btSelect = document.getElementById("btTarget");
  const prevVal = btSelect.value;
  btSelect.innerHTML =
    `<option value="__ALL__">🔥 全有効パターンを並行比較 (${list.filter((p) => p.enabled !== false).length}件)</option>` +
    list
      .map(
        (p) =>
          `<option value="${escapeHtml(p.pattern_id || p.mode)}">${escapeHtml(p.mode_label)} [${escapeHtml(
            p.interval || "15m"
          )} / ${fmtJPY(p.initial_capital)}]</option>`
      )
      .join("");
  if (prevVal && Array.from(btSelect.options).some((o) => o.value === prevVal)) {
    btSelect.value = prevVal;
  }

  tbody.innerHTML = list
    .map((p) => {
      const pid = p.pattern_id || p.mode;
      const isSelected = pid === selId;
      const isEnabled = p.enabled !== false;
      const pnlCls = p.net_pnl > 0 ? "up" : p.net_pnl < 0 ? "down" : "";
      const pat = p.pattern || {};
      const ruleHint = `買≥${pat.buy_score_threshold ?? 66} / 利確+${((pat.take_profit_pct ?? 0.035) * 100).toFixed(1)}% / 損切-${((pat.stop_loss_pct ?? 0.02) * 100).toFixed(1)}%${pat.close_before_eod ? " / 引け決済" : ""}`;
      const disableAttr = canEditPatterns() ? "" : "disabled";

      return `
        <tr class="pattern-row ${isSelected ? "active-row" : ""} ${isEnabled ? "" : "disabled-row"}" data-pid="${escapeHtml(pid)}">
          <td>
            <span class="badge ${isEnabled ? "badge-buy" : "badge-hold"}">${isEnabled ? "稼働中" : "停止"}</span>
          </td>
          <td>
            <div class="stock-name">${escapeHtml(p.mode_label)}</div>
            <div class="stock-meta">${escapeHtml(ruleHint)}</div>
          </td>
          <td><b>${fmtJPY(p.initial_capital)}</b></td>
          <td>
            <span class="badge badge-hold">${escapeHtml(p.interval || "15m")}足</span>
            <span class="stock-meta">${p.unit_size}株</span>
          </td>
          <td><b>${fmtJPY(p.total_equity)}</b></td>
          <td class="${pnlCls}">
            <b>${fmtSignedJPY(p.net_pnl)}</b>
            <div class="stock-meta ${pnlCls}">${fmtPct(p.return_pct)}</div>
          </td>
          <td>
            <div>${Number(p.win_rate || 0).toFixed(1)}% / PF ${Number(p.profit_factor || 0).toFixed(2)}</div>
            <div class="stock-meta">最大DD -${Number(p.max_drawdown_pct || 0).toFixed(2)}% (${p.trade_count}取引)</div>
          </td>
          <td>${(p.positions || []).length}銘柄</td>
          <td>
            <div class="row-actions" onclick="event.stopPropagation()">
              <button class="btn btn-ghost btn-xs act-edit" data-pid="${escapeHtml(pid)}" ${disableAttr} title="資金・タイミング編集">編集</button>
              <button class="btn btn-ghost btn-xs act-dup" data-pid="${escapeHtml(pid)}" ${disableAttr} title="このパターンを複製">複製</button>
              <button class="btn btn-ghost btn-xs act-toggle" data-pid="${escapeHtml(pid)}" ${disableAttr} title="自動監視ON/OFF">${isEnabled ? "停止" : "再開"}</button>
              <button class="btn btn-danger-ghost btn-xs act-del" data-pid="${escapeHtml(pid)}" ${disableAttr} ${list.length <= 1 ? "disabled" : ""} title="削除">削除</button>
            </div>
          </td>
        </tr>
      `;
    })
    .join("");

  tbody.querySelectorAll(".pattern-row").forEach((tr) => {
    tr.addEventListener("click", () => {
      state.selectedPatternId = tr.dataset.pid;
      renderDashboard();
    });
  });

  tbody.querySelectorAll(".act-edit").forEach((btn) =>
    btn.addEventListener("click", () => openPatternModal(btn.dataset.pid, false))
  );
  tbody.querySelectorAll(".act-dup").forEach((btn) =>
    btn.addEventListener("click", () => openPatternModal(btn.dataset.pid, true))
  );
  tbody.querySelectorAll(".act-toggle").forEach((btn) =>
    btn.addEventListener("click", () => togglePatternEnabled(btn.dataset.pid))
  );
  tbody.querySelectorAll(".act-del").forEach((btn) =>
    btn.addEventListener("click", () => deletePattern(btn.dataset.pid))
  );
}

function renderDashboard() {
  const dash = state.dashboard;
  if (!dash) return;

  const ms = dash.market_status || {};
  const msBadge = document.getElementById("marketStatusBadge");
  msBadge.textContent = ms.session || "東証ステータス不明";
  msBadge.className = `pill ${ms.is_open ? "pill-ok" : "pill-warn"}`;

  renderMlMonitor(dash.ml_status || null);
  renderPatternsTable();

  const p = getSelectedPatternData();
  if (!p) return;

  document.getElementById("lastUpdatedText").textContent = `最終更新: ${fmtTime(dash.generated_at)}`;
  document.getElementById("selectedPatternTitle").textContent = p.mode_label;
  const pat = p.pattern || {};
  document.getElementById("selectedPatternMeta").textContent =
    `初期資金 ${fmtJPY(p.initial_capital)} ｜ ${p.interval || "15m"}足 ｜ ${p.unit_size}株単位 ｜ 最大${pat.max_positions ?? 3}銘柄 ｜ 買いスコア≥${pat.buy_score_threshold ?? 66} / 売り≤${pat.sell_score_threshold ?? 30} ｜ 利確+${((pat.take_profit_pct ?? 0.035) * 100).toFixed(1)}% / 損切-${((pat.stop_loss_pct ?? 0.02) * 100).toFixed(1)}% ｜ 最新足: ${fmtTime(p.last_bar_ts)}`;

  document.getElementById("kpiCapitalLabel").textContent = `税引後・手取り純資産（元手 ${fmtJPY(p.initial_capital)}）`;

  const eqEl = document.getElementById("kpiTotalEquity");
  eqEl.textContent = fmtJPY(p.total_equity);
  eqEl.className = `kpi-value ${p.net_pnl > 0 ? "up" : p.net_pnl < 0 ? "down" : ""}`;

  const pnlEl = document.getElementById("kpiNetPnl");
  pnlEl.textContent = `手取り損益 ${fmtSignedJPY(p.net_pnl)} (${fmtPct(p.return_pct)})`;
  pnlEl.className = `kpi-sub ${p.net_pnl > 0 ? "up" : p.net_pnl < 0 ? "down" : ""}`;

  const realEl = document.getElementById("kpiRealizedPnl");
  realEl.textContent = fmtSignedJPY(p.realized_pnl);
  realEl.className = `kpi-value ${p.realized_pnl > 0 ? "up" : p.realized_pnl < 0 ? "down" : ""}`;
  document.getElementById("kpiUnrealizedPnl").textContent =
    `含み損益: ${fmtSignedJPY(p.unrealized_pnl)} ｜ 手数料 ${fmtJPY(p.total_fees)} ｜ 税控除 ${fmtJPY(p.total_tax || 0)}`;

  document.getElementById("kpiCash").textContent = fmtJPY(p.cash);
  document.getElementById("kpiHoldings").textContent =
    `保有評価額: ${fmtJPY(p.holdings_value)} (${(p.positions || []).length}銘柄)`;

  document.getElementById("kpiWinRate").textContent = `${Number(p.win_rate || 0).toFixed(1)}%`;
  document.getElementById("kpiStats").textContent =
    `決済 ${p.closed_trades}回 / PF ${Number(p.profit_factor || 0).toFixed(2)} / 最大DD -${Number(p.max_drawdown_pct || 0).toFixed(2)}%`;

  renderPendingOrders(p.pending_orders || [], p.interval || "15m");
  renderEquitySVG("liveEquityChart", p.equity_history, p.initial_capital);
  renderSellEvaluations(p.sell_evaluations || [], p.positions || []);
  renderRecommendations(p.recommendations || []);
  renderTrades("tradesBody", p.trades || []);
}

function renderMlLearningCurveSVG(containerId, curve) {
  const box = document.getElementById(containerId);
  if (!box) return;
  if (!curve || curve.length < 2) {
    box.innerHTML = `<div class="empty-state" style="padding:36px 12px;">Walk-Forward 学習経過データはまだ十分に蓄積されていません。</div>`;
    return;
  }

  const width = 560;
  const height = 195;
  const pad = { top: 22, right: 48, bottom: 28, left: 48 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const hits = curve.map((c) => Number(c.hit_rate_pct || 0));
  const ics = curve.map((c) => Number(c.oos_ic || 0));
  const minHit = Math.min(...hits, 40) - 5;
  const maxHit = Math.max(...hits, 65) + 5;
  const hitRange = Math.max(maxHit - minHit, 10);

  const minIc = Math.min(...ics, -0.05) - 0.02;
  const maxIc = Math.max(...ics, 0.1) + 0.02;
  const icRange = Math.max(maxIc - minIc, 0.05);

  const xAt = (i) => pad.left + (i / (curve.length - 1)) * plotW;
  const yHit = (v) => pad.top + plotH - ((v - minHit) / hitRange) * plotH;
  const yIc = (v) => pad.top + plotH - ((v - minIc) / icRange) * plotH;

  const hitLine = curve.map((c, i) => `${xAt(i).toFixed(1)},${yHit(Number(c.hit_rate_pct || 0)).toFixed(1)}`).join(" ");
  const icLine = curve.map((c, i) => `${xAt(i).toFixed(1)},${yIc(Number(c.oos_ic || 0)).toFixed(1)}`).join(" ");

  const midIdx = Math.floor(curve.length / 2);
  const labels = [0, midIdx, curve.length - 1]
    .map((idx) => {
      const anchor = idx === 0 ? "start" : idx === curve.length - 1 ? "end" : "middle";
      return `<text x="${xAt(idx).toFixed(1)}" y="${height - 6}" fill="#8ea0be" font-size="10" text-anchor="${anchor}">F${curve[idx].fold} (${Number(curve[idx].train_samples).toLocaleString("ja-JP")}件)</text>`;
    })
    .join("");

  box.innerHTML = `
    <div class="chart-legend" style="padding:0 4px 4px;">
      <span class="legend-item"><span class="legend-swatch" style="background:#22c55e"></span>高確信的中率 (%) [左軸]</span>
      <span class="legend-item"><span class="legend-swatch" style="background:#3b82f6"></span>OOS相関係数 IC [右軸]</span>
    </div>
    <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" style="height:175px;">
      <line x1="${pad.left}" y1="${yHit(50).toFixed(1)}" x2="${width - pad.right}" y2="${yHit(50).toFixed(1)}" stroke="#1e2d4a" stroke-dasharray="4,4" />
      <text x="${pad.left - 6}" y="${(yHit(maxHit) + 4).toFixed(1)}" fill="#86efac" font-size="10" text-anchor="end">${Math.round(maxHit)}%</text>
      <text x="${pad.left - 6}" y="${(yHit(minHit) + 4).toFixed(1)}" fill="#86efac" font-size="10" text-anchor="end">${Math.round(minHit)}%</text>
      <text x="${width - pad.right + 6}" y="${(yIc(maxIc) + 4).toFixed(1)}" fill="#93c5fd" font-size="10" text-anchor="start">${maxIc.toFixed(2)}</text>
      <text x="${width - pad.right + 6}" y="${(yIc(minIc) + 4).toFixed(1)}" fill="#93c5fd" font-size="10" text-anchor="start">${minIc.toFixed(2)}</text>
      <polyline points="${hitLine}" fill="none" stroke="#22c55e" stroke-width="2.2" />
      <polyline points="${icLine}" fill="none" stroke="#3b82f6" stroke-width="2.0" />
      ${curve.map((c, i) => `<circle cx="${xAt(i).toFixed(1)}" cy="${yHit(Number(c.hit_rate_pct || 0)).toFixed(1)}" r="3" fill="#22c55e" />`).join("")}
      ${labels}
    </svg>
  `;
}

function renderMlMonitor(ml) {
  const cardsEl = document.getElementById("mlStatusCards");
  const weightsEl = document.getElementById("mlFeatureWeights");
  const sourcesBody = document.getElementById("mlSourcesBody");
  const historyBody = document.getElementById("mlHistoryBody");
  const historyCount = document.getElementById("mlHistoryCount");
  if (!cardsEl) return;

  if (!ml || (!ml.total_bars && !ml.macro_total_rows)) {
    cardsEl.innerHTML = `<div class="bt-stat"><div class="bt-stat-label">蓄積学習データ</div><div class="bt-stat-val">未収集</div><div class="stock-meta">スキャンまたは「外部データ収集＆再学習」で自動蓄積を開始します</div></div>`;
    if (weightsEl) weightsEl.innerHTML = "";
    renderMlLearningCurveSVG("mlLearningCurveChart", []);
    return;
  }

  const models = ml.models || {};
  const primaryModel = models["30m"] || models["15m"] || models["60m"] || Object.values(models)[0] || null;
  const ivText = (ml.by_interval || [])
    .map((x) => `${x.interval}: ${Number(x.bars).toLocaleString("ja-JP")}本`)
    .join(" / ");
  const spanText = ml.earliest_bar && ml.latest_bar ? `${fmtTime(ml.earliest_bar)} 〜 ${fmtTime(ml.latest_bar)}` : "--";
  const icVal = primaryModel ? Number(primaryModel.oos_ic || 0) : 0;
  const icCls = icVal > 0 ? "up" : icVal < 0 ? "down" : "";
  const rIc = primaryModel ? Number(primaryModel.ridge_ic || 0) : 0;
  const gIc = primaryModel ? Number(primaryModel.gbdt_ic || 0) : 0;

  cardsEl.innerHTML = `
    <div class="bt-stat">
      <div class="bt-stat-label">蓄積済み日中足バー総数</div>
      <div class="bt-stat-val">${Number(ml.total_bars || 0).toLocaleString("ja-JP")} 本</div>
      <div class="stock-meta">${ml.unique_symbols || 0} 銘柄 (${escapeHtml(ivText || "--")})</div>
    </div>
    <div class="bt-stat">
      <div class="bt-stat-label">外部マクロデータ (Yahoo以外)</div>
      <div class="bt-stat-val">${Number(ml.macro_total_rows || 0).toLocaleString("ja-JP")} 日分</div>
      <div class="stock-meta">FRED米連銀: ドル円・VIX・米10年債・日経 (${(ml.macro_sources || []).length}系列)</div>
    </div>
    <div class="bt-stat">
      <div class="bt-stat-label">Walk-Forward OOS 相関 (IC)</div>
      <div class="bt-stat-val ${icCls}">${icVal >= 0 ? "+" : ""}${icVal.toFixed(4)}</div>
      <div class="stock-meta">Ridge ${rIc >= 0 ? "+" : ""}${rIc.toFixed(3)} / GBDT ${gIc >= 0 ? "+" : ""}${gIc.toFixed(3)}</div>
    </div>
    <div class="bt-stat">
      <div class="bt-stat-label">高確信ML予測 的中率 / RMSE</div>
      <div class="bt-stat-val">${primaryModel ? Number(primaryModel.oos_hit_rate_pct || 0).toFixed(1) : "0.0"}%</div>
      <div class="stock-meta">学習 ${primaryModel ? Number(primaryModel.train_samples || 0).toLocaleString("ja-JP") : 0}件 / 誤差 ${primaryModel ? Number(primaryModel.oos_rmse || 0).toFixed(2) : "0.00"}%</div>
    </div>
    <div class="bt-stat">
      <div class="bt-stat-label">稼働モデル (${primaryModel ? primaryModel.feature_count || 22 : 22}特徴量)</div>
      <div class="bt-stat-val" style="font-size:0.84rem">${escapeHtml(primaryModel ? primaryModel.model_type : "Walk-Forward アンサンブル")}</div>
      <div class="stock-meta">蓄積期間: ${escapeHtml(spanText)}</div>
    </div>
  `;

  if (weightsEl && primaryModel && primaryModel.top_features && primaryModel.top_features.length > 0) {
    weightsEl.innerHTML =
      `<span class="stock-meta" style="margin-right:4px;">学習済み主要特徴量:</span>` +
      primaryModel.top_features
        .map(
          (f) =>
            `<span class="reason-chip">${escapeHtml(f.label)} (${f.weight >= 0 ? "+" : ""}${f.weight} / ${escapeHtml(f.direction)})</span>`
        )
        .join("");
  } else if (weightsEl) {
    weightsEl.innerHTML = "";
  }

  // Render Walk-Forward learning curve SVG
  renderMlLearningCurveSVG("mlLearningCurveChart", primaryModel ? primaryModel.learning_curve || [] : []);

  // Render data sources breakdown table
  if (sourcesBody) {
    const srcRows = [];
    (ml.by_interval || []).forEach((iv) => {
      srcRows.push(`
        <tr>
          <td><b>Yahoo Finance (${escapeHtml(iv.interval)}足)</b> <span class="stock-meta">${iv.symbols}銘柄</span></td>
          <td>${Number(iv.bars).toLocaleString("ja-JP")} 本</td>
          <td class="stock-meta">${escapeHtml(fmtTime(iv.min_ts))} 〜 ${escapeHtml(fmtTime(iv.max_ts))}</td>
        </tr>
      `);
    });
    (ml.macro_sources || []).forEach((m) => {
      srcRows.push(`
        <tr>
          <td><b>${escapeHtml(m.source)}</b> <span class="stock-meta">${escapeHtml(m.label)}</span></td>
          <td>${Number(m.days).toLocaleString("ja-JP")} 日分</td>
          <td class="stock-meta">${escapeHtml(m.min_date)} 〜 ${escapeHtml(m.max_date)}</td>
        </tr>
      `);
    });
    sourcesBody.innerHTML = srcRows.length
      ? srcRows.join("")
      : `<tr><td colspan="3" class="empty-cell">データ未収集</td></tr>`;
  }

  // Render training history log table
  const hist = ml.training_history || [];
  if (historyCount) historyCount.textContent = String(hist.length);
  if (historyBody) {
    if (hist.length === 0) {
      historyBody.innerHTML = `<tr><td colspan="8" class="empty-cell">学習履歴はまだありません。</td></tr>`;
    } else {
      historyBody.innerHTML = hist
        .map((h) => {
          const ic = Number(h.oos_ic || 0);
          const icClsRow = ic > 0 ? "up" : ic < 0 ? "down" : "";
          return `
            <tr>
              <td>${escapeHtml(fmtTime(h.timestamp))}</td>
              <td><span class="badge badge-hold">${escapeHtml(h.interval)}足</span></td>
              <td class="stock-meta">${escapeHtml(h.model_type)}</td>
              <td><b>${Number(h.train_samples).toLocaleString("ja-JP")}</b>件</td>
              <td class="${icClsRow}"><b>${ic >= 0 ? "+" : ""}${ic.toFixed(4)}</b> <span class="stock-meta">(R:${Number(h.ridge_ic || 0).toFixed(3)} / G:${Number(h.gbdt_ic || 0).toFixed(3)})</span></td>
              <td><b>${Number(h.oos_hit_rate_pct || 0).toFixed(1)}%</b></td>
              <td>${Number(h.oos_rmse || 0).toFixed(2)}%</td>
              <td><span class="reason-chip">${escapeHtml(h.top_feature || "--")}</span></td>
            </tr>
          `;
        })
        .join("");
    }
  }
}

function renderPendingOrders(orders, interval) {
  const container = document.getElementById("pendingOrdersContainer");
  if (!container) return;
  if (!orders || orders.length === 0) {
    container.innerHTML = `<div class="empty-state">現在予約されている次回売買提案はありません（条件を満たす銘柄が出ると、次の ${escapeHtml(interval)}足 始値での売買予約としてここに表示されます）。</div>`;
    return;
  }
  container.innerHTML = orders
    .map((o) => {
      const isBuy = o.side === "BUY";
      const badge = isBuy
        ? `<span class="badge badge-strong-buy">次回 買い予約 (BUY)</span>`
        : `<span class="badge badge-sell">次回 売り予約 (SELL)</span>`;
      return `
        <div class="sell-item ${isBuy ? "" : "urgent"}">
          <div class="sell-item-top">
            <div>
              <span class="sell-title">${escapeHtml(o.name)}</span>
              <span class="stock-meta">(${escapeHtml(o.symbol)} / スコア ${Number(o.score || 0).toFixed(1)})</span>
            </div>
            ${badge}
          </div>
          <div class="sell-metrics">
            <span>事前予測時刻: <b>${escapeHtml(fmtTime(o.signal_time))}</b> → 次の ${escapeHtml(interval)}足 始値で約定予定</span>
            <span>提案時株価: <b>${fmtPrice(o.signal_price)}</b></span>
            <span>利確目安 ${fmtPrice(o.target_price)} / 損切り ${fmtPrice(o.stop_loss_price)}</span>
          </div>
          <div class="sell-reason">📌 ${escapeHtml(o.reason)}</div>
        </div>
      `;
    })
    .join("");
}

function renderSellEvaluations(evals, positions) {
  const container = document.getElementById("sellTimingContainer");
  if ((!evals || evals.length === 0) && (!positions || positions.length === 0)) {
    container.innerHTML = `<div class="empty-state">現在保有中の銘柄はありません。<br/>買い条件を満たす銘柄が見つかると自動でエントリーし、ここで売却タイミングを常時推定します。</div>`;
    return;
  }

  const list = evals.length > 0 ? evals : positions.map((p) => ({
    symbol: p.symbol,
    name: p.name,
    quantity: p.quantity,
    entry_price: p.avg_cost,
    current_price: p.last_price,
    unrealized_pnl: (p.last_price - p.avg_cost) * p.quantity,
    unrealized_pnl_pct: p.avg_cost ? ((p.last_price - p.avg_cost) / p.avg_cost) * 100 : 0,
    should_sell: false,
    urgency: 20,
    timing_label: "保有継続",
    primary_reason: "次回スキャン待ち",
    target_price: p.target_price,
    stop_loss_price: p.stop_loss_price,
    trailing_stop_price: null,
  }));

  container.innerHTML = list
    .map((ev) => {
      const pnlClass = ev.unrealized_pnl >= 0 ? "up" : "down";
      const badgeClass = ev.should_sell ? "badge-sell" : ev.urgency >= 55 ? "badge-warn" : "badge-buy";
      const barColor = ev.urgency >= 75 ? "#ef4444" : ev.urgency >= 50 ? "#f59e0b" : "#22c55e";
      const trailText = ev.trailing_stop_price ? ` / トレール ${fmtPrice(ev.trailing_stop_price)}` : "";
      return `
        <div class="sell-item ${ev.should_sell ? "urgent" : ""}">
          <div class="sell-item-top">
            <div>
              <span class="sell-title">${escapeHtml(ev.name)}</span>
              <span class="stock-meta">(${escapeHtml(ev.symbol)} × ${ev.quantity}株)</span>
            </div>
            <span class="badge ${badgeClass}">${escapeHtml(ev.timing_label)} (緊急度 ${Math.round(ev.urgency)})</span>
          </div>
          <div class="sell-metrics">
            <span>取得 ${fmtPrice(ev.entry_price)} → 現在 <b>${fmtPrice(ev.current_price)}</b></span>
            <span class="${pnlClass}"><b>${fmtSignedJPY(ev.unrealized_pnl)} (${fmtPct(ev.unrealized_pnl_pct)})</b></span>
            <span>利確目安 ${fmtPrice(ev.target_price)} / 損切り ${fmtPrice(ev.stop_loss_price)}${trailText}</span>
          </div>
          <div class="sell-reason">💡 ${escapeHtml(ev.primary_reason)}</div>
          <div class="urgency-bar"><div class="urgency-fill" style="width:${Math.min(100, ev.urgency)}%;background:${barColor}"></div></div>
        </div>
      `;
    })
    .join("");
}

function renderRecommendations(recs) {
  const tbody = document.getElementById("recommendationsBody");
  const filterVal = document.getElementById("actionFilter").value;
  const filtered = (recs || []).filter((r) => {
    if (filterVal === "BUY_ONLY") return r.action === "STRONG_BUY" || r.action === "BUY";
    if (filterVal === "SELL_ONLY") return r.action === "SELL";
    return true;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" class="empty-cell">該当する監視銘柄データがありません。「全パターン最新スキャン＆補完」を実行してください。</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered
    .map((r) => {
      const badgeClass =
        r.action === "STRONG_BUY"
          ? "badge-strong-buy"
          : r.action === "BUY"
          ? "badge-buy"
          : r.action === "SELL"
          ? "badge-sell"
          : "badge-hold";
      const chgClass = r.change_pct > 0 ? "up" : r.change_pct < 0 ? "down" : "";
      const barColor = r.score >= 70 ? "#22c55e" : r.score >= 60 ? "#10b981" : r.score <= 35 ? "#ef4444" : "#64748b";
      const chips = (r.reasons || []).map((txt) => `<span class="reason-chip">${escapeHtml(txt)}</span>`).join("");

      return `
        <tr>
          <td>
            <div class="stock-name">${escapeHtml(r.name)}</div>
            <div class="stock-meta">${escapeHtml(r.symbol)} · ${escapeHtml(r.sector)}</div>
          </td>
          <td><span class="badge ${badgeClass}">${escapeHtml(r.action_label)}</span></td>
          <td>
            <div class="score-pill">
              <span>${Number(r.score).toFixed(1)}</span>
              <span class="score-bar"><span class="score-bar-fill" style="width:${r.score}%;background:${barColor}"></span></span>
            </div>
          </td>
          <td><b>${fmtPrice(r.price)}</b><div class="stock-meta">VWAP ${fmtPrice(r.vwap)}</div></td>
          <td class="${chgClass}">${fmtPct(r.change_pct)}</td>
          <td>
            <div class="up">利確 ${fmtPrice(r.target_price)}</div>
            <div class="down">損切 ${fmtPrice(r.stop_loss_price)} (RR ${r.risk_reward})</div>
          </td>
          <td>
            <div>RSI ${Number(r.rsi14).toFixed(1)}</div>
            <div class="stock-meta">出来高 ${Number(r.volume_ratio).toFixed(2)}倍</div>
          </td>
          <td><div class="reason-list">${chips}</div></td>
        </tr>
      `;
    })
    .join("");
}

function renderTrades(tbodyId, trades) {
  const tbody = document.getElementById(tbodyId);
  if (!trades || trades.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" class="empty-cell">売買履歴はまだありません。</td></tr>`;
    return;
  }

  tbody.innerHTML = trades
    .slice(0, 120)
    .map((t) => {
      const isBuy = t.side === "BUY";
      const sideBadge = isBuy
        ? `<span class="badge badge-buy">買い BUY</span>`
        : `<span class="badge badge-sell">売り SELL</span>`;
      const pnlText = isBuy
        ? "--"
        : `<span class="${t.realized_pnl >= 0 ? "up" : "down"}"><b>${fmtSignedJPY(t.realized_pnl)}</b></span>`;
      const sigPart = t.signal_time ? `${fmtTime(t.signal_time)} → ` : "";
      const taxVal = Number(t.tax || 0);
      const taxText = taxVal === 0 ? "税¥0" : `税${taxVal > 0 ? "-" : "+"}¥${Math.abs(Math.round(taxVal)).toLocaleString("ja-JP")}`;
      return `
        <tr>
          <td><span class="stock-meta">${escapeHtml(sigPart)}</span><b>${escapeHtml(fmtTime(t.timestamp))}</b></td>
          <td>${sideBadge}</td>
          <td><b>${escapeHtml(t.name)}</b> <span class="stock-meta">${escapeHtml(t.symbol)}</span></td>
          <td>${Number(t.quantity).toLocaleString("ja-JP")}株</td>
          <td>${fmtPrice(t.price)}</td>
          <td><span class="stock-meta">手数料¥${Math.round(Number(t.fee || 0)).toLocaleString("ja-JP")} / ${taxText}</span></td>
          <td>${pnlText}</td>
          <td>${escapeHtml(t.reason)}</td>
        </tr>
      `;
    })
    .join("");
}

async function loadDashboard() {
  const connBadge = document.getElementById("connectionBadge");
  try {
    const resp = await fetch("/api/dashboard", { cache: "no-store" });
    if (!resp.ok) throw new Error("API unavailable");
    state.dashboard = await resp.json();
    state.apiAvailable = true;
    connBadge.textContent = "🟢 ローカルAPI接続中";
    connBadge.className = "pill pill-ok";
    renderDashboard();
    return;
  } catch {
    state.apiAvailable = false;
  }

  // 1. Check encrypted static snapshot first (GitHub Pages with DASHBOARD_PASSPHRASE)
  try {
    const encResp = await fetch("./data/dashboard_snapshot.enc.json", { cache: "no-store" });
    if (encResp.ok) {
      const envelope = await encResp.json();
      if (envelope && envelope.encrypted) {
        state.encryptedMode = true;
        state.encryptedEnvelope = envelope;
        const savedPass = sessionStorage.getItem(SESSION_KEY);
        if (savedPass) {
          try {
            await tryUnlockWithPassphrase(savedPass);
            return;
          } catch {
            sessionStorage.removeItem(SESSION_KEY);
          }
        }
        connBadge.textContent = "🔒 暗号化ロック中";
        connBadge.className = "pill pill-warn";
        document.getElementById("loginOverlay").classList.remove("hidden");
        document.getElementById("loginPassphrase").focus();
        return;
      }
    }
  } catch {
    // No encrypted snapshot; fall through to plaintext static snapshot
  }

  // 2. Plaintext static snapshot fallback
  try {
    const resp = await fetch("./data/dashboard_snapshot.json", { cache: "no-store" });
    if (!resp.ok) throw new Error("Snapshot missing");
    state.dashboard = await resp.json();
    connBadge.textContent = "☁ 静的ビューアモード（GitHub Actions更新）";
    connBadge.className = "pill pill-warn";
    setStaticButtonsDisabled();
    renderDashboard();
  } catch {
    connBadge.textContent = "🔴 データ未取得";
    connBadge.className = "pill pill-warn";
    showToast("サーバーを起動するか、スナップショットを生成してください。", true);
  }
}

async function handleCatchupScan() {
  if (!canEditPatterns()) return;
  const btn = document.getElementById("btnCatchup");
  btn.disabled = true;
  if (!state.apiAvailable && state.githubConnected) {
    btn.textContent = "⏳ GitHub Actions 起動中...";
    try {
      await ghDispatchMonitorWorkflow({ force_run: true });
      showToast("GitHub Actions の全パターン最新スキャンを起動しました！約1〜2分後に画面を再読み込みすると反映されます。");
    } catch (err) {
      showToast(`GitHub Actions起動エラー: ${err.message}`, true);
    } finally {
      btn.disabled = false;
      btn.textContent = "⚡ 全パターン最新スキャン＆補完";
    }
    return;
  }

  btn.textContent = "⏳ 全パターン取得・補完中...";
  try {
    const resp = await fetch("/api/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "all", catchup: true, days: 5 }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "スキャンに失敗しました");
    state.dashboard = data.dashboard;
    renderDashboard();
    const summary = (data.results || [])
      .map((r) => `${r.name || r.pattern_id}: ${r.bars_processed}本 / ${r.trades_executed || 0}約定`)
      .join(" ｜ ");
    showToast(`全パターン更新完了 — ${summary}`);
  } catch (err) {
    showToast(`エラー: ${err.message}`, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "⚡ 全パターン最新スキャン＆補完";
  }
}

async function handleBacktest(e) {
  e.preventDefault();
  if (!state.apiAvailable) {
    showToast("バックテスト実行にはローカルサーバーの起動が必要です。", true);
    return;
  }
  const btn = document.getElementById("btnRunBacktest");
  btn.disabled = true;
  btn.textContent = "⏳ 実データ取得・並行測定中...";

  const targetVal = document.getElementById("btTarget").value;
  const intervalVal = document.getElementById("btInterval").value;
  const capRaw = document.getElementById("btCapital").value.trim();

  const payload = {
    compare_all: targetVal === "__ALL__",
    pattern_id: targetVal === "__ALL__" ? null : targetVal,
    days: Number(document.getElementById("btDays").value),
    interval: intervalVal || null,
    initial_capital: capRaw ? Number(capRaw) : null,
  };

  try {
    const resp = await fetch("/api/backtest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const res = await resp.json();
    if (!resp.ok) throw new Error(res.detail || "バックテストに失敗しました");

    document.getElementById("backtestResults").classList.remove("hidden");

    if (res.compare_mode) {
      document.getElementById("btComparePanel").classList.remove("hidden");
      document.getElementById("btSinglePanel").classList.add("hidden");
      state.lastCompareResults = res.results || [];
      renderCompareBacktest(state.lastCompareResults);
      showToast(`全${res.pattern_count}パターンの並行測定完了 — トップ: ${res.best_pattern} (${fmtPct(res.best_return_pct)})`);
    } else {
      document.getElementById("btComparePanel").classList.add("hidden");
      document.getElementById("btSinglePanel").classList.remove("hidden");
      renderSingleBacktest(res);
      showToast(`実データ測定完了: ${fmtJPY(res.initial_capital)} → ${fmtJPY(res.final_equity)} (${fmtPct(res.return_pct)})`);
    }
  } catch (err) {
    showToast(`バックテストエラー: ${err.message}`, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "▶ 実データで並行測定を実行";
  }
}

function renderCompareBacktest(results) {
  const tbody = document.getElementById("btCompareBody");
  tbody.innerHTML = results
    .map((r, i) => {
      const cls = r.net_pnl >= 0 ? "up" : "down";
      const bm = r.benchmark_nikkei225_pct != null ? fmtPct(r.benchmark_nikkei225_pct) : "--";
      const color = SERIES_COLORS[i % SERIES_COLORS.length];
      return `
        <tr class="pattern-row" data-idx="${i}">
          <td><b>#${i + 1}</b></td>
          <td><span class="legend-swatch" style="background:${color};margin-right:6px;"></span><b>${escapeHtml(r.pattern_name)}</b></td>
          <td>${escapeHtml(r.interval)}足 / ${r.unit_size}株</td>
          <td>${fmtJPY(r.initial_capital)} → <b>${fmtJPY(r.final_equity)}</b></td>
          <td class="${cls}"><b>${fmtSignedJPY(r.net_pnl)} (${fmtPct(r.return_pct)})</b></td>
          <td>${bm}</td>
          <td>${r.total_trades}回 / 勝率 ${Number(r.win_rate).toFixed(1)}% / PF ${r.profit_factor}</td>
          <td class="down">-${Number(r.max_drawdown_pct).toFixed(2)}%</td>
        </tr>
      `;
    })
    .join("");

  tbody.querySelectorAll("tr").forEach((tr) => {
    tr.addEventListener("click", () => {
      const item = results[Number(tr.dataset.idx)];
      if (item) {
        document.getElementById("btTradeCount").textContent = String((item.trades || []).length);
        renderTrades("btTradesBody", item.trades || []);
      }
    });
  });

  renderMultiCompareSVG("btCompareChart", results);
  if (results[0]) {
    document.getElementById("btTradeCount").textContent = String((results[0].trades || []).length);
    renderTrades("btTradesBody", results[0].trades || []);
  }
}

function renderSingleBacktest(res) {
  const pnlCls = res.net_pnl >= 0 ? "up" : "down";
  const bmText = res.benchmark_nikkei225_pct != null ? fmtPct(res.benchmark_nikkei225_pct) : "取得不可";

  document.getElementById("btSummaryCards").innerHTML = `
    <div class="bt-stat">
      <div class="bt-stat-label">税引後最終資産 (元手 ${fmtJPY(res.initial_capital)})</div>
      <div class="bt-stat-val ${pnlCls}">${fmtJPY(res.final_equity)}</div>
      <div class="stock-meta ${pnlCls}">${fmtSignedJPY(res.net_pnl)} (${fmtPct(res.return_pct)})</div>
    </div>
    <div class="bt-stat">
      <div class="bt-stat-label">勝率 / プロフィットファクター</div>
      <div class="bt-stat-val">${Number(res.win_rate).toFixed(1)}% / PF ${res.profit_factor}</div>
      <div class="stock-meta">総取引 ${res.total_trades}回 (決済 ${res.closed_trades}回)</div>
    </div>
    <div class="bt-stat">
      <div class="bt-stat-label">最大ドローダウン / コスト</div>
      <div class="bt-stat-val down">-${Number(res.max_drawdown_pct).toFixed(2)}%</div>
      <div class="stock-meta">手数料 ${fmtJPY(res.total_fees)} / 税控除 ${fmtJPY(res.total_tax || 0)}</div>
    </div>
    <div class="bt-stat">
      <div class="bt-stat-label">同期間 日経平均 (^N225)</div>
      <div class="bt-stat-val">${bmText}</div>
      <div class="stock-meta">対象 ${res.symbols_loaded}銘柄 / ${res.bars_simulated}本</div>
    </div>
    <div class="bt-stat">
      <div class="bt-stat-label">測定期間 (${escapeHtml(res.interval)}足)</div>
      <div class="bt-stat-val" style="font-size:0.88rem">${fmtTime(res.start_time)} 〜 ${fmtTime(res.end_time)}</div>
      <div class="stock-meta">${escapeHtml(res.pattern_name)}</div>
    </div>
  `;

  renderEquitySVG("btEquityChart", res.equity_curve, res.initial_capital);
  document.getElementById("btTradeCount").textContent = String((res.trades || []).length);
  renderTrades("btTradesBody", res.trades || []);
}

function openPatternModal(patternId = null, duplicate = false) {
  if (!canEditPatterns()) return;
  const form = document.getElementById("patternForm");
  const titleEl = document.getElementById("patternModalTitle");
  const resetWrap = document.getElementById("resetOnSaveWrap");

  const list = getPatternList();
  const existing = patternId ? list.find((p) => (p.pattern_id || p.mode) === patternId) : null;
  const pat = existing && existing.pattern ? existing.pattern : null;

  if (pat && !duplicate) {
    titleEl.textContent = `⚙ パターン編集: ${pat.name}`;
    form.pattern_id.value = pat.id;
    resetWrap.classList.remove("hidden");
  } else {
    titleEl.textContent = duplicate ? "📋 パターンを複製して新規作成" : "＋ 新規シミュレーションパターン作成";
    form.pattern_id.value = "";
    resetWrap.classList.add("hidden");
  }

  form.name.value = pat ? (duplicate ? `${pat.name} (コピー)` : pat.name) : "🤖 MLカスタム 30万円（30分足）";
  form.description.value = pat ? pat.description || "" : "";
  form.strategy_type.value = pat ? pat.strategy_type || "ml_walkforward" : "ml_walkforward";
  form.initial_capital.value = pat ? pat.initial_capital : 300000;
  form.interval.value = pat ? pat.interval : "30m";
  form.ml_min_pred_return.value = ((pat ? pat.ml_min_pred_return ?? 0.008 : 0.008) * 100).toFixed(1);
  form.max_hold_bars.value = pat ? (pat.max_hold_bars ?? 12) : 12;
  form.cooldown_bars.value = pat ? (pat.cooldown_bars ?? 6) : 6;
  form.unit_size.value = String(pat ? pat.unit_size : 1);
  form.watchlist_key.value = pat ? pat.watchlist_key : "lot1";
  form.max_positions.value = pat ? pat.max_positions : 3;
  form.max_allocation_pct.value = Math.round((pat ? pat.max_allocation_pct : 0.4) * 100);
  form.buy_score_threshold.value = pat ? pat.buy_score_threshold : 66;
  form.strong_buy_threshold.value = pat ? pat.strong_buy_threshold : 76;
  form.sell_score_threshold.value = pat ? pat.sell_score_threshold : 30;
  form.take_profit_pct.value = ((pat ? pat.take_profit_pct : 0.035) * 100).toFixed(1);
  form.stop_loss_pct.value = ((pat ? pat.stop_loss_pct : 0.02) * 100).toFixed(1);
  form.trailing_activation_pct.value = ((pat ? pat.trailing_activation_pct : 0.015) * 100).toFixed(1);
  form.trailing_stop_pct.value = ((pat ? pat.trailing_stop_pct : 0.01) * 100).toFixed(1);
  form.fee_rate.value = ((pat ? pat.fee_rate : 0.001) * 100).toFixed(2);
  form.tax_rate.value = ((pat ? pat.tax_rate : 0.20315) * 100).toFixed(3);
  form.slippage_bps.value = pat ? pat.slippage_bps : 5;
  form.close_before_eod.checked = pat ? Boolean(pat.close_before_eod) : false;
  form.enabled.checked = pat ? Boolean(pat.enabled) : true;
  form.reset_portfolio.checked = false;

  document.getElementById("patternModal").classList.remove("hidden");
}

async function savePatternForm(e) {
  e.preventDefault();
  const form = e.target;
  const pid = form.pattern_id.value.trim();

  const payload = {
    name: form.name.value.trim(),
    description: form.description.value.trim(),
    strategy_type: form.strategy_type.value,
    initial_capital: Number(form.initial_capital.value),
    interval: form.interval.value,
    ml_min_pred_return: Number(form.ml_min_pred_return.value) / 100,
    max_hold_bars: Number(form.max_hold_bars.value),
    cooldown_bars: Number(form.cooldown_bars.value),
    unit_size: Number(form.unit_size.value),
    watchlist_key: form.watchlist_key.value,
    max_positions: Number(form.max_positions.value),
    max_allocation_pct: Number(form.max_allocation_pct.value) / 100,
    buy_score_threshold: Number(form.buy_score_threshold.value),
    strong_buy_threshold: Number(form.strong_buy_threshold.value),
    sell_score_threshold: Number(form.sell_score_threshold.value),
    take_profit_pct: Number(form.take_profit_pct.value) / 100,
    stop_loss_pct: Number(form.stop_loss_pct.value) / 100,
    trailing_activation_pct: Number(form.trailing_activation_pct.value) / 100,
    trailing_stop_pct: Number(form.trailing_stop_pct.value) / 100,
    fee_rate: Number(form.fee_rate.value) / 100,
    tax_rate: Number(form.tax_rate.value) / 100,
    slippage_bps: Number(form.slippage_bps.value),
    close_before_eod: form.close_before_eod.checked,
    enabled: form.enabled.checked,
  };

  if (pid) {
    payload.reset_portfolio = form.reset_portfolio.checked;
  }

  // GitHub Pages direct-edit path (via GitHub Contents + Actions API)
  if (!state.apiAvailable && state.githubConnected) {
    try {
      const { cfg, sha } = await ghLoadRepoConfig();
      let savedPattern = null;
      if (pid) {
        const idx = cfg.patterns.findIndex((p) => p.id === pid);
        if (idx < 0) throw new Error("対象パターンが設定ファイルに見つかりません");
        const { reset_portfolio, ...cleanUpdates } = payload;
        cfg.patterns[idx] = { ...cfg.patterns[idx], ...cleanUpdates, id: pid };
        savedPattern = cfg.patterns[idx];
      } else {
        const newId = slugifyPatternIdJS(payload.name, cfg.patterns.map((p) => p.id));
        savedPattern = { id: newId, ...payload };
        cfg.patterns.push(savedPattern);
      }

      await ghSaveRepoConfig(
        cfg,
        sha,
        pid ? `chore(config): update pattern ${savedPattern.id}` : `chore(config): add pattern ${savedPattern.id}`
      );
      await ghDispatchMonitorWorkflow({
        force_run: true,
        reset_pattern: pid && payload.reset_portfolio ? pid : "",
      });

      // Optimistically reflect in UI immediately
      if (state.dashboard && state.dashboard.patterns) {
        const prev = state.dashboard.patterns[savedPattern.id];
        state.dashboard.patterns[savedPattern.id] = {
          ...(prev || {
            total_equity: savedPattern.initial_capital,
            cash: savedPattern.initial_capital,
            holdings_value: 0,
            net_pnl: 0,
            return_pct: 0,
            realized_pnl: 0,
            unrealized_pnl: 0,
            total_fees: 0,
            total_tax: 0,
            trade_count: 0,
            closed_trades: 0,
            win_rate: 0,
            profit_factor: 0,
            max_drawdown_pct: 0,
            positions: [],
            pending_orders: [],
            sell_evaluations: [],
            recommendations: [],
            trades: [],
            equity_history: [],
          }),
          pattern_id: savedPattern.id,
          mode: savedPattern.id,
          mode_label: savedPattern.name,
          initial_capital: savedPattern.initial_capital,
          unit_size: savedPattern.unit_size,
          interval: savedPattern.interval,
          enabled: savedPattern.enabled,
          pattern: savedPattern,
        };
        if (!state.dashboard.pattern_order.includes(savedPattern.id)) {
          state.dashboard.pattern_order.push(savedPattern.id);
        }
        state.selectedPatternId = savedPattern.id;
      }

      document.getElementById("patternModal").classList.add("hidden");
      renderDashboard();
      showToast(`GitHubへ「${savedPattern.name}」を保存し、自動シミュレーションを起動しました（約1〜2分後に最新結果が反映されます）。`);
    } catch (err) {
      showToast(`GitHub保存エラー: ${err.message}`, true);
    }
    return;
  }

  try {
    const url = pid ? `/api/patterns/${encodeURIComponent(pid)}` : "/api/patterns";
    const method = pid ? "PUT" : "POST";
    const resp = await fetch(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "保存に失敗しました");
    state.dashboard = data.dashboard;
    state.selectedPatternId = data.pattern.id;
    document.getElementById("patternModal").classList.add("hidden");
    renderDashboard();
    showToast(pid ? `パターン「${data.pattern.name}」を更新しました。` : `新規パターン「${data.pattern.name}」を追加しました。`);
  } catch (err) {
    showToast(`パターン保存エラー: ${err.message}`, true);
  }
}

async function togglePatternEnabled(pid) {
  const list = getPatternList();
  const target = list.find((p) => (p.pattern_id || p.mode) === pid);
  if (!target) return;
  if (!state.apiAvailable && state.githubConnected) {
    try {
      const { cfg, sha } = await ghLoadRepoConfig();
      const item = cfg.patterns.find((p) => p.id === pid);
      if (!item) throw new Error("対象パターンが見つかりません");
      item.enabled = !target.enabled;
      await ghSaveRepoConfig(cfg, sha, `chore(config): toggle pattern ${pid}`);
      await ghDispatchMonitorWorkflow({ force_run: true });
      if (state.dashboard && state.dashboard.patterns && state.dashboard.patterns[pid]) {
        state.dashboard.patterns[pid].enabled = item.enabled;
        if (state.dashboard.patterns[pid].pattern) state.dashboard.patterns[pid].pattern.enabled = item.enabled;
      }
      renderDashboard();
      showToast(`GitHub設定を更新：「${target.mode_label}」を${item.enabled ? "稼働中" : "停止"}に変更しました。`);
    } catch (err) {
      showToast(`GitHub更新エラー: ${err.message}`, true);
    }
    return;
  }
  try {
    const resp = await fetch(`/api/patterns/${encodeURIComponent(pid)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: !target.enabled }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "切替失敗");
    state.dashboard = data.dashboard;
    renderDashboard();
    showToast(`「${target.mode_label}」を${!target.enabled ? "稼働中" : "停止"}に変更しました。`);
  } catch (err) {
    showToast(`エラー: ${err.message}`, true);
  }
}

async function deletePattern(pid) {
  const list = getPatternList();
  const target = list.find((p) => (p.pattern_id || p.mode) === pid);
  if (!target) return;
  if (!confirm(`パターン「${target.mode_label}」とそのシミュレーション履歴を削除しますか？`)) return;
  if (!state.apiAvailable && state.githubConnected) {
    try {
      const { cfg, sha } = await ghLoadRepoConfig();
      if (cfg.patterns.length <= 1) throw new Error("最低1つのパターンを残す必要があります");
      cfg.patterns = cfg.patterns.filter((p) => p.id !== pid);
      await ghSaveRepoConfig(cfg, sha, `chore(config): delete pattern ${pid}`);
      await ghDispatchMonitorWorkflow({ force_run: true });
      if (state.dashboard && state.dashboard.patterns) {
        delete state.dashboard.patterns[pid];
        state.dashboard.pattern_order = (state.dashboard.pattern_order || []).filter((id) => id !== pid);
      }
      if (state.selectedPatternId === pid) state.selectedPatternId = null;
      renderDashboard();
      showToast(`GitHubから「${target.mode_label}」を削除し、スナップショット更新を起動しました。`);
    } catch (err) {
      showToast(`GitHub削除エラー: ${err.message}`, true);
    }
    return;
  }
  try {
    const resp = await fetch(`/api/patterns/${encodeURIComponent(pid)}`, { method: "DELETE" });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "削除失敗");
    state.dashboard = data.dashboard;
    if (state.selectedPatternId === pid) state.selectedPatternId = null;
    renderDashboard();
    showToast(`パターン「${target.mode_label}」を削除しました。`);
  } catch (err) {
    showToast(`エラー: ${err.message}`, true);
  }
}

async function handleResetSelected() {
  const p = getSelectedPatternData();
  if (!p || !canEditPatterns()) return;
  const pid = p.pattern_id || p.mode;
  const capStr = prompt(
    `「${p.mode_label}」のシミュレーションをリセットします。\n新しい初期資金（円）を入力してください:`,
    String(Math.round(p.initial_capital))
  );
  if (capStr === null) return;
  const newCap = Number(capStr);
  if (!Number.isFinite(newCap) || newCap < 1000) {
    showToast("1,000円以上の数値を入力してください。", true);
    return;
  }
  if (!state.apiAvailable && state.githubConnected) {
    try {
      const { cfg, sha } = await ghLoadRepoConfig();
      const item = cfg.patterns.find((x) => x.id === pid);
      if (item) {
        item.initial_capital = newCap;
        await ghSaveRepoConfig(cfg, sha, `chore(config): reset capital of ${pid} to ${newCap}`);
      }
      await ghDispatchMonitorWorkflow({ force_run: true, reset_pattern: pid });
      showToast(`「${p.mode_label}」を元手 ${fmtJPY(newCap)} でリセットするよう GitHub Actions を起動しました（約1〜2分後に反映）。`);
    } catch (err) {
      showToast(`GitHubリセットエラー: ${err.message}`, true);
    }
    return;
  }
  try {
    const resp = await fetch(`/api/patterns/${encodeURIComponent(pid)}/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ initial_capital: newCap }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "リセット失敗");
    state.dashboard = data.dashboard;
    renderDashboard();
    showToast(`「${p.mode_label}」を元手 ${fmtJPY(newCap)} でリセットしました。`);
  } catch (err) {
    showToast(`エラー: ${err.message}`, true);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("actionFilter").addEventListener("change", renderDashboard);
  document.getElementById("btnCatchup").addEventListener("click", handleCatchupScan);
  document.getElementById("btnAddPattern").addEventListener("click", () => openPatternModal(null, false));
  document.getElementById("btnAddPatternTop").addEventListener("click", () => openPatternModal(null, false));
  document.getElementById("btnEditSelected").addEventListener("click", () => {
    const p = getSelectedPatternData();
    if (p) openPatternModal(p.pattern_id || p.mode, false);
  });
  document.getElementById("btnResetSelected").addEventListener("click", handleResetSelected);
  document.getElementById("backtestForm").addEventListener("submit", handleBacktest);
  document.getElementById("patternForm").addEventListener("submit", savePatternForm);
  document.getElementById("btnClosePatternModal").addEventListener("click", () =>
    document.getElementById("patternModal").classList.add("hidden")
  );
  document.getElementById("btnCancelPatternModal").addEventListener("click", () =>
    document.getElementById("patternModal").classList.add("hidden")
  );
  document.querySelectorAll(".cap-preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.getElementById("patternForm").initial_capital.value = btn.dataset.val;
    });
  });

  document.getElementById("btnCollectMl").addEventListener("click", async () => {
    if (!canEditPatterns()) return;
    const btn = document.getElementById("btnCollectMl");
    btn.disabled = true;
    if (!state.apiAvailable && state.githubConnected) {
      btn.textContent = "⏳ GitHub Actions 起動中...";
      try {
        await ghDispatchMonitorWorkflow({ force_run: true, collect_ml: true });
        showToast("GitHub Actions で外部マクロ＋長期60mデータの再収集＆ML再学習を起動しました（約2分後に反映）。");
      } catch (err) {
        showToast(`ML起動エラー: ${err.message}`, true);
      } finally {
        btn.disabled = false;
        btn.textContent = "📥 外部データ収集＆再学習";
      }
      return;
    }
    btn.textContent = "⏳ 学習データ収集＆再学習中...";
    try {
      const resp = await fetch("/api/ml/collect", { method: "POST" });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "MLデータ収集に失敗しました");
      state.dashboard = data.dashboard;
      renderDashboard();
      const total = data.ml && data.ml.ml_status ? data.ml.ml_status.total_bars : 0;
      showToast(`機械学習データ収集＆再学習完了（蓄積総数: ${Number(total).toLocaleString("ja-JP")}本）`);
    } catch (err) {
      showToast(`ML収集エラー: ${err.message}`, true);
    } finally {
      btn.disabled = false;
      btn.textContent = "📥 外部データ収集＆再学習";
    }
  });

  // GitHub Pages Direct Edit Connection Modal wiring
  const openGithubModal = () => {
    const form = document.getElementById("githubForm");
    const defs = detectDefaultGithubRepo();
    form.gh_owner.value = state.githubConfig.owner || defs.owner;
    form.gh_repo.value = state.githubConfig.repo || defs.repo;
    form.gh_token.value = state.githubConfig.token || "";
    form.gh_remember.checked = Boolean(localStorage.getItem(GH_LOCAL_KEY));
    document.getElementById("ghConnectError").classList.add("hidden");
    document.getElementById("btnDisconnectGithub").classList.toggle("hidden", !state.githubConnected);
    document.getElementById("githubModal").classList.remove("hidden");
  };
  document.getElementById("btnGithubConnect").addEventListener("click", openGithubModal);
  document.getElementById("btnCloseGithubModal").addEventListener("click", () =>
    document.getElementById("githubModal").classList.add("hidden")
  );
  document.getElementById("btnCancelGithubModal").addEventListener("click", () =>
    document.getElementById("githubModal").classList.add("hidden")
  );
  document.getElementById("btnDisconnectGithub").addEventListener("click", () => {
    sessionStorage.removeItem(GH_SESSION_KEY);
    localStorage.removeItem(GH_LOCAL_KEY);
    state.githubConnected = false;
    state.githubConfig.token = "";
    document.getElementById("githubModal").classList.add("hidden");
    syncActionButtonsForMode();
    renderDashboard();
    showToast("GitHub 編集連携を解除しました。");
  });
  document.getElementById("githubForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const errEl = document.getElementById("ghConnectError");
    const btn = document.getElementById("btnSaveGithub");
    errEl.classList.add("hidden");
    btn.disabled = true;
    btn.textContent = "⏳ 接続確認中...";

    const nextCfg = {
      owner: form.gh_owner.value.trim(),
      repo: form.gh_repo.value.trim(),
      token: form.gh_token.value.trim(),
      branch: "main",
    };
    const prevCfg = { ...state.githubConfig };
    state.githubConfig = nextCfg;
    try {
      const resp = await ghApiFetch("");
      if (!resp.ok) {
        throw new Error(`接続できませんでした (HTTP ${resp.status})。リポジトリ名とトークンを確認してください。`);
      }
      state.githubConnected = true;
      const serialized = JSON.stringify(nextCfg);
      sessionStorage.setItem(GH_SESSION_KEY, serialized);
      if (form.gh_remember.checked) {
        localStorage.setItem(GH_LOCAL_KEY, serialized);
      } else {
        localStorage.removeItem(GH_LOCAL_KEY);
      }
      document.getElementById("githubModal").classList.add("hidden");
      syncActionButtonsForMode();
      renderDashboard();
      showToast("GitHub 編集連携を有効化しました！このページから直接パターンの追加・編集・削除ができます。");
    } catch (err) {
      state.githubConfig = prevCfg;
      errEl.textContent = err.message;
      errEl.classList.remove("hidden");
    } finally {
      btn.disabled = false;
      btn.textContent = "接続して編集を有効化";
    }
  });

  document.getElementById("btnLogout").addEventListener("click", handleLogout);
  document.getElementById("loginForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const passInput = document.getElementById("loginPassphrase");
    const errEl = document.getElementById("loginError");
    const btn = document.getElementById("btnUnlock");
    btn.disabled = true;
    btn.textContent = "⏳ 復号中...";
    errEl.classList.add("hidden");
    try {
      await tryUnlockWithPassphrase(passInput.value);
    } catch {
      errEl.classList.remove("hidden");
      passInput.select();
    } finally {
      btn.disabled = false;
      btn.textContent = "🔓 復号してログイン";
    }
  });

  loadDashboard();
});
