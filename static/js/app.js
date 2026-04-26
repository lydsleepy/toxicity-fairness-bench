'use strict';

// ── State ──────────────────────────────────────────────────────────────────────
const state = {
  selectedModels: new Set(),
  selectedAttr: '',
};

// ── Chart palette ──────────────────────────────────────────────────────────────
const COLORS = ['#1a7a6e', '#C4899A', '#A89BB8', '#7FA8BC', '#B89AA8', '#9AADB8'];
const FONT   = "'DM Sans', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif";

// ── Plotly base layout ─────────────────────────────────────────────────────────
function baseLayout(extra = {}) {
  return Object.assign({
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor:  'rgba(0,0,0,0)',
    font: { family: FONT, color: '#6e6e73', size: 12 },
    margin: { l: 16, r: 16, t: 20, b: 8, pad: 0 },
    xaxis: {
      gridcolor: 'rgba(0,0,0,0.05)',
      linecolor: 'rgba(0,0,0,0.08)',
      tickfont: { size: 11, color: '#98989d' },
      zeroline: false,
      automargin: true,
    },
    yaxis: {
      gridcolor: 'rgba(0,0,0,0.05)',
      linecolor: 'rgba(0,0,0,0.08)',
      tickfont: { size: 11, color: '#98989d' },
      zeroline: false,
      automargin: true,
    },
    legend: {
      bgcolor: 'rgba(255,255,255,0.6)',
      borderwidth: 0,
      font: { size: 11, color: '#6e6e73' },
      orientation: 'h',
      yanchor: 'bottom',
      y: 1.02,
      xanchor: 'left',
      x: 0,
    },
  }, extra);
}

const PLOTLY_CONFIG = { responsive: true, displayModeBar: false };

// ── Init ───────────────────────────────────────────────────────────────────────
async function init() {
  let filters;
  try {
    const res = await fetch('/api/filters');
    filters = await res.json();
  } catch (err) {
    showError(`Could not reach the API: ${err.message}`);
    return;
  }

  if (filters.error) {
    showError(`Failed to load benchmark data: ${filters.error}`);
    return;
  }

  if (!filters.models || !filters.models.length) {
    showNoData();
    return;
  }

  buildModelFilters(filters.models);
  buildAttrSelect(filters.protected_attributes);
  attachListeners();
  await fetchAndRender();
}

// ── Build sidebar controls ─────────────────────────────────────────────────────
function buildModelFilters(models) {
  const el = document.getElementById('model-filters');
  models.forEach(m => {
    state.selectedModels.add(m);
    const label = document.createElement('label');
    label.className = 'chip-label checked';
    label.dataset.model = m;
    label.innerHTML = `<input type="checkbox" checked value="${escHtml(m)}" />${escHtml(displayModel(m))}`;
    el.appendChild(label);
  });
}

function buildAttrSelect(attrs) {
  const el = document.getElementById('attr-select');
  attrs.forEach(a => el.add(new Option(a, a)));
  state.selectedAttr = attrs[0] || '';
}

// ── Event listeners ────────────────────────────────────────────────────────────
function attachListeners() {
  // Model chips
  document.getElementById('model-filters').addEventListener('change', e => {
    if (e.target.type !== 'checkbox') return;
    const label = e.target.closest('.chip-label');
    const m = label.dataset.model;
    if (e.target.checked) {
      state.selectedModels.add(m);
      label.classList.add('checked');
    } else {
      state.selectedModels.delete(m);
      label.classList.remove('checked');
    }
    fetchAndRender();
  });

  // Attribute dropdown
  document.getElementById('attr-select').addEventListener('change', e => {
    state.selectedAttr = e.target.value;
    fetchAndRender();
  });

  // Tab buttons
  document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => {
        t.classList.remove('active');
        t.setAttribute('aria-selected', 'false');
      });
      document.querySelectorAll('.tab-panel').forEach(p => { p.hidden = true; });
      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');
      const panel = document.getElementById(`tab-${btn.dataset.tab}`);
      if (panel) {
        panel.hidden = false;
        // Re-layout charts if switching to a tab with charts
        setTimeout(() => {
          ['chart-accuracy-group', 'chart-scatter'].forEach(id => {
            const el = document.getElementById(id);
            if (el && el.querySelector('.js-plotly-plot')) Plotly.Plots.resize(el);
          });
        }, 50);
      }
    });
  });

  // Scorer
  document.getElementById('score-btn').addEventListener('click', runScorer);
  document.getElementById('score-input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) runScorer();
  });
}

// ── Fetch & render ─────────────────────────────────────────────────────────────
async function fetchAndRender() {
  if (state.selectedModels.size === 0) {
    clearCharts();
    return;
  }

  setLoading(true);
  try {
    const params = new URLSearchParams();
    state.selectedModels.forEach(m => params.append('models', m));
    if (state.selectedAttr) params.set('attribute', state.selectedAttr);

    const data = await fetch(`/api/metrics?${params}`).then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    });

    renderTiles(data.accuracy_tiles);
    renderAccuracyChart(data.accuracy_by_group);
    renderFairnessTable(data.fairness_report);
    renderFairnessSkewWarning(data.skewed_subgroups);
    renderScatterChart(data.scatter_points);

    const titleEl = document.getElementById('group-chart-title');
    if (titleEl) titleEl.textContent = `Accuracy by ${state.selectedAttr} Group`;
  } catch (err) {
    console.error('fetchAndRender error:', err);
  } finally {
    setLoading(false);
  }
}

// ── Render: accuracy tiles ─────────────────────────────────────────────────────
function renderTiles(tiles) {
  const el = document.getElementById('accuracy-tiles');
  if (!el) return;
  el.innerHTML = tiles.map(t => `
    <div class="tile">
      <div class="tile-label">Overall Accuracy</div>
      <div class="tile-value">${fmtPct(t.accuracy)}</div>
      <div class="tile-model">${escHtml(t.model)}</div>
    </div>
  `).join('');
}

// ── Render: accuracy bar chart ─────────────────────────────────────────────────
function renderAccuracyChart(rows) {
  const el = document.getElementById('chart-accuracy-group');
  if (!el || !rows.length) return;

  const models = [...new Set(rows.map(r => r.model))];
  const traces = models.map((model, i) => {
    const pts = rows.filter(r => r.model === model);
    return {
      type: 'bar',
      name: model,
      x: pts.map(r => r.group),
      y: pts.map(r => r.accuracy),
      text: pts.map(r => `${(r.accuracy * 100).toFixed(0)}%`),
      textposition: 'outside',
      textfont: { size: 10, color: '#98989d' },
      marker: {
        color: COLORS[i % COLORS.length],
        line: { width: 0 },
        opacity: 0.85,
      },
      hovertemplate: '<b>%{x}</b><br>Accuracy: %{y:.1%}<extra>%{fullData.name}</extra>',
    };
  });

  Plotly.react(el, traces, baseLayout({
    barmode: 'group',
    bargap: 0.24,
    bargroupgap: 0.08,
    height: 280,
    yaxis: { tickformat: '.0%', range: [0, 1.14], gridcolor: 'rgba(0,0,0,0.05)' },
  }), PLOTLY_CONFIG);
}

// ── Render: fairness table ─────────────────────────────────────────────────────
function renderFairnessTable(rows) {
  const el = document.getElementById('fairness-table-wrapper');
  if (!el) return;

  const cols = [
    { key: 'model',            label: 'Model',         isGap: false },
    { key: 'overall_accuracy', label: 'Overall Acc.',  isGap: false },
    { key: 'accuracy_gap',     label: 'Acc. Gap',      isGap: true  },
    { key: 'dp_gap',           label: 'DP Gap',        isGap: true  },
    { key: 'tpr_gap',          label: 'TPR Gap',       isGap: true  },
    { key: 'fpr_gap',          label: 'FPR Gap',       isGap: true  },
  ];

  const header = `<tr>${cols.map(c => `<th>${c.label}</th>`).join('')}</tr>`;
  const body = rows.map(row => {
    const cells = cols.map(c => {
      if (c.key === 'model') return `<td>${escHtml(row[c.key])}</td>`;
      const v = row[c.key];
      if (v == null) return `<td class="na">n/a</td>`;
      const pct = fmtPct(v);
      let cls = '';
      if (c.isGap) {
        if (v > 0.3) cls = 'gap-high';
        else if (v < 0.15) cls = 'gap-low';
      }
      return `<td${cls ? ` class="${cls}"` : ''}>${pct}</td>`;
    });
    return `<tr>${cells.join('')}</tr>`;
  }).join('');

  el.innerHTML = `<table class="data-table"><thead>${header}</thead><tbody>${body}</tbody></table>`;
}

// ── Render: skew warning ───────────────────────────────────────────────────────
function renderFairnessSkewWarning(skewed) {
  const el = document.getElementById('fairness-skew-warning');
  if (!el) return;
  if (!skewed || skewed.length === 0) {
    el.hidden = true;
    return;
  }
  const count = skewed.length;
  el.querySelector('.skew-msg').textContent =
    `${count} subgroup${count !== 1 ? 's' : ''} excluded from gap metrics due to ` +
    `insufficient class representation (fewer than 30 examples of either class): ` +
    `${skewed.join(', ')}.`;
  el.hidden = false;
}

// ── Render: FPR vs FNR scatter ─────────────────────────────────────────────────
function renderScatterChart(rows) {
  const el = document.getElementById('chart-scatter');
  if (!el || !rows.length) return;

  const models = [...new Set(rows.map(r => r.model))];
  const traces = models.map((model, i) => {
    const pts = rows.filter(r => r.model === model);
    return {
      type: 'scatter',
      mode: 'markers+text',
      name: model,
      x: pts.map(p => p.fpr),
      y: pts.map(p => p.fnr),
      text: pts.map(p => p.group),
      textposition: 'top center',
      textfont: { size: 10, color: '#98989d' },
      marker: {
        size: pts.map(p => Math.max(8, Math.sqrt(p.n) * 2.2)),
        color: COLORS[i % COLORS.length],
        opacity: 0.82,
        line: { width: 0 },
      },
      hovertemplate: '<b>%{text}</b><br>FPR: %{x:.2f}<br>FNR: %{y:.2f}<extra>%{fullData.name}</extra>',
    };
  });

  const diagonal = {
    type: 'scatter',
    mode: 'lines',
    showlegend: false,
    x: [0, 1],
    y: [0, 1],
    line: { color: 'rgba(0,0,0,0.12)', dash: 'dot', width: 1.5 },
    hoverinfo: 'skip',
  };

  Plotly.react(el, [...traces, diagonal], baseLayout({
    height: 340,
    margin: { l: 56, r: 16, t: 20, b: 48, pad: 0 },
    xaxis: { title: { text: 'False Positive Rate', font: { size: 11, color: '#98989d' } }, range: [-0.06, 1.06] },
    yaxis: { title: { text: 'False Negative Rate', font: { size: 11, color: '#98989d' } }, range: [-0.06, 1.06] },
  }), PLOTLY_CONFIG);
}

// ── Scorer ─────────────────────────────────────────────────────────────────────
async function runScorer() {
  const text = document.getElementById('score-input').value.trim();
  if (!text) return;

  const btn = document.getElementById('score-btn');
  const resultsEl = document.getElementById('scorer-results');

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-sm" aria-hidden="true"></span> Analyzing...';
  resultsEl.hidden = true;

  try {
    const { results } = await fetch('/api/score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    }).then(r => r.json());

    resultsEl.innerHTML = results.map(renderScorerCard).join('');
    resultsEl.hidden = false;
  } catch (err) {
    resultsEl.innerHTML = `<div class="alert alert-error"><span>Request failed: ${escHtml(err.message)}</span></div>`;
    resultsEl.hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Analyze';
  }
}

function renderScorerCard(r) {
  if (r.error && !r.score) {
    return `<div class="scorer-result-card error">
      <span class="scorer-model-name">${escHtml(r.model)}</span>
      <span class="scorer-error-msg">${escHtml(r.error)}</span>
    </div>`;
  }
  const isToxic = r.label === 'toxic';
  const chipClass = isToxic ? 'chip-toxic' : 'chip-safe';
  const labelText = r.label ?? 'unknown';
  return `<div class="scorer-result-card">
    <span class="scorer-model-name">${escHtml(r.model)}</span>
    <span class="scorer-score">${r.score_pct ?? '—'}</span>
    <span class="scorer-label-chip ${chipClass}">${escHtml(labelText)}</span>
  </div>`;
}

// ── Helpers ────────────────────────────────────────────────────────────────────
function displayModel(key) {
  const map = {
    'perspective': 'Perspective',
    'claude/claude-haiku-4-5-20251001': 'Claude Haiku',
    'gemini': 'Gemini',
  };
  return map[key] || key.split('/').pop().replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function fmtPct(v) {
  if (v == null) return 'n/a';
  return `${(v * 100).toFixed(1)}%`;
}

function escHtml(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function setLoading(on) {
  const el = document.getElementById('loading-state');
  if (el) el.hidden = !on;
}

function showNoData() {
  const el = document.getElementById('no-data-alert');
  if (el) el.hidden = false;
  setLoading(false);
}

function showError(msg) {
  const el = document.getElementById('error-alert');
  if (el) {
    el.querySelector('.error-msg').textContent = msg;
    el.hidden = false;
  }
  setLoading(false);
}

function clearCharts() {
  ['chart-accuracy-group', 'chart-scatter'].forEach(id => {
    const el = document.getElementById(id);
    if (el) Plotly.purge(el);
  });
  const tiles = document.getElementById('accuracy-tiles');
  if (tiles) tiles.innerHTML = '';
  const table = document.getElementById('fairness-table-wrapper');
  if (table) table.innerHTML = '';
}

// ── Boot ───────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
