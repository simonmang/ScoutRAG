"use strict";

const API_PREFIX = "/api/v1";

const state = {
  pack: null,
};

const elements = {
  form: document.querySelector("#search-form"),
  query: document.querySelector("#query-input"),
  resultCount: document.querySelector("#result-count"),
  loading: document.querySelector("#loading-state"),
  results: document.querySelector("#results"),
  queryId: document.querySelector("#query-id"),
  runtime: document.querySelector("#total-runtime"),
  verdict: document.querySelector("#verdict-badge"),
  quality: document.querySelector("#quality-score"),
  reason: document.querySelector("#verdict-reason"),
  warnings: document.querySelector("#warnings"),
  factors: document.querySelector("#factor-grid"),
  candidateCount: document.querySelector("#candidate-count"),
  candidateList: document.querySelector("#candidate-list"),
  emptyCandidates: document.querySelector("#empty-candidates"),
  answerButton: document.querySelector("#answer-button"),
  answerOutput: document.querySelector("#answer-output"),
  answerVerdict: document.querySelector("#answer-verdict"),
  answerText: document.querySelector("#answer-text"),
  answerCitations: document.querySelector("#answer-citations"),
  traceToggle: document.querySelector("#trace-toggle"),
  traceContent: document.querySelector("#trace-content"),
  strategies: document.querySelector("#strategy-list"),
  timings: document.querySelector("#timing-list"),
  filters: document.querySelector("#filter-output"),
  dialog: document.querySelector("#player-dialog"),
  dialogContent: document.querySelector("#dialog-content"),
  dialogClose: document.querySelector("#dialog-close"),
  error: document.querySelector("#error-toast"),
  errorText: document.querySelector("#error-text"),
  errorClose: document.querySelector("#error-close"),
};

const verdictLabels = {
  sufficient: "Belastbar",
  limited: "Eingeschränkt",
  insufficient: "Nicht ausreichend",
  conflicting: "Widersprüchlich",
  out_of_scope: "Außerhalb Scope",
};

const factorLabels = {
  data_coverage: "Datenabdeckung",
  played_minutes: "Gespielte Minuten",
  feature_availability: "Feature-Abdeckung",
  requested_trait_coverage: "Angefragte Merkmale",
  retrieval_agreement: "Retriever-Übereinstimmung",
  ranking_separation: "Ranking-Abstand",
  comparison_group_availability: "Vergleichsgruppe",
  season_consistency: "Saison-Konsistenz",
  missing_value_completeness: "Werte-Vollständigkeit",
  hard_filter_fulfillment: "Harte Filter",
};

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await retrieve(elements.query.value.trim());
});

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", async () => {
    elements.query.value = button.dataset.example;
    await retrieve(button.dataset.example);
  });
});

elements.answerButton.addEventListener("click", renderAnswer);
elements.errorClose.addEventListener("click", () => {
  elements.error.hidden = true;
});
elements.traceToggle.addEventListener("click", () => {
  const expanded = elements.traceToggle.getAttribute("aria-expanded") === "true";
  elements.traceToggle.setAttribute("aria-expanded", String(!expanded));
  elements.traceContent.hidden = expanded;
});
elements.dialogClose.addEventListener("click", () => elements.dialog.close());
elements.dialog.addEventListener("click", (event) => {
  if (event.target === elements.dialog) {
    elements.dialog.close();
  }
});

async function retrieve(query) {
  if (!query) {
    return;
  }
  setLoading(true);
  elements.error.hidden = true;
  elements.answerOutput.hidden = true;
  try {
    const response = await fetch(`${API_PREFIX}/retrieve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        result_count: Number(elements.resultCount.value),
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(readError(payload));
    }
    state.pack = payload;
    renderPack(payload);
    elements.results.hidden = false;
    elements.results.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showError(error.message || "Unbekannter Fehler");
  } finally {
    setLoading(false);
  }
}

function renderPack(pack) {
  elements.queryId.textContent = `Query ${shortId(pack.retrieval_trace.query_id)}`;
  elements.runtime.textContent = `${formatNumber(pack.runtime_metrics.total_ms, 1)} ms`;
  elements.verdict.textContent = verdictLabels[pack.governance.verdict] || pack.governance.verdict;
  elements.verdict.dataset.verdict = pack.governance.verdict;
  elements.quality.textContent = formatNumber(pack.governance.evidence_quality_score, 3);
  elements.reason.textContent = pack.governance.reasons.join(" ");
  renderNotices(pack);
  renderFactors(pack.governance.factors);
  renderCandidates(pack);
  renderTrace(pack.retrieval_trace);
}

function renderNotices(pack) {
  elements.warnings.replaceChildren();
  const notices = [
    ...pack.governance.warnings.map((text) => ({ text, type: "warning" })),
    ...pack.governance.missing_evidence.map((text) => ({ text, type: "missing" })),
  ];
  notices.forEach((notice) => {
    const item = node("div", `notice ${notice.type === "missing" ? "missing" : ""}`);
    item.textContent = notice.text;
    elements.warnings.append(item);
  });
}

function renderFactors(factors) {
  elements.factors.replaceChildren();
  Object.entries(factors).forEach(([name, value]) => {
    const item = node("div", "factor-item");
    const label = node("span");
    label.textContent = factorLabels[name] || humanize(name);
    label.title = label.textContent;
    const score = node("strong");
    score.textContent = formatNumber(value, 3);
    const bar = node("div", "factor-bar");
    const fill = document.createElement("i");
    fill.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
    bar.append(fill);
    item.append(label, score, bar);
    elements.factors.append(item);
  });
}

function renderCandidates(pack) {
  const candidates = pack.candidates;
  elements.candidateList.replaceChildren();
  elements.candidateCount.textContent = `${candidates.length} ${
    candidates.length === 1 ? "Spieler" : "Spieler"
  }`;
  elements.emptyCandidates.hidden = candidates.length > 0;
  candidates.forEach((candidate) => {
    const card = node("article", "candidate-card");
    const rank = node("span", "rank-number");
    rank.textContent = String(candidate.rank).padStart(2, "0");

    const main = node("div", "candidate-main");
    const name = document.createElement("h4");
    name.textContent = candidate.profile.player_name;
    const context = document.createElement("p");
    context.textContent = `${candidate.profile.team_name} · ${candidate.profile.competition_name} ${candidate.profile.season_name}`;
    const tags = node("div", "candidate-tags");
    [
      humanize(candidate.profile.position_group),
      `${formatNumber(candidate.profile.minutes_played, 0)} Min.`,
      `Data Q ${formatNumber(candidate.profile.data_quality, 2)}`,
      ...candidate.retrieval_trace.retrieved_by.slice(0, 3),
    ].forEach((text) => {
      const tag = document.createElement("span");
      tag.textContent = text;
      tags.append(tag);
    });
    main.append(name, context, tags);

    const detail = node("button", "detail-button");
    detail.type = "button";
    detail.textContent = "Evidence ansehen";
    detail.addEventListener("click", () => openPlayer(candidate, pack));
    card.append(rank, main, detail);
    elements.candidateList.append(card);
  });
}

function renderTrace(trace) {
  elements.strategies.replaceChildren();
  elements.timings.replaceChildren();
  Object.entries(trace.candidates_per_strategy).forEach(([name, count]) => {
    elements.strategies.append(traceRow(humanize(name), `${count} Kandidaten`));
  });
  Object.entries(trace.stage_timings_ms).forEach(([name, duration]) => {
    elements.timings.append(traceRow(humanize(name), `${formatNumber(duration, 2)} ms`));
  });
  elements.filters.textContent = JSON.stringify(trace.filters_applied, null, 2);
}

async function renderAnswer() {
  if (!state.pack) {
    return;
  }
  elements.answerButton.disabled = true;
  elements.answerButton.textContent = "Evidence Pack wird geprüft …";
  try {
    const response = await fetch(`${API_PREFIX}/answer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ evidence_pack: state.pack }),
    });
    const answer = await response.json();
    if (!response.ok) {
      throw new Error(readError(answer));
    }
    elements.answerVerdict.textContent = verdictLabels[answer.verdict] || answer.verdict;
    elements.answerText.textContent = answer.text;
    elements.answerCitations.textContent = answer.cited_player_ids.length
      ? `Belegte Player IDs: ${answer.cited_player_ids.join(", ")}`
      : "Keine Spielerzitate — das System enthält sich.";
    elements.answerOutput.hidden = false;
  } catch (error) {
    showError(error.message || "Antwort konnte nicht erzeugt werden.");
  } finally {
    elements.answerButton.disabled = false;
    elements.answerButton.textContent = "Erklärung erzeugen";
  }
}

function openPlayer(candidate, pack) {
  const profile = candidate.profile;
  const evidence = pack.metric_evidence[profile.player_id] || [];
  elements.dialogContent.replaceChildren();

  const hero = node("div", "dialog-hero");
  const rank = node("span", "rank-number");
  rank.textContent = String(candidate.rank).padStart(2, "0");
  const name = document.createElement("h2");
  name.textContent = profile.player_name;
  const context = document.createElement("p");
  context.textContent = `${profile.team_name} · ${profile.competition_name} · ${profile.season_name}`;
  hero.append(rank, name, context);

  const body = node("div", "dialog-body");
  const stats = node("div", "profile-stats");
  [
    ["Position", humanize(profile.position_group)],
    ["Minuten", formatNumber(profile.minutes_played, 0)],
    ["Data Quality", formatNumber(profile.data_quality, 3)],
  ].forEach(([label, value]) => {
    const item = node("div", "profile-stat");
    const itemLabel = document.createElement("span");
    itemLabel.textContent = label;
    const itemValue = document.createElement("strong");
    itemValue.textContent = value;
    item.append(itemLabel, itemValue);
    stats.append(item);
  });

  const table = node("table", "metric-table");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["Metrik", "Rohwert", "Normalisiert", "Perzentil", "Sample"].forEach((label) => {
    const cell = document.createElement("th");
    cell.textContent = label;
    headRow.append(cell);
  });
  head.append(headRow);
  const tableBody = document.createElement("tbody");
  evidence.forEach((item) => {
    const row = document.createElement("tr");
    [
      humanize(item.metric_name),
      nullableNumber(item.raw_value),
      nullableNumber(item.normalized_value),
      item.percentile == null ? "—" : `P${formatNumber(item.percentile, 0)}`,
      nullableNumber(item.sample_size),
    ].forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    });
    tableBody.append(row);
  });
  table.append(head, tableBody);
  body.append(stats, table);
  elements.dialogContent.append(hero, body);
  elements.dialog.showModal();
}

function traceRow(label, value) {
  const row = node("div", "trace-row");
  const left = document.createElement("span");
  left.textContent = label;
  const right = document.createElement("strong");
  right.textContent = value;
  row.append(left, right);
  return row;
}

function node(tag, className = "") {
  const element = document.createElement(tag);
  if (className) {
    element.className = className;
  }
  return element;
}

function setLoading(active) {
  elements.loading.hidden = !active;
  elements.form.querySelector("button[type='submit']").disabled = active;
}

function showError(message) {
  elements.errorText.textContent = message;
  elements.error.hidden = false;
}

function readError(payload) {
  if (typeof payload.detail === "string") {
    return payload.detail;
  }
  if (Array.isArray(payload.detail)) {
    return payload.detail.map((item) => item.msg).join("; ");
  }
  return "API-Fehler ohne Detail.";
}

function humanize(value) {
  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatNumber(value, digits = 2) {
  return Number(value).toLocaleString("de-DE", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function nullableNumber(value) {
  return value == null ? "—" : formatNumber(value, 2);
}

function shortId(value) {
  return value ? String(value).slice(0, 8) : "—";
}
