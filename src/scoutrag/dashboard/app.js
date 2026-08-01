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
  answerMode: document.querySelector("#answer-mode-badge"),
  answerText: document.querySelector("#answer-text"),
  answerCitations: document.querySelector("#answer-citations"),
  answerGrounding: document.querySelector("#answer-grounding"),
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
  sufficient: "Sufficient",
  limited: "Limited",
  insufficient: "Insufficient",
  conflicting: "Conflicting",
  out_of_scope: "Out of scope",
};

const factorLabels = {
  data_coverage: "Data coverage",
  played_minutes: "Minutes played",
  feature_availability: "Feature availability",
  requested_trait_coverage: "Requested traits",
  retrieval_agreement: "Retriever agreement",
  ranking_separation: "Ranking separation",
  comparison_group_availability: "Comparison group",
  season_consistency: "Season consistency",
  missing_value_completeness: "Value completeness",
  hard_filter_fulfillment: "Hard filters",
};

const generationModeLabels = {
  template: "Template mode",
  grounded_model: "Grounded model",
  safe_fallback: "Safe fallback",
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
    showError(error.message || "Unknown error");
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
    candidates.length === 1 ? "player" : "players"
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
      `${formatNumber(candidate.profile.minutes_played, 0)} min`,
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
    detail.textContent = "View evidence";
    detail.addEventListener("click", () => openPlayer(candidate, pack));
    card.append(rank, main, detail);
    elements.candidateList.append(card);
  });
}

function renderTrace(trace) {
  elements.strategies.replaceChildren();
  elements.timings.replaceChildren();
  Object.entries(trace.candidates_per_strategy).forEach(([name, count]) => {
    elements.strategies.append(traceRow(humanize(name), `${count} candidates`));
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
  elements.answerButton.textContent = "Checking Evidence Pack …";
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
    elements.answerMode.textContent =
      generationModeLabels[answer.generation_mode] || answer.generation_mode;
    elements.answerText.textContent = answer.text;
    elements.answerCitations.textContent = answer.cited_player_ids.length
      ? `Cited player IDs: ${answer.cited_player_ids.join(", ")}`
      : "No player citations — the system abstains.";
    const grounding = answer.grounding;
    elements.answerGrounding.textContent = grounding.fallback_used
      ? `Grounding check blocked: ${grounding.violations.join("; ")}`
      : grounding.claim_count
        ? `Grounding ${formatNumber(grounding.grounding_score, 3)} · ${grounding.supported_claim_count}/${grounding.claim_count} claims · ${grounding.cited_fact_ids.length} fact IDs`
        : "Deterministic projection of the Evidence Pack";
    elements.answerOutput.hidden = false;
  } catch (error) {
    showError(error.message || "Answer could not be generated.");
  } finally {
    elements.answerButton.disabled = false;
    elements.answerButton.textContent = "Generate explanation";
  }
}

function openPlayer(candidate, pack) {
  const profile = candidate.profile;
  const evidence = pack.metric_evidence[profile.profile_id || profile.player_id] || [];
  const temporal = pack.temporal_context?.[profile.player_id] || null;
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
    ["Minutes", formatNumber(profile.minutes_played, 0)],
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
  ["Metric", "Raw value", "Normalized", "Percentile", "Sample"].forEach((label) => {
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
  body.append(stats);
  if (temporal) {
    body.append(renderTemporalContext(temporal, profile, pack.query_profile.requested_metrics));
  }
  body.append(table);
  elements.dialogContent.append(hero, body);
  elements.dialog.showModal();
}

function renderTemporalContext(context, currentProfile, requestedMetrics) {
  const section = node("section", "temporal-context");
  const heading = document.createElement("h3");
  heading.textContent = "Season history & form";
  const note = document.createElement("p");
  note.className = "temporal-note";
  note.textContent =
    "The newest season is authoritative. Earlier seasons remain separate context and fallback evidence.";
  section.append(heading, note);

  const seasons = node("div", "season-timeline");
  context.season_profiles.slice(0, 4).forEach((item, index) => {
    const card = node("article", `season-card ${index === 0 ? "current" : ""}`);
    const badge = document.createElement("span");
    badge.textContent = index === 0 ? "CURRENT" : "HISTORICAL";
    const title = document.createElement("strong");
    title.textContent = item.season_name;
    const team = document.createElement("p");
    team.textContent = `${item.team_name} · ${item.competition_name}`;
    const sample = document.createElement("small");
    sample.textContent = `${formatNumber(item.minutes_played, 0)} min · Data Q ${formatNumber(item.data_quality, 2)}`;
    card.append(badge, title, team, sample);
    seasons.append(card);
  });
  section.append(seasons);

  const currentForm = context.recent_forms.find(
    (item) => item.profile_id === (currentProfile.profile_id || currentProfile.player_id),
  );
  if (currentForm) {
    const form = node("div", "form-summary");
    const title = document.createElement("strong");
    title.textContent = "Recent matches";
    const detail = document.createElement("span");
    detail.textContent = `${currentForm.matches_in_window} matches · ${formatNumber(currentForm.minutes_in_window, 0)} min · Form data Q ${formatNumber(currentForm.data_quality, 2)}`;
    form.append(title, detail);
    section.append(form);
  }

  const requested = new Set(requestedMetrics || []);
  const trends = context.season_trends
    .filter(
      (item) =>
        item.current_profile_id === (currentProfile.profile_id || currentProfile.player_id) &&
        (requested.size === 0 || requested.has(item.metric_name)),
    )
    .slice(0, 6);
  if (trends.length) {
    const trendList = node("div", "trend-list");
    trends.forEach((item) => {
      const row = node("div", "trend-row");
      const metric = document.createElement("span");
      metric.textContent = humanize(item.metric_name);
      const direction = document.createElement("strong");
      direction.textContent = humanize(item.direction);
      direction.dataset.direction = item.direction;
      row.append(metric, direction);
      trendList.append(row);
    });
    section.append(trendList);
  }

  const external = context.external_context;
  if (external) {
    const externalSection = node("div", "external-context");
    const title = document.createElement("strong");
    title.textContent = "Biography (Wikidata, CC0)";
    externalSection.append(title);
    const facts = [
      external.national_team_name
        ? `${external.national_team_name} · ${external.national_team_caps ?? 0} caps`
        : null,
      external.footedness ? `Footedness: ${external.footedness}` : null,
      external.earlier_clubs?.length ? `Earlier clubs: ${external.earlier_clubs.join(", ")}` : null,
      external.honours?.length ? `Honours: ${external.honours.join(", ")}` : null,
    ].filter(Boolean);
    if (facts.length) {
      const list = node("ul", "external-context-list");
      facts.forEach((fact) => {
        const item = document.createElement("li");
        item.textContent = fact;
        list.append(item);
      });
      externalSection.append(list);
      section.append(externalSection);
    }
  }

  const career = context.career_events;
  if (career) {
    const careerSection = node("div", "external-context");
    const title = document.createElement("strong");
    title.textContent = "Career events (API-Football)";
    careerSection.append(title);
    const facts = [
      career.transfers?.length
        ? `Transfers: ${career.transfers
            .map((item) => `${item.from_team} → ${item.to_team} (${item.fee_text})`)
            .join("; ")}`
        : null,
      career.trophies?.length
        ? `Trophies: ${career.trophies.length} recorded (latest: ${career.trophies[0].competition_name} ${career.trophies[0].season}, ${career.trophies[0].place})`
        : null,
      career.injury_spells?.length
        ? `Injury history: ${career.injury_spells.length} recorded spell(s)`
        : null,
    ].filter(Boolean);
    if (facts.length) {
      const list = node("ul", "external-context-list");
      facts.forEach((fact) => {
        const item = document.createElement("li");
        item.textContent = fact;
        list.append(item);
      });
      careerSection.append(list);
      section.append(careerSection);
    }
  }

  return section;
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
  return "API error without detail.";
}

function humanize(value) {
  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatNumber(value, digits = 2) {
  return Number(value).toLocaleString("en-US", {
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
