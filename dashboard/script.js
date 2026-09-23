const API_URL = "http://127.0.0.1:8000/scan_video";

const scanBtn = document.getElementById("scan-btn");
const loading = document.getElementById("loading");
const errorBox = document.getElementById("error-box");
const ticker = document.getElementById("ticker");
const logSection = document.getElementById("log-section");
const logList = document.getElementById("log-list");
const logCount = document.getElementById("log-count");

scanBtn.addEventListener("click", runScan);

async function runScan() {
  const videoId = document.getElementById("video-id-input").value.trim();
  const maxComments = parseInt(document.getElementById("max-comments-input").value) || 100;

  errorBox.classList.add("hidden");
  if (!videoId) {
    showError("Enter a YouTube video ID first.");
    return;
  }

  loading.classList.remove("hidden");
  ticker.classList.add("hidden");
  logSection.classList.add("hidden");
  logList.innerHTML = "";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: videoId, max_comments: maxComments }),
    });
    const data = await response.json();

    loading.classList.add("hidden");

    if (data.error) {
      showError(data.error);
      return;
    }

    renderTicker(data);
    renderLog(data);

  } catch (e) {
    loading.classList.add("hidden");
    showError("Could not reach the backend. Is the server running on port 8000?");
    console.error(e);
  }
}

function showError(msg) {
  errorBox.innerText = msg;
  errorBox.classList.remove("hidden");
}

function renderTicker(data) {
  document.getElementById("stat-time-saved").innerText = data.time_saved_minutes;
  document.getElementById("stat-percent-filtered").innerText = data.percent_filtered + "%";
  document.getElementById("stat-flagged").innerText = data.summary["Auto-Flag"] || 0;
  document.getElementById("stat-review").innerText = data.summary["Needs Review"] || 0;
  document.getElementById("stat-live").innerText = data.flagged_still_live_count;
  ticker.classList.remove("hidden");
}

function renderLog(data) {
  const flagged = data.results.filter(r => r.tier !== "No Action");
  logCount.innerText = `${flagged.length} of ${data.total_comments} comments need attention`;

  if (flagged.length === 0) {
    logList.innerHTML = `<p style="color:var(--ink-soft); font-size:14px;">Nothing flagged in this batch.</p>`;
  }

  for (const r of flagged) {
    const row = document.createElement("div");
    const tierClass = r.tier === "Auto-Flag" ? "tier-flag" : "tier-review";
    const tagClass = r.tier === "Auto-Flag" ? "flag" : "review";
    row.className = `log-row ${tierClass}`;

    const confidence = r.threeclass ? Math.round(r.threeclass.confidence * 100) + "%" : "—";
    const termsHtml = (r.matched_terms && r.matched_terms.length)
      ? `<div class="terms">${r.matched_terms.map(t => `<span>${escapeHtml(t)}</span>`).join("")}</div>`
      : "";

    row.innerHTML = `
      <div class="log-row-top">
        <div class="log-comment">${escapeHtml(r.comment)}</div>
        <div class="log-meta">
          <span class="log-confidence">${confidence}</span>
          <span class="log-tag ${tagClass}">${r.tier === "Auto-Flag" ? r.final_label : "Review"}</span>
          <span class="log-explain-toggle">why?</span>
        </div>
      </div>
      <div class="log-explain">
        <div>${escapeHtml(r.explanation || "")}</div>
        <div style="margin-top:6px;"><strong>Action:</strong> ${escapeHtml(r.recommended_action || "")}</div>
        ${termsHtml}
      </div>
    `;

    row.addEventListener("click", () => row.classList.toggle("expanded"));
    logList.appendChild(row);
  }

  logSection.classList.remove("hidden");
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.innerText = text;
  return div.innerHTML;
}