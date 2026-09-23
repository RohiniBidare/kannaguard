// content.js - runs on YouTube video pages
// Scrapes visible comments, sends them to the BINARY model endpoint
// (fast, for live browsing), and highlights Toxic comments inline.

const API_URL = "http://127.0.0.1:8000/predict/ensemble";  // change this after you deploy to Hugging Face

const processedComments = new WeakSet();

let counts = { "Auto-Flag": 0, "Needs Review": 0, "No Action": 0 };

function saveCounts() {
  chrome.storage.local.set({ moderationCounts: counts });
}

async function classifyComment(text) {
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) return null;
    return await response.json();
  } catch (e) {
    console.error("Moderation API error:", e);
    return null;
  }
}

function applyLabel(commentElement, result) {
  if (!result) return;

  const { tier, final_label, threeclass, binary } = result;
  counts[tier] = (counts[tier] || 0) + 1;
  saveCounts();

  if (tier === "No Action") return;

  commentElement.classList.add("mod-flagged");
  commentElement.classList.add(tier === "Auto-Flag" ? "mod-toxic" : "mod-review");

  const conf = threeclass ? Math.round(threeclass.confidence * 100) : null;
  const badge = document.createElement("span");
  badge.className = "mod-badge";
  badge.innerText = tier === "Auto-Flag"
    ? `⚠ ${final_label}${conf !== null ? ` (${conf}%)` : ""}`
    : `❓ Needs Review${conf !== null ? ` (${conf}%)` : ""}`;
  commentElement.prepend(badge);
}

async function scanComments() {
  const commentNodes = document.querySelectorAll("#content-text");

  for (const node of commentNodes) {
    if (processedComments.has(node)) continue;
    processedComments.add(node);

    const text = node.innerText.trim();
    if (!text || text.length < 2) continue;

    const result = await classifyComment(text);
    applyLabel(node, result);
  }
}

scanComments();
const observer = new MutationObserver(() => {
  clearTimeout(window._modScanTimeout);
  window._modScanTimeout = setTimeout(scanComments, 800);
});
observer.observe(document.body, { childList: true, subtree: true });