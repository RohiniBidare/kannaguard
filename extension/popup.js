function updateDisplay(counts) {
  document.getElementById("flag-count").innerText = counts["Auto-Flag"] || 0;
  document.getElementById("review-count").innerText = counts["Needs Review"] || 0;
  document.getElementById("none-count").innerText = counts["No Action"] || 0;
}

chrome.storage.local.get("moderationCounts", (data) => {
  updateDisplay(data.moderationCounts || { "Auto-Flag": 0, "Needs Review": 0, "No Action": 0 });
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.moderationCounts) {
    updateDisplay(changes.moderationCounts.newValue);
  }
});