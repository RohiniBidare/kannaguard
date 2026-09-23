"""
FastAPI backend serving BOTH your trained models:
- Binary model (Toxic/Neutral) -> used by the browser extension for fast, live highlighting
- 3-class model (Hate Speech/Offensive/Neutral) -> used by the moderator dashboard for detail

Setup:
    pip install -r requirements.txt

Run locally:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.preprocessing import preprocess_kannada_english_text
import emoji
import re
import time

# ============================================
# CONFIG - paths to your two downloaded model folders
# ============================================
BINARY_MODEL_PATH = "./models/indicbertv2_binary_final"
THREECLASS_MODEL_PATH = "./models/indicbertv2_3class_final"

BINARY_LABELS = ["Neutral", "Toxic"]                       # must match your LabelEncoder order
THREECLASS_LABELS = ["Hate Speech", "Neutral", "Offensive"]  # must match your LabelEncoder order (alphabetical from sklearn)

# Severity mapping - used for the action recommendation feature
SEVERITY_ACTION = {
    "Hate Speech": "Remove + Escalate",
    "Offensive": "Warn user",
    "Toxic": "Review needed",
    "Neutral": "No action",
}

app = FastAPI(title="Kannada-English Dual-Model Moderation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

print("Loading binary model...")
binary_tokenizer = AutoTokenizer.from_pretrained(BINARY_MODEL_PATH)
binary_model = AutoModelForSequenceClassification.from_pretrained(BINARY_MODEL_PATH).to(device)
binary_model.eval()

print("Loading 3-class model...")
threeclass_tokenizer = AutoTokenizer.from_pretrained(THREECLASS_MODEL_PATH)
threeclass_model = AutoModelForSequenceClassification.from_pretrained(THREECLASS_MODEL_PATH).to(device)
threeclass_model.eval()

print("Both models loaded on", device)


class CommentRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    texts: list[str]


def is_classifiable(text: str) -> bool:
    """Skip comments too short or low-signal for reliable classification.
    Counts real words BEFORE emoji conversion, so emoji-only comments
    (which become fake 'word' descriptions after preprocessing) are
    correctly treated as having no real content."""
    if not isinstance(text, str):
        return False
    # Strip emojis entirely (not converted to text) just for counting purposes
    text_no_emoji = emoji.replace_emoji(text, replace="")
    text_no_emoji = re.sub(r'https?://\S+|www\.\S+', ' ', text_no_emoji)  # strip links too
    word_count = len(re.findall(r'\w+', text_no_emoji))
    return word_count >= 3


# ============================================
# Benign-pattern whitelist - catches common harmless comment types
# (greetings, self-introductions, generic requests) that the model
# sometimes misclassifies with high confidence due to training noise.
# Add more patterns here as you discover them through testing.
# ============================================
BENIGN_PATTERNS = [
    r'\bhow are you\b',
    r'\bhi+\b.*\bbro\b',
    r'\bhello\b.*\bbro\b',
    r'\bhii+\b',
    r'\bmobile\s*(no|number)\b',
    r'\bplease visit\b',
    r'\bnice video\b',
    r'\bsuper video\b',
    r'\bthank you\b',
    r'\bthanks for\b',
    r'\bcome to\b',
    r'\bplease visit\b',
]

def is_benign_pattern(text: str) -> bool:
    """True if the comment matches a known common harmless pattern."""
    lowered = text.lower()
    return any(re.search(p, lowered) for p in BENIGN_PATTERNS)


def _classify(text: str, tokenizer, model, labels: list[str]):
    if not is_classifiable(text):
        return {
            "label": "Neutral",
            "confidence": 0.0,
            "scores": {l: 0.0 for l in labels},
            "recommended_action": "No action (insufficient text to classify)",
            "skipped": True,
        }

    cleaned = preprocess_kannada_english_text(text)

    inputs = tokenizer(
        cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    label = labels[pred_idx]

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "scores": {labels[i]: round(probs[i].item(), 4) for i in range(len(labels))},
        "recommended_action": SEVERITY_ACTION.get(label, "No action"),
    }


# ============================================
# MITIGATION LAYER - reduces false positives without retraining
# ============================================

CONFIDENCE_THRESHOLD = 0.85  # below this, route to "Needs Review" instead of auto-flagging

# Keyword safety net: put your OWN list of genuinely hostile Kannada/English
# words/phrases here, drawn from your confirmed Hate Speech training examples.
# Load from a private local file so no word list needs to live in shared code.
# File format: one word/phrase per line, plain text, UTF-8.
HOSTILE_KEYWORDS_FILE = "./hostile_keywords.txt"

def _load_hostile_keywords():
    try:
        with open(HOSTILE_KEYWORDS_FILE, "r", encoding="utf-8") as f:
            return [line.strip().lower() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"WARNING: {HOSTILE_KEYWORDS_FILE} not found - keyword safety net disabled.")
        print("Create this file yourself with hostile words/phrases from your own training data.")
        return []

HOSTILE_KEYWORDS = _load_hostile_keywords()


def _keyword_safety_net_passes(text: str) -> bool:
    """True if the comment contains at least one known hostile term."""
    if not HOSTILE_KEYWORDS:
        return True  # if no keyword file provided, don't block on this check
    lowered = text.lower()
    return any(kw in lowered for kw in HOSTILE_KEYWORDS)


def _matched_keywords(text: str) -> list[str]:
    """Returns which hostile keywords were found in this comment, for the explain feature."""
    lowered = text.lower()
    return [kw for kw in HOSTILE_KEYWORDS if kw in lowered]


def _is_toxic_label(label: str) -> bool:
    return label in ("Toxic", "Hate Speech", "Offensive")


def ensemble_classify(text: str) -> dict:
    """
    Combines binary + 3-class predictions with confidence tiering and a
    keyword safety net, to reduce false positives from either model alone.

    Returns one of three tiers:
      - "Auto-Flag"     : both models agree it's toxic, high confidence, keyword check passes
      - "Needs Review"   : models disagree, OR confidence is low, OR keyword check fails
      - "No Action"      : both models agree it's Neutral, was pre-filtered, or matched a
                            known benign pattern (greeting, self-intro, etc.)
    """
    if is_benign_pattern(text):
        return {
            "tier": "No Action",
            "final_label": "Neutral",
            "binary": None,
            "threeclass": None,
            "recommended_action": "No action (matched known benign pattern)",
            "explanation": "Matches a common harmless phrase pattern (greeting, self-introduction, etc).",
            "matched_terms": [],
        }

    binary_result = _classify(text, binary_tokenizer, binary_model, BINARY_LABELS)
    threeclass_result = _classify(text, threeclass_tokenizer, threeclass_model, THREECLASS_LABELS)

    if binary_result.get("skipped"):
        return {
            "tier": "No Action",
            "final_label": "Neutral",
            "binary": binary_result,
            "threeclass": threeclass_result,
            "recommended_action": "No action (insufficient text to classify)",
            "explanation": "Comment too short or low-signal to classify reliably.",
            "matched_terms": [],
        }

    binary_says_toxic = _is_toxic_label(binary_result["label"])
    threeclass_says_toxic = _is_toxic_label(threeclass_result["label"])

    both_agree_toxic = binary_says_toxic and threeclass_says_toxic
    both_agree_neutral = (not binary_says_toxic) and (not threeclass_says_toxic)

    matched_terms = _matched_keywords(text)

    if both_agree_neutral:
        tier = "No Action"
        final_label = "Neutral"
        action = "No action"
        explanation = "Both models independently classified this as Neutral."

    elif both_agree_toxic:
        avg_confidence = (binary_result["confidence"] + threeclass_result["confidence"]) / 2
        keyword_ok = _keyword_safety_net_passes(text)

        if avg_confidence >= 0.95:
            # Very high confidence from both models - trust it, skip keyword requirement
            tier = "Auto-Flag"
            final_label = threeclass_result["label"]
            action = SEVERITY_ACTION.get(final_label, "Review needed")
            explanation = (
                f"Both models agree with very high confidence ({avg_confidence*100:.0f}%)."
                + (f" Matched term(s): {', '.join(matched_terms)}." if matched_terms else "")
            )
        elif avg_confidence >= CONFIDENCE_THRESHOLD and keyword_ok:
            tier = "Auto-Flag"
            final_label = threeclass_result["label"]
            action = SEVERITY_ACTION.get(final_label, "Review needed")
            explanation = f"Both models agree ({avg_confidence*100:.0f}%), matched term(s): {', '.join(matched_terms)}."
        else:
            tier = "Needs Review"
            final_label = threeclass_result["label"]
            action = "Needs human review (borderline confidence or no matched hostile terms)"
            explanation = (
                f"Models agree it may be {final_label} ({avg_confidence*100:.0f}% avg confidence), "
                "but no specific hostile term was matched - flagged on overall language pattern, "
                "which can be less reliable. Recommend human check."
            )

    else:
        # Models disagree with each other
        tier = "Needs Review"
        final_label = threeclass_result["label"]
        action = "Needs human review (models disagree)"
        explanation = (
            f"The two models disagree: fast check says '{binary_result['label']}', "
            f"detailed check says '{threeclass_result['label']}'. Recommend human check."
        )

    return {
        "tier": tier,
        "final_label": final_label,
        "binary": binary_result,
        "threeclass": threeclass_result,
        "recommended_action": action,
        "explanation": explanation,
        "matched_terms": matched_terms,
    }


@app.get("/")
def root():
    return {"status": "ok", "message": "Dual-model moderation API is running"}


@app.post("/predict/binary")
def predict_binary(req: CommentRequest):
    """Fast binary check - used by the browser extension."""
    return _classify(req.text, binary_tokenizer, binary_model, BINARY_LABELS)


@app.post("/predict/3class")
def predict_3class(req: CommentRequest):
    """Detailed 3-class check - used by the moderator dashboard."""
    return _classify(req.text, threeclass_tokenizer, threeclass_model, THREECLASS_LABELS)


@app.post("/predict/binary/batch")
def predict_binary_batch(req: BatchRequest):
    return {"results": [_classify(t, binary_tokenizer, binary_model, BINARY_LABELS) for t in req.texts]}


@app.post("/predict/3class/batch")
def predict_3class_batch(req: BatchRequest):
    return {"results": [_classify(t, threeclass_tokenizer, threeclass_model, THREECLASS_LABELS) for t in req.texts]}


@app.post("/predict/ensemble")
def predict_ensemble(req: CommentRequest):
    """Combined, mitigated prediction - recommended for both extension and dashboard."""
    return ensemble_classify(req.text)


@app.post("/predict/ensemble/batch")
def predict_ensemble_batch(req: BatchRequest):
    return {"results": [ensemble_classify(t) for t in req.texts]}


# ============================================
# Dashboard endpoint - fetches YouTube comments and classifies with 3-class model
# ============================================
import os
from googleapiclient.discovery import build

YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "AIzaSyDt_aSsTSFf6wVjw7BX05edgdmsyeI1Euk")


class VideoScanRequest(BaseModel):
    video_id: str
    max_comments: int = 100


@app.post("/scan_video")
def scan_video(req: VideoScanRequest):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    comments = []
    next_page_token = None

    try:
        while len(comments) < req.max_comments:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=req.video_id,
                maxResults=min(100, req.max_comments - len(comments)),
                pageToken=next_page_token,
                textFormat="plainText",
            ).execute()

            for item in response.get("items", []):
                text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(text)

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
    except Exception as e:
        return {"error": str(e), "comments": []}

    # Safety: remove exact duplicate comments (e.g. re-posted spam, or any
    # pagination edge case) before classifying, so counts stay accurate.
    seen = set()
    deduped_comments = []
    for c in comments:
        if c not in seen:
            seen.add(c)
            deduped_comments.append(c)
    comments = deduped_comments

    scan_start = time.time()
    results = []
    for text in comments:
        classification = ensemble_classify(text)
        results.append({"comment": text, **classification})
    scan_duration_seconds = round(time.time() - scan_start, 2)

    summary = {"Auto-Flag": 0, "Needs Review": 0, "No Action": 0}
    for r in results:
        summary[r["tier"]] = summary.get(r["tier"], 0) + 1

    # "Time saved" estimate: comments marked No Action don't need any human
    # attention at all. We assume ~15 seconds per comment for a human to read
    # and judge manually (a conservative, disclosed assumption - adjust as needed).
    SECONDS_PER_COMMENT_MANUAL_REVIEW = 15
    comments_filtered_out = summary.get("No Action", 0)
    time_saved_seconds = comments_filtered_out * SECONDS_PER_COMMENT_MANUAL_REVIEW
    percent_filtered = round((comments_filtered_out / len(results)) * 100, 1) if results else 0

    return {
        "total_comments": len(results),
        "summary": summary,
        "results": results,
        "scan_duration_seconds": scan_duration_seconds,
        "percent_filtered": percent_filtered,
        "time_saved_minutes": round(time_saved_seconds / 60, 1),
        # Every comment shown here was fetched live from YouTube's own API,
        # meaning YouTube has NOT removed it - this is proof-by-construction
        # that these flagged comments are still publicly visible.
        "flagged_still_live_count": summary.get("Auto-Flag", 0) + summary.get("Needs Review", 0),
    }