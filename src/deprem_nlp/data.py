"""Local validation; reports never include original text or identifiers."""
import re
import unicodedata
import pandas as pd

PATTERNS = {
    "email": r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b",
    "handle": r"(?<!\w)@[A-Za-z0-9_]+",
    "url": r"https?://\S+|www\.\S+",
    "phone": r"(?<!\d)(?:\+?90[\s.-]?)?0?5\d{2}[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}(?!\d)",
    "address_hint": r"\b(?:mahalle|mahallesi|mah\.|sokak|sokağı|sok\.|apartman|apartmanı|cadde|caddesi|blok|kat|daire)\b",
    "health_hint": r"\b(?:yaralı|hastası|hasta|diyabet|hamile|kanama|ilaç|engelli)\b",
    "child_hint": r"\b(?:çocuk|çocuğu|bebek|bebeği|yaşında)\b",
}

def text_key(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).replace("I", "ı").replace("İ", "i").casefold().split())

def feature_text(text):
    text = text_key(text)
    for name in ("url", "email", "phone", "handle"):
        text = re.sub(PATTERNS[name], f" {name}token ", text, flags=re.I)
    return text

def load_frame(path):
    frame = pd.read_csv(path)
    frame = frame.rename(columns={"Tweets": "text", "Tweet": "text", "Label": "label"})
    if "text" not in frame:
        raise ValueError("CSV must contain text (or Tweets) column.")
    return frame

def audit(frame):
    text = frame["text"].fillna("").astype(str)
    keys = text.map(text_key)
    conflicts = 0
    if "label" in frame:
        conflicts = int(pd.DataFrame({"key": keys, "label": frame["label"]}).groupby("key")["label"].nunique().gt(1).sum())
    return {
        "rows": len(frame), "empty_text": int(keys.eq("").sum()),
        "duplicate_rows_normalized": int(keys.duplicated().sum()),
        "conflicting_text_groups": conflicts,
        "label_source_counts": frame["label_source"].fillna("unknown").value_counts().to_dict() if "label_source" in frame else {"unknown": len(frame)},
        "privacy_pattern_rows": {name: int(text.str.contains(pattern, flags=re.I, regex=True).sum()) for name, pattern in PATTERNS.items()},
        "note": "Heuristic screening, not proof of anonymity, legality or label origin. No raw text is included.",
    }

def validated_training_rows(frame, demo=False):
    required = {"text", "label", "label_source"}
    if not required.issubset(frame.columns):
        raise ValueError("Training requires text, label, label_source. Mixed labels without provenance cannot be used as ground truth.")
    frame = frame.copy()
    if frame["text"].isna().any() or frame["text"].map(text_key).eq("").any():
        raise ValueError("Empty text must be reviewed before training.")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Labels must be exactly 0 or 1.")
    frame["label"] = labels.astype(int)
    expected = "synthetic" if demo else "manual"
    if not frame["label_source"].eq(expected).all():
        raise ValueError(f"This baseline accepts only {expected} labels; pseudo/unknown labels are excluded from evaluation.")
    frame["_key"] = frame["text"].map(text_key)
    if frame.groupby("_key")["label"].nunique().gt(1).any():
        raise ValueError("Conflicting duplicate labels require manual review; no arbitrary label is selected.")
    frame = frame.drop_duplicates("_key").reset_index(drop=True)
    if set(frame["label"]) != {0, 1} or frame["label"].value_counts().min() < 5:
        raise ValueError("At least five unique examples per class are required for a stratified holdout.")
    return frame
