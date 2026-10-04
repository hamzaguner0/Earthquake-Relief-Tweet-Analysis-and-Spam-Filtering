"""Reconstructed TF-IDF / logistic regression baseline; not the original lost code."""
import json
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from .data import validated_training_rows, feature_text, text_key

def train(frame, output, demo=False, seed=42):
    frame = validated_training_rows(frame, demo=demo)
    train_rows, test_rows = train_test_split(frame, test_size=0.2, random_state=seed, stratify=frame["label"])
    assert set(train_rows["_key"]).isdisjoint(test_rows["_key"])
    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(preprocessor=feature_text, ngram_range=(1, 2), min_df=1)),
            ("char", TfidfVectorizer(preprocessor=feature_text, analyzer="char_wb", ngram_range=(3, 5), min_df=1)),
        ])),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
    ])
    pipeline.fit(train_rows["text"], train_rows["label"])
    prediction = pipeline.predict(test_rows["text"])
    report = {
        "evaluation_source": "synthetic_smoke_test" if demo else "manual_labels_held_out",
        "train_rows": len(train_rows), "test_rows": len(test_rows), "seed": seed,
        "threshold": 0.5, "labels": {"0": "other", "1": "relief_request"},
        "classification_report": classification_report(test_rows["label"], prediction, output_dict=True, zero_division=0),
        "confusion_matrix_labels_0_1": confusion_matrix(test_rows["label"], prediction, labels=[0, 1]).tolist(),
        "limitation": "Synthetic scores are not real-world performance. Random holdout does not eliminate event/near-duplicate leakage; use grouped or time-based independent manual evaluation for stronger claims.",
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Models and normalized training fingerprints are private artifacts.
    bundle = {"pipeline": pipeline, "report": report, "train_keys": set(train_rows["_key"]), "test_keys": set(test_rows["_key"])}
    joblib.dump(bundle, output / "baseline.joblib")
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report

def load_model(path):
    # joblib/pickle can execute code: only load a locally created trusted file.
    return joblib.load(path)

def score(bundle, texts):
    pipeline = bundle["pipeline"]
    position = list(pipeline.classes_).index(1)
    return pipeline.predict_proba(texts)[:, position]

def pseudo_label(bundle, frame, threshold=0.8):
    if not 0.5 < threshold <= 1:
        raise ValueError("Pseudo-label confidence threshold must be greater than 0.5 and at most 1.")
    frame = frame.copy()
    if frame["text"].isna().any() or frame["text"].map(text_key).eq("").any():
        raise ValueError("Empty text must be reviewed.")
    known = bundle["train_keys"] | bundle["test_keys"]
    frame = frame.loc[~frame["text"].map(text_key).isin(known)].copy()
    frame = frame.drop_duplicates("text")
    if len(frame) == 0:
        return frame.assign(label=[], label_source=[], prob_1=[], needs_review=[])
    probabilities = score(bundle, frame["text"])
    frame["label"] = (probabilities >= 0.5).astype(int)
    frame["label_source"] = "pseudo"
    frame["prob_1"] = probabilities
    frame["needs_review"] = (probabilities < threshold) & (probabilities > 1 - threshold)
    return frame
