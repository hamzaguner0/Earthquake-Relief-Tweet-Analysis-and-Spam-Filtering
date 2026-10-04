import argparse, json
from pathlib import Path
from .data import load_frame, audit
from .model import train, load_model, score, pseudo_label

def main():
    parser = argparse.ArgumentParser(description="Local Turkish NLP research; raw data stays local.")
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit")
    a.add_argument("csv")
    a.add_argument("--output", default="artifacts/audit.json")
    t = sub.add_parser("train")
    t.add_argument("csv")
    t.add_argument("--output", default="artifacts")
    t.add_argument("--demo", action="store_true", help="Accept synthetic labels only.")
    t.add_argument("--authorized-data", action="store_true", help="Assert collection, training and processing rights were verified.")
    p = sub.add_parser("predict")
    p.add_argument("text")
    p.add_argument("--model", default="artifacts/baseline.joblib")
    p.add_argument("--threshold", type=float, default=0.5)
    s = sub.add_parser("pseudo-label")
    s.add_argument("csv")
    s.add_argument("--model", default="artifacts/baseline.joblib")
    s.add_argument("--output", default="private/pseudo_labeled.csv")
    s.add_argument("--confidence", type=float, default=0.8)
    s.add_argument("--authorized-data", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "audit":
            report = audit(load_frame(args.csv))
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print("Aggregate audit saved; no text was exported.")
        elif args.command == "train":
            if not args.demo and not args.authorized_data:
                parser.error("Real data requires verified rights (--authorized-data) and manual label provenance.")
            report = train(load_frame(args.csv), args.output, demo=args.demo)
            print(json.dumps({k: report[k] for k in ("evaluation_source", "train_rows", "test_rows")}, ensure_ascii=False))
        elif args.command == "predict":
            if not args.text.strip() or not 0 <= args.threshold <= 1:
                parser.error("Text must be nonempty and threshold must be within 0..1.")
            probability = float(score(load_model(args.model), [args.text])[0])
            print(json.dumps({"label": int(probability >= args.threshold), "model_score_1": round(probability, 4), "threshold": args.threshold}))
        else:
            if not args.authorized_data:
                parser.error("Pseudo-labeling real data requires verified rights (--authorized-data).")
            result = pseudo_label(load_model(args.model), load_frame(args.csv), args.confidence)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(args.output, index=False)
            print(f"Saved {len(result)} rows locally, explicitly marked pseudo; excluded training and test texts.")
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")

if __name__ == "__main__":
    main()
