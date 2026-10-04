import tempfile, unittest
from pathlib import Path
import pandas as pd
from deprem_nlp.data import validated_training_rows, audit, text_key
from deprem_nlp.model import train, load_model, pseudo_label

ROOT = Path(__file__).resolve().parents[1]

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.read_csv(ROOT / "examples/synthetic_demo.csv")

    def test_unknown_or_pseudo_labels_not_ground_truth(self):
        with self.assertRaises(ValueError):
            validated_training_rows(self.frame.drop(columns="label_source"))
        self.frame["label_source"] = "pseudo"
        with self.assertRaises(ValueError):
            validated_training_rows(self.frame)

    def test_conflicting_duplicate_is_rejected(self):
        duplicate = self.frame.iloc[[0]].copy()
        duplicate["label"] = 1 - duplicate["label"]
        with self.assertRaises(ValueError):
            validated_training_rows(pd.concat([self.frame, duplicate]), demo=True)

    def test_duplicates_cannot_cross_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            train(pd.concat([self.frame, self.frame.iloc[[0]]]), tmp, demo=True)
            bundle = load_model(Path(tmp) / "baseline.joblib")
            self.assertTrue(bundle["train_keys"].isdisjoint(bundle["test_keys"]))
            known = next(iter(bundle["test_keys"]))
            candidate = pd.DataFrame({"text": [known, "tamamen sentetik yeni bir deneme"]})
            labeled = pseudo_label(bundle, candidate)
            self.assertEqual(len(labeled), 1)
            self.assertTrue(labeled["label_source"].eq("pseudo").all())

    def test_audit_contains_counts_only(self):
        text = "Sentetik: @demo_user demo@example.com 0555 000 11 22"
        report = audit(pd.DataFrame({"text": [text]}))
        self.assertEqual(report["privacy_pattern_rows"]["email"], 1)
        self.assertEqual(report["privacy_pattern_rows"]["phone"], 1)
        self.assertNotIn(text, str(report))

    def test_turkish_and_whitespace_duplicate_key(self):
        self.assertEqual(text_key("  İSTANBUL\nçağrı "), text_key("istanbul çağrı"))

if __name__ == "__main__":
    unittest.main()
