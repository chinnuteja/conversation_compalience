# src/eval/eval_bfsi_synth.py

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Add repo root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.telugu_offense_inference import TeluguOffenseClassifier

DATA_PATH = Path("data/bfsi_synth/agent_utterances.csv")
THRESHOLD = 0.8   # stricter threshold for BFSI usage


def main():
    df = pd.read_csv(DATA_PATH)

    texts = df["text"].tolist()
    y_true = df["offensive_label"].tolist()

    clf = TeluguOffenseClassifier()
    probs = clf.predict_proba(texts)

    df["p_offensive"] = probs

    # try multiple thresholds
    for thr in [0.4, 0.5, 0.6, 0.7, 0.8]:
        y_pred = [1 if p >= thr else 0 for p in probs]

        print(f"\n=== Threshold {thr} ===")
        print(classification_report(y_true, y_pred, digits=3))
        print("Confusion matrix [ [TN FP], [FN TP] ]:")
        print(confusion_matrix(y_true, y_pred))

    # save scored file for manual inspection
    df.to_csv(DATA_PATH.with_name("agent_utterances_scored.csv"), index=False, encoding="utf-8")
    print("\nScored rows saved to:", DATA_PATH.with_name("agent_utterances_scored.csv"))


if __name__ == "__main__":
    main()
