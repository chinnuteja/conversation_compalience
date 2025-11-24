# src/models/telugu_offense_inference.py

from pathlib import Path
from typing import List, Union

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

MODEL_DIR = Path("models/telugu_offense_v1_small")  # where Trainer saved it

class TeluguOffenseClassifier:
    def __init__(self, model_dir: Union[str, Path] = MODEL_DIR, device: str = None):
        model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, texts: List[str]) -> List[float]:
        """Return P(offensive) for each text."""
        if isinstance(texts, str):
            texts = [texts]

        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=32,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**enc).logits
        probs = F.softmax(logits, dim=-1)[:, 1]  # index 1 = offensive
        return probs.cpu().tolist()

    @torch.no_grad()
    def predict_label(self, texts: List[str], threshold: float = 0.45) -> List[int]:
        """Return 1 = offensive, 0 = not offensive, using a threshold."""
        probs = self.predict_proba(texts)
        return [1 if p >= threshold else 0 for p in probs]


# Example usage (you can test in a small script / REPL):
# from src.models.telugu_offense_inference import TeluguOffenseClassifier
#
# clf = TeluguOffenseClassifier()
#
# texts = [
#     "inka ila maatladithe complaint chestha",  # example
#     "sir emi problem ledu, emi tension padakandi",
# ]
#
# probs = clf.predict_proba(texts)
# labels = clf.predict_label(texts, threshold=0.6)
#
# for t, p, l in zip(texts, probs, labels):
#     print(t)
#     print("  p_offensive =", round(p, 3), "label =", l)
