import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def normalise_labels(series):
    """
    Map all possible label values into:
      0 = NOT_OFFENSIVE / NOT_HATE
      1 = OFFENSIVE / HATE
    """
    print("Unique labels in this file:", series.unique())

    label_map = {
        # numeric
        0: 0,
        1: 1,

        # common strings
        "hate": 1,
        "HATE": 1,
        "Offensive": 1,
        "OFFENSIVE": 1,
        "offensive": 1,
        "HOF": 1,

        "not_hate": 0,
        "NOT_HATE": 0,
        "NOT HATE": 0,
        "Not Hate": 0,
        "non-hate": 0,
        "NONE": 0,
        "Normal": 0,
    }

    mapped = series.map(label_map)
    return mapped


def load_telugu_train_val():
    """
    telugu_train.csv + telugu_val.csv
    (your screenshot shows columns: 'label', 'text')
    """

    dfs = []
    for fname, src_name in [
        ("telugu_train.csv", "telugu_train"),
        ("telugu_val.csv", "telugu_val"),
    ]:
        path = RAW_DIR / fname
        df = pd.read_csv(path, encoding="utf-8")
        print(f"{fname} columns:", df.columns)

        # if your columns are different, change these two names
        TEXT_COL = "text"
        LABEL_COL = "label"

        labels = normalise_labels(df[LABEL_COL])
        df_out = pd.DataFrame(
            {
                "text": df[TEXT_COL].astype(str),
                "label": labels,
                "source": src_name,
            }
        ).dropna(subset=["label"])
        df_out["label"] = df_out["label"].astype(int)
        dfs.append(df_out)

    return pd.concat(dfs, ignore_index=True)


def load_training_telugu_hate():
    """
    training_data_telugu-hate.xlsx (Kaggle-style file)
    """
    path = RAW_DIR / "training_data_telugu-hate.xlsx"
    df = pd.read_excel(path, engine="openpyxl")
    print("training_data_telugu-hate.xlsx columns:", df.columns)

    # CHANGE these if your columns have different names
    # Observed columns in this file: 'S.No', 'Comments', 'Label'
    # Use 'Comments' for the text and 'Label' for the label
    TEXT_COL = "Comments"
    LABEL_COL = "Label"

    labels = normalise_labels(df[LABEL_COL])
    df_out = pd.DataFrame(
        {
            "text": df[TEXT_COL].astype(str),
            "label": labels,
            "source": "training_telugu_hate",
        }
    ).dropna(subset=["label"])
    df_out["label"] = df_out["label"].astype(int)
    return df_out


def load_telugu_english_hold():
    """
    telugu-english-test-data-with-labels.xlsx (HOLD-style file)
    """
    path = RAW_DIR / "telugu-english-test-data-with-labels.xlsx"
    df = pd.read_excel(path, engine="openpyxl")
    print("telugu-english-test-data-with-labels.xlsx columns:", df.columns)

    # Observed columns in this file: 'S.No', 'Comments', 'Label'
    TEXT_COL = "Comments"
    LABEL_COL = "Label"

    labels = normalise_labels(df[LABEL_COL])
    df_out = pd.DataFrame(
        {
            "text": df[TEXT_COL].astype(str),
            "label": labels,
            "source": "hold_test",
        }
    ).dropna(subset=["label"])
    df_out["label"] = df_out["label"].astype(int)
    return df_out


def main():
    dfs = []

    # CSVs (train/val)
    dfs.append(load_telugu_train_val())

    # Excel: Kaggle train
    try:
        dfs.append(load_training_telugu_hate())
    except FileNotFoundError:
        print("training_data_telugu-hate.xlsx not found, skipping.")

    # Excel: HOLD test with labels
    try:
        dfs.append(load_telugu_english_hold())
    except FileNotFoundError:
        print("telugu-english-test-data-with-labels.xlsx not found, skipping.")

    unified = pd.concat(dfs, ignore_index=True)

    # basic cleaning
    unified["text"] = unified["text"].astype(str).str.strip()
    unified = unified[unified["text"].str.len() > 0]

    print("Unified dataset shape:", unified.shape)
    print(unified["label"].value_counts())

    out_path = PROCESSED_DIR / "offense_unified.csv"
    unified.to_csv(out_path, index=False, encoding="utf-8")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
