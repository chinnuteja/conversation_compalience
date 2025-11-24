import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path("data/processed")

def main():
    df = pd.read_csv(PROCESSED_DIR / "offense_unified.csv")

    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 80% train, 10% val, 10% test (stratified so label balance stays similar)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=42,
    )

    train_df.to_csv(PROCESSED_DIR / "offense_train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "offense_val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "offense_test.csv", index=False)

    print("Train:", train_df.shape)
    print("Val  :", val_df.shape)
    print("Test :", test_df.shape)

if __name__ == "__main__":
    main()
