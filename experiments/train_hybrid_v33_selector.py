import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib


DATA = Path(
    "results/hybrid_training_data.csv"
)

MODEL = Path(
    "results/hybrid_v33_selector.pkl"
)


def main():

    df = pd.read_csv(DATA)


    print("="*80)
    print("HYBRID V3.3 SELECTOR TRAINING")
    print("="*80)


    print(df)


    # Encode categorical columns

    dataset_encoder = LabelEncoder()

    method_encoder = LabelEncoder()


    df["dataset_id"] = dataset_encoder.fit_transform(
        df["dataset"]
    )

    df["method_id"] = method_encoder.fit_transform(
        df["method"]
    )


    features = [
        "dataset_id",
        "method_id",
        "total_runtime",
        "partition_runtime",
        "solver_runtime",
        "initial_violations",
    ]


    X = df[features]

    y = df["best_choice"]


    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )


    model.fit(
        X,
        y
    )


    MODEL.parent.mkdir(
        exist_ok=True
    )


    joblib.dump(
        {
            "model": model,
            "dataset_encoder": dataset_encoder,
            "method_encoder": method_encoder,
            "features": features,
        },
        MODEL
    )


    print()
    print("Training complete")
    print("Saved:")
    print(MODEL)



if __name__ == "__main__":
    main()