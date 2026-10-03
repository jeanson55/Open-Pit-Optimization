import joblib
import pandas as pd
from pathlib import Path


MODEL = Path(
    "results/hybrid_v33_selector.pkl"
)

DATA = Path(
    "results/hybrid_training_data.csv"
)


def main():

    print("=" * 90)
    print("HYBRID V3.3 SELECTOR VALIDATION")
    print("=" * 90)


    bundle = joblib.load(MODEL)

    model = bundle["model"]
    dataset_encoder = bundle["dataset_encoder"]
    method_encoder = bundle["method_encoder"]
    features = bundle["features"]


    df = pd.read_csv(DATA)


    # recreate encoded features
    df["dataset_id"] = dataset_encoder.transform(
        df["dataset"]
    )

    df["method_id"] = method_encoder.transform(
        df["method"]
    )


    X = df[features]


    df["prediction"] = model.predict(X)


    for dataset in df["dataset"].unique():

        print("\n")
        print("-" * 90)
        print(dataset.upper())
        print("-" * 90)


        subset = df[
            df["dataset"] == dataset
        ]


        for _, row in subset.iterrows():

            print(
                f"{row['method']:12s}"
                f" actual={row['best_choice']} "
                f" predicted={row['prediction']} "
                f" NPV={row['final_npv']:,.0f}"
            )


    accuracy = (
        df["prediction"]
        ==
        df["best_choice"]
    ).mean()


    print("\n")
    print("=" * 90)
    print(
        f"Training-set accuracy: {accuracy:.2%}"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()