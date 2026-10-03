"""
Hybrid V3.5 Selector Training
"""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier


DATA = Path(
    "results/hybrid_v35_training_data.csv"
)

MODEL = Path(
    "results/hybrid_v35_selector.pkl"
)


def main():

    print("="*80)
    print("HYBRID V3.5 SELECTOR TRAINING")
    print("="*80)


    df = pd.read_csv(DATA)


    # Only train Spatial vs METIS decisions
    df = df[
        df["method"].isin(
            [
                "Spatial",
                "METIS"
            ]
        )
    ]


    method_map = {
        "Spatial":0,
        "METIS":1
    }


    df["method_id"] = (
        df["method"]
        .map(method_map)
    )


    features = [

        "dataset_id",

        "method_id",

        "runtime",

        "violations",

        "npv_per_node",

        "economic_score"

    ]


    X = df[features]

    y = df["best_choice"]



    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )


    model.fit(
        X,
        y
    )


    bundle = {

        "model":model,

        "features":features

    }


    joblib.dump(
        bundle,
        MODEL
    )


    print(df)

    print()

    print("Training complete")

    print("Features:")
    for f in features:
        print("-",f)

    print()

    print("Saved:")
    print(MODEL)



if __name__=="__main__":
    main()