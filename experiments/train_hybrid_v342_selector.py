import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier


DATA = "results/hybrid_v342_training_data.csv"

MODEL = "results/hybrid_v342_selector.pkl"



df = pd.read_csv(DATA)



features = [

    "dataset_id",

    "method_id",

    "runtime",

    "violations",

    "economic_score"

]


X = df[features]

y = df["best_choice"]



model = RandomForestClassifier(

    n_estimators=300,

    random_state=42,

    class_weight="balanced"

)



model.fit(X,y)



joblib.dump(

    {
        "model":model,
        "features":features
    },

    MODEL

)



print("HYBRID V3.4.2 TRAINING COMPLETE")
print("Saved:",MODEL)