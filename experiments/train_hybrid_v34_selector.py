import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier


DATA = "results/hybrid_v34_training_data.csv"

MODEL = "results/hybrid_v34_selector.pkl"



print("="*80)
print("HYBRID V3.4 ECONOMIC SELECTOR TRAINING")
print("="*80)



# Load dataset

df = pd.read_csv(DATA)


print(df)



# Features used by the ML selector

features = [

    "dataset_id",

    "method_id",

    "final_npv",

    "runtime",

    "violations",

    "economic_score",

    "npv_per_node",

    "nodes"

]



X = df[features]


y = df["best_choice"]



# Random Forest selector

model = RandomForestClassifier(

    n_estimators=300,

    random_state=42,

    class_weight="balanced",

    max_depth=None

)



model.fit(

    X,

    y

)



# Save model and feature order

joblib.dump(

    {

        "model": model,

        "features": features

    },

    MODEL

)



print()

print("Training complete")

print("Features used:")

for f in features:

    print(" -", f)


print()

print("Saved:")

print(MODEL)