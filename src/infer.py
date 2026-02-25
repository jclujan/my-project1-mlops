"""
Module: Inference
-----------------
Role: Make predictions on new, unseen data.
Input: Trained Model + New Data.
Output: Predictions (Array or DataFrame).
"""

import pandas as pd
import joblib

from src.load_data import load
from src.clean_data import clean
from src.validate import validate

def run(input_path, model_path, output_path):
    df = load(input_path)
    df = clean(df)
    validate(df)

    model = joblib.load(model_path)     # pipeline
    preds = model.predict(df)

    out = pd.DataFrame({"Id": df["Id"], "SalePrice": preds})
    out.to_csv(output_path, index=False)

if __name__ == "__main__":
    run("data/raw/test.csv", "models/model.joblib", "reports/predictions.csv")