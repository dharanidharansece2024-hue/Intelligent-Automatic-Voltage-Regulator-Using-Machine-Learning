"""Train and evaluate the AVR compensation model."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["input_voltage_v", "load_current_a", "temperature_c"]
TARGET = "compensation_voltage_v"

def main():
    data_path = ROOT / "data" / "avr_training_data.csv"
    if not data_path.exists():
        raise FileNotFoundError("Dataset missing. Run: python src/generate_data.py")

    df = pd.read_csv(data_path)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=180, max_depth=16, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    model_dir = ROOT / "models"
    results_dir = ROOT / "results"
    model_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    joblib.dump(model, model_dir / "avr_model.joblib")

    metrics = {
        "algorithm": "RandomForestRegressor",
        "features": FEATURES,
        "target": TARGET,
        "test_rows": int(len(y_test)),
        "mae_volts": round(float(mae), 4),
        "rmse_volts": round(float(rmse), 4),
        "target_voltage_volts": 230.0,
        "note": "Synthetic educational dataset; not validated for real hardware."
    }
    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Model training completed.")
    print(f"Test MAE : {mae:.3f} V")
    print(f"Test RMSE: {rmse:.3f} V")
    print(f"Saved model: {model_dir / 'avr_model.joblib'}")

if __name__ == "__main__":
    main()
