"""Safe, software-only AVR controller simulation."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["input_voltage_v", "load_current_a", "temperature_c"]
TARGET_VOLTAGE = 230.0
MAX_CORRECTION_V = 40.0

class IntelligentAVR:
    def __init__(self, model_path=None):
        path = Path(model_path) if model_path else ROOT / "models" / "avr_model.joblib"
        if not path.exists():
            raise FileNotFoundError("Trained model missing. Run: python src/train_model.py")
        self.model = joblib.load(path)

    def regulate(self, input_voltage_v, load_current_a, temperature_c):
        """Return a simulated regulated output and model correction."""
        values = [input_voltage_v, load_current_a, temperature_c]
        if not all(np.isfinite(v) for v in values):
            raise ValueError("All inputs must be finite numbers.")
        if not 150 <= input_voltage_v <= 280:
            raise ValueError("Input voltage must be between 150 V and 280 V in this simulation.")
        if not 0 <= load_current_a <= 15:
            raise ValueError("Load current must be between 0 A and 15 A.")
        if not 0 <= temperature_c <= 80:
            raise ValueError("Temperature must be between 0 °C and 80 °C.")

        features = pd.DataFrame([{
            "input_voltage_v": input_voltage_v,
            "load_current_a": load_current_a,
            "temperature_c": temperature_c,
        }], columns=FEATURES)
        predicted = float(self.model.predict(features)[0])
        correction = float(np.clip(predicted, -MAX_CORRECTION_V, MAX_CORRECTION_V))

        # Simplified simulated plant: correction is applied and load/temperature
        # introduce a small voltage drop. This is not a physical power-stage model.
        load_drop = 0.18 * load_current_a + 0.015 * (temperature_c - 25)
        output_voltage = input_voltage_v + correction - load_drop
        return {
            "input_voltage_v": float(input_voltage_v),
            "load_current_a": float(load_current_a),
            "temperature_c": float(temperature_c),
            "predicted_correction_v": predicted,
            "applied_correction_v": correction,
            "output_voltage_v": float(output_voltage),
            "voltage_error_v": float(output_voltage - TARGET_VOLTAGE),
        }
