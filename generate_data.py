"""Create a synthetic dataset for the educational AVR simulation."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

def main():
    rng = np.random.default_rng(42)
    n = 5000
    input_voltage = rng.uniform(180, 260, n)
    load_current = rng.uniform(0.2, 10.0, n)
    temperature = rng.uniform(20, 60, n)

    # Educational plant approximation: load causes a small voltage drop.
    load_drop = 0.18 * load_current + 0.015 * (temperature - 25)
    target_compensation = 230.0 - input_voltage + load_drop
    # Add small sensor/label variation so the model learns a realistic mapping.
    target_compensation += rng.normal(0, 0.12, n)

    df = pd.DataFrame({
        "input_voltage_v": input_voltage.round(3),
        "load_current_a": load_current.round(3),
        "temperature_c": temperature.round(3),
        "compensation_voltage_v": target_compensation.round(3),
    })
    path = DATA_DIR / "avr_training_data.csv"
    df.to_csv(path, index=False)
    print(f"Created {len(df)} training rows: {path}")

if __name__ == "__main__":
    main()
