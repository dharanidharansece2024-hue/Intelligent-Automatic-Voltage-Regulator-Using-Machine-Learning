"""Run a sample changing-voltage AVR simulation and save results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from avr_controller import IntelligentAVR, TARGET_VOLTAGE

ROOT = Path(__file__).resolve().parents[1]

def main():
    controller = IntelligentAVR()
    steps = 100
    t = np.arange(steps)
    # A changing supply voltage plus a small ripple.
    input_voltage = 220 + 22 * np.sin(2 * np.pi * t / 35) + 3 * np.sin(2 * np.pi * t / 9)
    load_current = 4.5 + 2.0 * np.sin(2 * np.pi * t / 50)
    temperature = 32 + 4 * np.sin(2 * np.pi * t / 70)

    rows = []
    for i in range(steps):
        rows.append(controller.regulate(
            float(input_voltage[i]), float(load_current[i]), float(temperature[i])
        ))
    df = pd.DataFrame(rows)
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    df.to_csv(results_dir / "simulation_results.csv", index=False)

    mean_abs_error = float(df["voltage_error_v"].abs().mean())
    summary = {
        "samples": steps,
        "target_voltage_v": TARGET_VOLTAGE,
        "average_absolute_output_error_v": round(mean_abs_error, 4),
        "note": "Software simulation only; not a real AVR hardware test."
    }
    with open(results_dir / "simulation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(10, 5))
    plt.plot(t, input_voltage, label="Input voltage")
    plt.plot(t, df["output_voltage_v"], label="Simulated AVR output")
    plt.axhline(TARGET_VOLTAGE, linestyle="--", label="230 V target")
    plt.xlabel("Simulation step")
    plt.ylabel("Voltage (V)")
    plt.title("Intelligent AVR: simulated voltage regulation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "voltage_response.png", dpi=160)
    plt.close()

    print("Simulation completed.")
    print(f"Average absolute output error: {mean_abs_error:.3f} V")
    print(f"Results saved in: {results_dir}")

if __name__ == "__main__":
    main()
