"""Train the final-exam model from the UCI Student Performance dataset.

Run this once (``python scripts/train_model.py``). It downloads the data,
cleans it, trains the quantile models, evaluates them, and writes
``models/gb_quantile.joblib`` and ``models/metrics.json``.
"""

import json
import sys
from pathlib import Path

from ucimlrepo import fetch_ucirepo

# Make "import gradecast" work when running this file directly from scripts/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gradecast import model  # noqa: E402  (must come after the path tweak)

MODEL_PATH = PROJECT_ROOT / "models" / "gb_quantile.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"


def main() -> None:
    """Download, clean, train, evaluate and save the exam model."""
    # id=320 is the UCI "Student Performance" dataset (Portuguese subset, 649 rows).
    dataset = fetch_ucirepo(id=320)
    # Grades live in the "targets" part (G3) and "features" part (G1, G2);
    # "original" has everything together.
    raw = dataset.data.original
    print(f"Downloaded {len(raw)} student rows.")

    data = model.prepare_data(raw)
    print(f"{len(data)} rows left after removing dropped-out students.")

    trained = model.train(data)
    metrics = model.evaluate(trained["models"], trained["X_test"], trained["y_test"])
    metrics["n_train"] = trained["n_train"]
    metrics["n_rows_total"] = int(len(raw))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(trained["models"], MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"Train rows: {metrics['n_train']}, test rows: {metrics['n_test']}")
    for set_name in model.FEATURES:
        result = metrics[set_name]
        print(
            f"{set_name:16s} MAE {result['mae']:.2f} "
            f"(baseline {result['baseline_mae']:.2f}), "
            f"coverage {result['coverage']:.2f}"
        )
    print(f"Saved {MODEL_PATH} and {METRICS_PATH}")


if __name__ == "__main__":
    main()
