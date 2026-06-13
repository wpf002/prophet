"""Smoke test for the neural model ladder.

NeuralForecast/torch segfaults when it shares a process with the LightGBM and
StatsForecast tests (conflicting OpenMP runtimes + fork state). So this test runs
the real train/predict path in a clean subprocess and asserts on its output —
isolation that keeps the rest of the suite stable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

_CHILD = """
import json
from prophet.data.loaders import load_synthetic
from prophet.evaluation.cross_validation import split_train_test
from prophet.models.neural import MODEL_NAMES, forecast_neural

df = load_synthetic(n_series=2, n_obs=300, frequency="1h", seed=42)
train, _ = split_train_test(df, horizon=12)
fc, info = forecast_neural(
    train, horizon=12, season_length=24, freq="h",
    input_size=48, max_steps=2, accelerator="cpu",
)
print("RESULT" + json.dumps({
    "columns": fc.columns,
    "height": fc.height,
    "ds_dtype": str(fc.schema["ds"]),
    "trained": list(info.timings),
    "devices": {k: v.device for k, v in info.timings.items()},
    "model_names": MODEL_NAMES,
}))
"""


def test_neural_trains_and_returns_forecasts() -> None:
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, f"child failed:\n{proc.stderr[-2000:]}"
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT"))
    result = json.loads(line[len("RESULT") :])

    trained = set(result["trained"])
    assert trained, "no neural model trained"
    assert trained.issubset(set(result["model_names"]))
    assert trained.issubset(set(result["columns"]))
    assert result["height"] == 2 * 12
    assert result["ds_dtype"] == "Datetime(time_unit='us', time_zone='UTC')"
    assert all(dev == "cpu" for dev in result["devices"].values())
