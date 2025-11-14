#!/usr/bin/env python3
import itertools
import json
import re
import subprocess
from pathlib import Path

import yaml

BASE_EXP = "OptimizedParameters"           # yamls dans Todo_List/
RESULTS_DIR = Path("Results")
GRID = {
    0: [0.8, 1.0],                         # classe 0
    1: [3.5, 4.0, 4.5],                    # classe 1
    2: [3.0, 3.5, 4.0],                    # classe 2
    3: [1.5, 2.0, 2.5],                    # classe 3
    4: [2.0, 2.3, 2.6],                    # classe 4
}
val_regex = re.compile(r"Validation Loss at epoch \d+: ([0-9.]+)")
iou_regex = re.compile(r"Mean IoU: ([0-9.]+)")

def run_once(weights, run_name):
    cfg_path = Path(f"Todo_List/{run_name}.yaml")
    base_cfg = yaml.safe_load(Path(f"Todo_List/{BASE_EXP}.yaml").read_text())
    base_cfg.setdefault("TRAINING", {})["CLASS_WEIGHTS"] = weights
    cfg_path.write_text(yaml.safe_dump(base_cfg))

    proc = subprocess.run(
        ["python", "main.py", "-exp", run_name],
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = proc.stdout
    val_losses = [float(m.group(1)) for m in val_regex.finditer(stdout)]
    mean_iou = iou_regex.search(stdout)
    metrics = {
        "best_val_loss": min(val_losses) if val_losses else None,
        "mean_iou": float(mean_iou.group(1)) if mean_iou else None,
    }
    log_dir = RESULTS_DIR / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    (log_dir / "training_log.txt").write_text(stdout)

    (RESULTS_DIR / run_name / "training_log.txt").write_text(stdout)
    return metrics

def main():
    results = []
    for combo in itertools.product(*GRID.values()):
        run_name = f"OptWeights_{'_'.join(map(lambda x: str(x).replace('.', 'p'), combo))}"
        weights = list(combo)
        metrics = run_once(weights, run_name)
        results.append({"run": run_name, "weights": weights, **metrics})
        print(f"{run_name}: {metrics}")
    Path("weight_sweep_results.json").write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
