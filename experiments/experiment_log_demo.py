from __future__ import annotations

import math
import time

from cs336_basics.experiment_logging import ExperimentLogger, make_run_dir


def main() -> None:
    run_dir = make_run_dir("outputs/runs", name="experiment_log_demo")
    logger = ExperimentLogger(run_dir)
    logger.save_run_config({"task": "experiment_log_demo", "steps": 50})

    for step in range(50):
        # Fake a decaying loss curve
        loss = 5.0 * math.exp(-step / 20)
        logger.log(step=step, split="train", loss=loss)
        time.sleep(0.01)

    logger.note("Demo finished.")
    print(f"wrote: {run_dir}")


if __name__ == "__main__":
    main()
