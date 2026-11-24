# SEGAN replication package

This package contains the data, models and outputs for the station scheduling experiments. Numbers assigned to tables in the original submission differ from those in the revision. The entries below identify each experiment by design.

## Install and run

Use Python 3.13 and install the packages with `python -m pip install -r requirements.txt`. For fresh revised-study solves, install a licensed IBM ILOG CPLEX 22.1.2.0 executable and set `CPLEX_EXE` to its full path. Its Python API writes native models and directly solves the original 12-pair control. The original fixed-schedule calibration check uses OSQP. The metadata in `review_guide/checks/` identifies each original-control solver.

Run these commands from the extracted directory.

    python validate_revision.py --results-only
    python -m pytest -q -p no:dash test_revision.py
    python build_revision_artifacts.py
    python review_guide/checks/original_table6_cplex.py --project-root . --output-dir runs/check_table6
    python review_guide/checks/matched_calibration.py . --output-dir runs/check_table9

The first command recomputes feasibility and objective values from 3,108 recorded solves and checks the chronological split. The third rebuilds the revised-study table fragments and figures from the recorded results. Compare the freshly produced CSVs in the two runs directories with the corresponding CSVs under `review_guide/checks/`. The imported versions in the calibration run were CVXPY 1.9.1 and OSQP 1.0.5. Metadata for the installed CVXPY distribution reported 1.7.4 on that machine.

To solve from scratch, extract a second copy of the archive and rename its `revision/solves` folder before the following commands. Wall times, memory readings and small floating-point differences can vary. To run a subset, append e.g. `--stages historical runtime` to the first command.

    python run_revision_experiments.py
    python build_revision_artifacts.py
    python validate_revision.py --results-only

To rerun the archived independent-coefficient fleet experiment, run `python prepare_original_replay.py`. Open the second notebook in the printed directory and select **Run All** with the project Python environment in Jupyter. The first notebook contains the archived single-EV experiments. The original CVXPY implementation selects a supported installed solver by fallback. The dedicated 12-pair control above fixes the backend for every pair.

## Results and source mapping

| Experiment | Source in this archive |
| --- | --- |
| Unclamped historical prices, tariff and throughput stress | `revision/results/price_audit.json`, `historical.csv`, `sensitivity.csv` |
| PV and export limits | `revision/results/boundaries.csv` |
| Original clamped-price single-EV control, 12 pairs | `review_guide/checks/original_table6_cplex.py`, CSV and JSON in the same directory, plus `outputs/q1_binary_complementarity_summary.csv` |
| Original base-day market charge, throughput and smoothing | `outputs/summary.csv` |
| Dependence and single-EV risk | `revision/results/correlation.csv` |
| Fixed January 24 schedule, all-year ranks and training-only correction | `review_guide/checks/matched_calibration.py`, CSV and JSON in the same directory; the 361 daily profiles in `data/` |
| Daily rescheduling on the raw-load proxy, fixed and rolling calibration | `revision/results/calibration.csv`, `calibration_summary.csv`, `calibration_split.json`, `exchangeable_control.json` |
| Original independent-coefficient fleets, 20 samples, five margins | `revision/original/02_fleet_capacity_tightness_experiment.ipynb`, `outputs/fleet_capacity_tightness_raw.csv` |
| Shared station disturbance and temporal correlation, four margins | `revision/results/fleet_reliability.csv`, `fleet_summary.csv` |
| Throughput and matched runtime/memory controls | `revision/results/degradation.csv`, `degradation_summary.csv`, `runtime.csv`, `runtime_summary.csv` |
| Model inputs, solve records, solver logs and schedules | `revision/instances/`, `revision/solves/<key>/` |
| Revised-study tables and figures | `revision/tables/`, `revision/figures/` |

The compact Caltech export supplies session times and delivered energy. Vehicle and battery assumptions enter through the models. Data attribution and preprocessing are described in `data/README.md`. `revision/results/run_manifest.json` gives the original experiment environment and core hashes. `PACKAGE_MANIFEST.json` records the SHA-256 digest of each other file in the ZIP.
