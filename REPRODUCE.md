# Reproduce the scheduling experiments

Run these commands from the repository root with Python 3.13. Install the Python dependencies first.

```text
python -m pip install -r requirements.txt
```

## Original single-EV checks

Both controls use the included OPSD profiles. The first uses the licensed IBM CPLEX Python API, and the second uses CVXPY with OSQP. Each command writes new results under `runs/`.

```text
python review_guide/checks/original_table6_cplex.py --project-root . --output-dir runs/check_table6
python review_guide/checks/matched_calibration.py . --output-dir runs/check_table9
```

Compare the first result with [the 12 CPLEX pairs](review_guide/checks/original_table6_cplex_pairs.csv) and [the historical comparison](outputs/q1_binary_complementarity_summary.csv). The second result can be compared with [the fixed-schedule reference](review_guide/checks/matched_calibration_results.csv). It holds each policy's January 24 schedule fixed while testing 121 daily coefficient paths.

## Revised station experiments

For the revised optimization stages and numerical unit tests, install a licensed IBM ILOG CPLEX executable and set the environment variable `CPLEX_EXE` to its full path. The study used CPLEX 22.1.2.0. A PowerShell example is shown below.

```powershell
$env:CPLEX_EXE = 'C:\Program Files\IBM\ILOG\CPLEX_Studio2212\cplex\bin\x64_win64\cplex.exe'
```

```text
python -m pytest -q -p no:dash test_revision.py
python run_revision_experiments.py --stages audit historical sensitivity boundaries calibration correlation degradation exchangeability
```

The `calibration` stage solves each day's schedule anew. Compare its aggregate outputs with [the daily-rescheduling summary](revision/results/calibration_summary.csv). The [throughput summary](revision/results/degradation_summary.csv) corresponds to the separate degradation sensitivity stage.

## Fleet experiments

Obtain the Caltech session data and save the three input fields to `caltech.csv` at the repository root, following the [data instructions](data/README.md). The revised fleet experiment uses a common station disturbance and the timing stage tests several fleet sizes and time grids.

```text
python run_revision_experiments.py --stages fleet runtime
```

The reference aggregates are [fleet reliability](revision/results/fleet_summary.csv) and [runtime and memory](revision/results/runtime_summary.csv). Sampling depends on the local session pool and its ordering. Running times and memory use depend on the machine. For the original independent-coefficient fleet design, run `python prepare_original_replay.py` and execute `02_fleet_capacity_tightness_experiment.ipynb` from the directory it prints.

To generate all revised outputs in one run, provide the local session file, configure CPLEX, and run the following in order.

```text
python run_revision_experiments.py
python build_revision_artifacts.py
python validate_revision.py --results-only
```

The driver creates its detailed result files under `revision/results/`, fleet inputs under `revision/instances/` and a local solve cache under `revision/solves/`. The second command builds tables and figures from those outputs. The last command checks the resulting solver records and chronological split. A fresh checkout provides a clean timing run.
