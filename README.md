# Reliability-aware V2G scheduling

This project compares relaxed and binary schedules for electric-vehicle charging and discharging. All optimization models use IBM ILOG CPLEX.

## Run

Use Python 3.13 and a licensed copy of CPLEX Studio 22.1.2. Activate your Python environment, then configure the executable and Python runtime. These PowerShell commands assume the standard Windows installation path. [IBM setup guide](https://www.ibm.com/docs/en/icos/22.1.2?topic=rnco2-cplex-python-api-now-installed-using-pip-conda)

```powershell
$studio = 'C:\Program Files\IBM\ILOG\CPLEX_Studio2212'
$env:CPLEX_EXE = Join-Path $studio 'cplex\bin\x64_win64\cplex.exe'
python -m pip install -r requirements.txt
docplex config --upgrade "$studio"
```

Run the Table 6 and Table 9 checks with the included OPSD station data.

```sh
python checks/original_table6_cplex.py
python checks/matched_calibration.py
```

For the complete study, place your Caltech ACN-Data export at `caltech.csv` in the project root. The [data instructions](data/README.md) describe the required columns.

```sh
python run_revision_experiments.py
python build_revision_artifacts.py
python validate_revision.py
```

For selected stages, use e.g. `python run_revision_experiments.py --stages audit historical calibration`. The checks and experiments write to [results/](results/). Figures go to `results/figures/` and solver records to `runs/solves/`. The included runtime summary records the original CPLEX timings. Rerunning the benchmark measures your machine.

The [original notebooks](notebooks/) use CVXPY to express the models and CPLEX to solve them. Run `python prepare_original_replay.py` and open the notebooks in the directory it prints. The [fleet notebook results](results/table8_capacity_tightness_frontier.csv) summarize the 20 seeds for each policy and capacity. Run the numerical tests with `python -m pytest -q test_revision.py`.
