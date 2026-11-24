# Reliability-aware vehicle-to-grid scheduling

This repository accompanies *Reliability-Aware Vehicle-to-Grid Scheduling: Applicability and Limits of Exact Convex Relaxation*. The code compares relaxed and binary charging schedules under station charging, import and export limits. The station experiments run on the included OPSD profiles. Fleet experiments use a Caltech ACN-Data session export supplied by the reader.

| Task | Where to start |
| --- | --- |
| Install dependencies, run the station controls or reproduce fleet results | [Reproduction guide](REPRODUCE.md) |
| Obtain and format the inputs | [Data sources](data/README.md) |
| Inspect the station model and run experiments | [Model](revision_model.py) · [Experiment driver](run_revision_experiments.py) |
| Recheck the 12 original binary comparisons and the fixed-schedule calibration | [Original-design check scripts and reference outputs](review_guide/checks/) |
| Inspect the original single-EV and fleet experiments | [Notebooks and original model](revision/original/) |
| Compare aggregate results | [Daily calibration](revision/results/calibration_summary.csv) · [Fleet reliability](revision/results/fleet_summary.csv) · [Throughput costs](revision/results/degradation_summary.csv) · [Solver performance](revision/results/runtime_summary.csv) |

The repository contains the analysis code, OPSD profiles and selected reference summaries. Running the experiments locally generates the detailed results, schedules and solver logs.
