# Reliability-aware V2G scheduling

Code, data and results for comparing relaxed and physical electric-vehicle charging and discharging schedules under station charging-capacity, net-import and export limits.

[Download the full replication archive](SEGAN_Replication.zip?raw=true) · [Reproduction commands and result mapping](REPRODUCE.md)

The archive contains the editable sources available here, the 361 daily input profiles, the original independent-coefficient fleet experiment, both original-design checks and the native solver records for all 3,108 revised-study solves. To verify the archived results or regenerate publication figures, extract the ZIP into an empty directory and run the commands in its README. New optimizations use IBM ILOG CPLEX 22.1.2.0. The original fixed-schedule calibration check uses OSQP.

**Archive SHA-256**: 14714372e53c4dd0448691c7bad0e50a86000aa0dbcdacf8d45b2a3dcd5b2531

| Locate | Files |
| --- | --- |
| Revised model and experiments | [Model](revision_model.py), [experiments](run_revision_experiments.py), [unit tests](test_revision.py) |
| Rebuild figures and summary tables | [Plot and table generation](build_revision_artifacts.py) |
| Original solver and calibration checks | [12-pair CPLEX check](review_guide/checks/original_table6_cplex.py), [fixed-schedule calibration](review_guide/checks/matched_calibration.py) |
| Original single-EV and fleet studies | [Archived notebooks](revision/original/), [recorded tables](outputs/) |
| Revised raw results and summaries | [CSV and JSON results](revision/results/), [fleet instances](revision/instances/) |
| OPSD and ACN-Data | [Input sources, transformations and attribution](data/README.md) |
