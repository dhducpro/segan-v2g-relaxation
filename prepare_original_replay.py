"""Prepare a separate working directory for the original single-EV and fleet notebooks."""

from pathlib import Path
from shutil import copy2, copytree, rmtree
from tempfile import mkdtemp


def main():
    root = Path(__file__).resolve().parent
    notebooks = root / "notebooks"
    runs = root / "runs"
    runs.mkdir(exist_ok=True)
    replay = runs / "original_replay"
    try:
        replay.mkdir()
    except FileExistsError:
        replay = Path(mkdtemp(prefix="original_replay_", dir=runs))

    try:
        for name in (
            "01_single_ev_station_experiments.ipynb",
            "02_fleet_capacity_tightness_experiment.ipynb",
            "ev_robust_charging.py",
        ):
            copy2(notebooks / name, replay / name)
        copytree(root / "data", replay / "data")

        sessions = root / "caltech.csv"
        if sessions.is_file():
            copy2(sessions, replay / sessions.name)
    except Exception:
        rmtree(replay)
        raise
    print(replay)


if __name__ == "__main__":
    main()
