"""Stage the archived notebooks with their original model and relative input paths."""

from datetime import datetime
from pathlib import Path
from shutil import copy2, copytree
from tempfile import mkdtemp


def main():
    root = Path(__file__).resolve().parent
    original = root / "revision" / "original"
    runs = root / "runs"
    runs.mkdir(exist_ok=True)
    replay = runs / "original_replay"
    try:
        replay.mkdir()
    except FileExistsError:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        replay = Path(mkdtemp(prefix=f"original-{stamp}-", dir=runs))
    for name in (
        "01_single_ev_station_experiments.ipynb",
        "02_fleet_capacity_tightness_experiment.ipynb",
        "ev_robust_charging.py",
    ):
        copy2(original / name, replay / name)
    copy2(root / "caltech.csv", replay / "caltech.csv")
    for folder in ("data", "outputs"):
        copytree(root / folder, replay / folder)
    print(replay)


if __name__ == "__main__":
    main()
