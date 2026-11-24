from __future__ import annotations

import platform
import sys
import time
import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

import cvxpy as cp
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class EVData:
    name: str
    profile_date: str
    T: int
    L: np.ndarray
    availability: np.ndarray
    price_buy: np.ndarray
    price_sell: np.ndarray
    u_nominal: np.ndarray
    X: float
    Y: float
    M: float
    E0: float
    Emin: float
    Emax: float
    Etarget: float
    eta_c: float
    eta_d: float
    alpha: float
    rho: float


@dataclass(frozen=True)
class FleetData:
    name: str
    profile_date: str
    N: int
    T: int
    L: np.ndarray
    availability: np.ndarray
    price_buy: np.ndarray
    price_sell: np.ndarray
    u_nominal: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    M: float
    E0: np.ndarray
    Emin: np.ndarray
    Emax: np.ndarray
    Etarget: np.ndarray
    eta_c: np.ndarray
    eta_d: np.ndarray
    alpha: float
    rho: np.ndarray
    source: str = "synthetic"
    acn_site: str = ""
    acn_cache: str = ""
    horizon_date: str = ""
    source_sessions: int = 0


@dataclass(frozen=True)
class FleetExperimentConfig:
    use_acn_sessions: bool = True
    acn_site: str = "caltech"
    acn_start: str = "2019-01-01"
    acn_end: str = "2019-03-31"
    acn_min_kwh: float = 2.0
    acn_cache: str | None = None
    slot_minutes: int = 60


@dataclass(frozen=True)
class Policy:
    name: str
    label: str
    uncertainty_quantile: float | None = None
    chance_epsilon: float | None = None


POLICIES = [
    Policy("deterministic", "Deterministic", 0.50),
    Policy("quantile_75", "Slotwise Q75", uncertainty_quantile=0.75),
    Policy("robust", "Robust", 1.00),
]

BASELINE_POLICIES: Final = tuple(POLICIES)
GAMMA_LEVELS: Final = (0.0, 1.0, 2.0, 4.0, 8.0)
EPSILON_LEVELS: Final = (0.10, 0.05, 0.025, 0.01)
FLEET_COUNTS: Final = (10, 20, 30)
FLEET_GRID_HEADROOM_PER_VEHICLE: Final = 3.0
FLEET_FIXED_GRID_LIMIT: Final = 560.0
FLEET_SEEDS: Final = (
    20260506,
    20260507,
    20260508,
    20260509,
    20260510,
    20260511,
    20260512,
    20260513,
    20260514,
    20260515,
    20260516,
    20260517,
    20260518,
    20260519,
    20260520,
    20260521,
    20260522,
    20260523,
    20260524,
    20260525,
)
SENSITIVITY_DELTAS: Final = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
SENSITIVITY_ALPHA: Final = (0.0, 1.0e-4, 2.0e-4, 4.0e-4, 8.0e-4)
SENSITIVITY_RHO: Final = (0.0, 5.0e-4, 1.0e-3, 2.0e-3, 4.0e-3)
ABLATION_EFFICIENCY_PAIRS: Final = ((0.90, 0.85), (0.95, 0.90), (0.98, 0.95), (1.0, 1.0))
ABLATION_PRICE_SPREADS: Final = (0.90, 0.95, 0.98, 1.00, 1.02)
ABLATION_CAPACITY_LEVELS: Final = (465.0, 470.0, 475.0, 480.0, 490.0)
ABLATION_TERMINAL_TARGETS: Final = (55.0, 60.0, 65.0, 70.0)

OPSD_URL: Final = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
OPSD_MONTH_CACHE: Final = Path("data/opsd_de_lu_2019_01.csv")
OPSD_YEAR_CACHE: Final = Path("data/opsd_de_lu_2019.csv")
OPSD_PROFILE_CACHE: Final = Path("data/opsd_de_lu_2019_01_24_profile.csv")
OPSD_COLUMNS: Final = [
    "utc_timestamp",
    "cet_cest_timestamp",
    "DE_LU_load_actual_entsoe_transparency",
    "DE_LU_price_day_ahead",
]
PUBLIC_PROFILE_DATE: Final = "2019-01-24"
PUBLIC_PROFILE_YEAR: Final = "2019"
ACN_SITE_TIMEZONES: Final = {
    "caltech": "America/Los_Angeles",
    "jpl": "America/Los_Angeles",
    "office001": "America/Los_Angeles",
}
ACN_DEFAULT_CHARGE_LIMIT_KW: Final = 7.2
ACN_DEFAULT_DISCHARGE_LIMIT_KW: Final = 4.0
ACN_NEUTRAL_INITIAL_SOC_FRACTION: Final = 0.40
CASE_LABELS: Final = {
    "base": "Base",
    "high_grid_impact": "High grid impact",
    "high_target": "High target",
    "late_arrival": "Late arrival",
}


def display_case_name(case_name: str) -> str:
    return CASE_LABELS.get(case_name, case_name.replace("_", " ").title())


def build_base_case() -> EVData:
    """Create a 24-hour EV/V2G case from an OPSD DE-LU public data profile."""
    return build_case_from_profile(PUBLIC_PROFILE_DATE, name="base")


def build_case_from_profile(profile_date: str, name: str | None = None) -> EVData:
    profile = load_public_profile(profile_date)
    L = profile["station_baseload"].to_numpy(dtype=float)

    # The EV is parked from hour 11 to hour 21. Outside this window, x_t=y_t=0.
    availability = np.zeros_like(L)
    availability[10:21] = 1.0

    price_buy = profile["buy_price_eur_per_kwh"].to_numpy(dtype=float)
    price_sell = profile["sell_price_eur_per_kwh"].to_numpy(dtype=float)

    return EVData(
        name=name or profile_date,
        profile_date=profile_date,
        T=len(L),
        L=L,
        availability=availability,
        price_buy=price_buy,
        price_sell=price_sell,
        u_nominal=np.full(len(L), 1.5),
        X=30.0,
        Y=25.0,
        M=475.0,
        E0=50.0,
        Emin=10.0,
        Emax=75.0,
        Etarget=60.0,
        eta_c=0.95,
        eta_d=0.90,
        alpha=2.0e-4,
        rho=0.001,
    )


def build_cases() -> list[EVData]:
    base = build_base_case()

    late_availability = np.zeros(base.T)
    late_availability[12:22] = 1.0

    return [
        base,
        replace(base, name="high_grid_impact", u_nominal=np.full(base.T, 2.0)),
        replace(base, name="high_target", Etarget=70.0),
        replace(base, name="late_arrival", availability=late_availability, E0=55.0, Etarget=65.0),
    ]


def build_multi_day_cases(dates: list[str] | tuple[str, ...] | None = None) -> list[EVData]:
    if dates is None:
        dates = available_profile_dates_2019()
    return [build_case_from_profile(date, name=date) for date in dates]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def acn_cache_path(site: str, start_date: str, end_date: str, cache_path: str | None = None) -> Path:
    if cache_path:
        return Path(cache_path)
    local_export = Path(f"{site}.csv")
    if local_export.exists():
        return local_export
    return Path(f"data/acn_sessions_{site}_{start_date}_{end_date}.csv")


def acn_site_timezone(site: str) -> ZoneInfo:
    try:
        return ZoneInfo(ACN_SITE_TIMEZONES[site])
    except KeyError as exc:
        valid = ", ".join(sorted(ACN_SITE_TIMEZONES))
        raise ValueError(f"Unsupported ACN site {site!r}; expected one of {valid}") from exc


def normalize_acn_session(raw: dict, site: str, timezone_name: str) -> dict:
    user_inputs = raw.get("userInputs") or []
    requested_wh_per_mile = None
    requested_energy = None
    if user_inputs:
        last_input = user_inputs[-1] or {}
        requested_wh_per_mile = last_input.get("WhPerMile")
        requested_energy = last_input.get("kWhRequested")

    return {
        "session_id": raw.get("sessionID") or raw.get("_id") or raw.get("id"),
        "site": site,
        "timezone": timezone_name,
        "connectionTime": raw.get("connectionTime"),
        "disconnectTime": raw.get("disconnectTime"),
        "kWhDelivered": raw.get("kWhDelivered"),
        "stationID": raw.get("stationID"),
        "spaceID": raw.get("spaceID"),
        "siteID": raw.get("siteID") or site,
        "doneChargingTime": raw.get("doneChargingTime"),
        "userInput_kWhRequested": requested_energy,
        "userInput_WhPerMile": requested_wh_per_mile,
    }


def load_acn_sessions(
    site: str,
    start_date: str,
    end_date: str,
    min_kwh: float,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load session-level ACN records from a local CSV export.

    The supported inputs are either API-like session columns
    (connectionTime/disconnectTime/kWhDelivered) or compact exported columns
    (Start/End/Energy). This function intentionally does not query the ACN API or the
    ACN `/ts` endpoint.
    """
    site = site.lower()
    timezone_name = ACN_SITE_TIMEZONES.get(site)
    if timezone_name is None:
        valid = ", ".join(sorted(ACN_SITE_TIMEZONES))
        raise ValueError(f"Unsupported ACN site {site!r}; expected one of {valid}")

    cache = Path(cache_path) if cache_path else acn_cache_path(site, start_date, end_date)
    if cache.exists():
        try:
            frame = pd.read_csv(cache)
        except PermissionError:
            # Some Windows tools keep CSVs open in a way that blocks pandas' C engine.
            frame = pd.read_csv(cache, engine="python")
        frame = standardize_acn_session_frame(frame, site=site, timezone_name=timezone_name, min_kwh=min_kwh)
        local_start = datetime.fromisoformat(start_date).replace(tzinfo=ZoneInfo(timezone_name))
        local_end = datetime.fromisoformat(end_date).replace(tzinfo=ZoneInfo(timezone_name)) + timedelta(days=1)
        connection_local = parse_acn_timestamp(frame["connectionTime"], timezone_name)
        frame = frame[(connection_local >= local_start) & (connection_local < local_end)].copy()
        return frame

    raise RuntimeError(
        f"Local ACN session file not found at {cache}. Provide --acn-cache caltech.csv "
        "or place a site-named CSV such as caltech.csv in the project root."
    )


def standardize_acn_session_frame(
    frame: pd.DataFrame,
    site: str,
    timezone_name: str,
    min_kwh: float,
) -> pd.DataFrame:
    """Normalize local ACN session exports to the internal session schema."""
    work = frame.copy()
    rename_map = {}
    if "connectionTime" not in work.columns and "Start" in work.columns:
        rename_map["Start"] = "connectionTime"
    if "disconnectTime" not in work.columns and "End" in work.columns:
        rename_map["End"] = "disconnectTime"
    if "kWhDelivered" not in work.columns and "Energy" in work.columns:
        rename_map["Energy"] = "kWhDelivered"
    if rename_map:
        work = work.rename(columns=rename_map)

    required = {"connectionTime", "disconnectTime", "kWhDelivered"}
    missing = sorted(required - set(work.columns))
    if missing:
        raise ValueError(
            "ACN session CSV must contain either connectionTime/disconnectTime/kWhDelivered "
            f"or Start/End/Energy columns. Missing: {missing}"
        )

    if "session_id" not in work.columns:
        work["session_id"] = np.arange(len(work))
    if "site" not in work.columns:
        work["site"] = site
    if "siteID" not in work.columns:
        work["siteID"] = site
    if "stationID" not in work.columns:
        work["stationID"] = ""
    if "spaceID" not in work.columns:
        work["spaceID"] = ""
    if "timezone" not in work.columns:
        work["timezone"] = timezone_name
    if "doneChargingTime" not in work.columns:
        work["doneChargingTime"] = ""
    if "userInput_kWhRequested" not in work.columns:
        work["userInput_kWhRequested"] = np.nan
    if "userInput_WhPerMile" not in work.columns:
        work["userInput_WhPerMile"] = np.nan

    work["kWhDelivered"] = pd.to_numeric(work["kWhDelivered"], errors="coerce")
    work = work[work["kWhDelivered"] >= float(min_kwh)].copy()
    return work[
        [
            "session_id",
            "site",
            "timezone",
            "connectionTime",
            "disconnectTime",
            "kWhDelivered",
            "stationID",
            "spaceID",
            "siteID",
            "doneChargingTime",
            "userInput_kWhRequested",
            "userInput_WhPerMile",
        ]
    ]


def parse_acn_timestamp(series: pd.Series, timezone_name: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(timezone_name, nonexistent="shift_forward", ambiguous="NaT")
    return parsed.dt.tz_convert(timezone_name)


def build_fleet_from_acn_sessions(
    sessions: pd.DataFrame,
    horizon_date: str,
    base: EVData,
    n_vehicles: int | None = None,
    slot_minutes: int = 60,
    seed: int = 20260506,
    site: str = "caltech",
    cache_path: str | Path | None = None,
    grid_headroom_per_vehicle: float = FLEET_GRID_HEADROOM_PER_VEHICLE,
) -> FleetData:
    """Build a 24-slot fleet instance from real ACN session-level records."""
    if sessions.empty:
        raise ValueError("ACN session table is empty")
    if slot_minutes != 60:
        raise ValueError("The current paper experiments use hourly 24-slot horizons")

    site = site.lower()
    timezone_name = sessions.get("timezone", pd.Series([ACN_SITE_TIMEZONES.get(site, "America/Los_Angeles")])).dropna()
    timezone_name = str(timezone_name.iloc[0]) if not timezone_name.empty else ACN_SITE_TIMEZONES.get(site, "America/Los_Angeles")
    tz = ZoneInfo(timezone_name)
    horizon_start = datetime.fromisoformat(horizon_date).replace(tzinfo=tz)
    horizon_end = horizon_start + timedelta(days=1)

    work = sessions.copy()
    work["connection_local"] = parse_acn_timestamp(work["connectionTime"], timezone_name)
    work["disconnect_local"] = parse_acn_timestamp(work["disconnectTime"], timezone_name)
    work["required_energy"] = pd.to_numeric(work["kWhDelivered"], errors="coerce")
    work = work.dropna(subset=["connection_local", "disconnect_local", "required_energy"])
    work = work[work["required_energy"] > 0.0].copy()
    work = work[work["disconnect_local"] > work["connection_local"]].copy()
    work = work[
        (work["connection_local"] < horizon_end)
        & (work["disconnect_local"] > horizon_start)
    ].copy()

    if work.empty:
        raise ValueError(f"No valid ACN sessions overlap local horizon date {horizon_date}")

    clipped_start = work["connection_local"].apply(lambda ts: max(ts.to_pydatetime(), horizon_start))
    clipped_end = work["disconnect_local"].apply(lambda ts: min(ts.to_pydatetime(), horizon_end))
    arrival_hours = clipped_start.apply(lambda ts: (ts - horizon_start).total_seconds() / 3600.0)
    departure_hours = clipped_end.apply(lambda ts: (ts - horizon_start).total_seconds() / 3600.0)
    work["arrival_slot"] = np.floor(arrival_hours).astype(int)
    work["departure_slot"] = np.ceil(departure_hours).astype(int)
    work["arrival_slot"] = work["arrival_slot"].clip(0, base.T - 1)
    work["departure_slot"] = work["departure_slot"].clip(1, base.T)
    work["available_slots"] = work["departure_slot"] - work["arrival_slot"]
    max_deliverable = base.eta_c * ACN_DEFAULT_CHARGE_LIMIT_KW * work["available_slots"]
    work = work[(work["available_slots"] > 0) & (work["required_energy"] <= max_deliverable)].copy()

    if work.empty:
        raise ValueError(f"No ACN sessions pass feasibility filters for {horizon_date}")

    work = work.sort_values(["connection_local", "disconnect_local", "session_id"], kind="mergesort")
    if n_vehicles is not None and len(work) > n_vehicles:
        work = work.sample(n=n_vehicles, random_state=seed).sort_values("connection_local")
    elif n_vehicles is not None and len(work) < n_vehicles:
        raise ValueError(
            f"Only {len(work)} feasible ACN sessions on {horizon_date}; requested {n_vehicles}"
        )

    n = len(work)
    availability = np.zeros((n, base.T), dtype=float)
    emax = np.empty(n)
    emin = np.empty(n)
    e0 = np.empty(n)
    etarget = np.empty(n)
    x_limit = np.full(n, ACN_DEFAULT_CHARGE_LIMIT_KW)
    y_limit = np.full(n, ACN_DEFAULT_DISCHARGE_LIMIT_KW)

    for row_idx, (_, row) in enumerate(work.iterrows()):
        arrival = int(row["arrival_slot"])
        departure = int(row["departure_slot"])
        required = float(row["required_energy"])
        availability[row_idx, arrival:departure] = 1.0

        # ACN-Data does not provide observed initial SOC in the session records used
        # here. We convert delivered energy into a terminal energy increment.
        capacity = max(55.0, min(100.0, required / 0.45 + 20.0))
        neutral_e0 = ACN_NEUTRAL_INITIAL_SOC_FRACTION * capacity
        target = neutral_e0 + required
        if target > 0.92 * capacity:
            capacity = target / 0.92
        emax[row_idx] = capacity
        emin[row_idx] = 0.10 * capacity
        e0[row_idx] = neutral_e0
        etarget[row_idx] = target

    vehicle_ids = np.arange(n).reshape(n, 1)
    time_grid_factors = np.linspace(0.97, 1.03, base.T).reshape(1, base.T)
    vehicle_grid_factors = 0.95 + 0.10 * ((vehicle_ids % 7) / 6.0)
    u_nominal = base.u_nominal.reshape(1, base.T) * vehicle_grid_factors * time_grid_factors

    return FleetData(
        name=f"acn_{site}_{horizon_date}_{n}",
        profile_date=base.profile_date,
        N=n,
        T=base.T,
        L=base.L,
        availability=availability,
        price_buy=base.price_buy,
        price_sell=base.price_sell,
        u_nominal=u_nominal,
        X=x_limit,
        Y=y_limit,
        M=base.M + grid_headroom_per_vehicle * n,
        E0=e0,
        Emin=emin,
        Emax=emax,
        Etarget=etarget,
        eta_c=np.full(n, base.eta_c),
        eta_d=np.full(n, base.eta_d),
        alpha=base.alpha,
        rho=np.full(n, base.rho),
        source="acn_sessions",
        acn_site=site,
        acn_cache=str(cache_path or ""),
        horizon_date=horizon_date,
        source_sessions=n,
    )


def feasible_acn_horizon_dates(
    sessions: pd.DataFrame,
    base: EVData,
    site: str,
    slot_minutes: int,
    min_sessions: int,
) -> list[str]:
    timezone_name = sessions.get("timezone", pd.Series([ACN_SITE_TIMEZONES.get(site, "America/Los_Angeles")])).dropna()
    timezone_name = str(timezone_name.iloc[0]) if not timezone_name.empty else ACN_SITE_TIMEZONES.get(site, "America/Los_Angeles")
    parsed = parse_acn_timestamp(sessions["connectionTime"], timezone_name)
    candidate_dates = sorted(parsed.dropna().dt.date.astype(str).unique())
    feasible = []
    for date in candidate_dates:
        try:
            build_fleet_from_acn_sessions(
                sessions,
                horizon_date=date,
                base=base,
                n_vehicles=min_sessions,
                slot_minutes=slot_minutes,
                seed=0,
                site=site,
            )
            feasible.append(date)
        except ValueError:
            continue
    return feasible


def prepare_acn_metadata(
    sessions: pd.DataFrame,
    base: EVData,
    config: FleetExperimentConfig,
    cache_path: Path,
    out_dir: Path,
) -> pd.DataFrame:
    max_fleet = max(FLEET_COUNTS)
    dates = feasible_acn_horizon_dates(
        sessions,
        base=base,
        site=config.acn_site,
        slot_minutes=config.slot_minutes,
        min_sessions=max_fleet,
    )
    metadata = pd.DataFrame(
        [
            {
                "acn_site": config.acn_site,
                "acn_start": config.acn_start,
                "acn_end": config.acn_end,
                "acn_min_kwh": config.acn_min_kwh,
                "acn_cache": str(cache_path),
                "raw_sessions": len(sessions),
                "valid_horizon_days_for_max_fleet": len(dates),
                "max_fleet_size": max_fleet,
                "slot_minutes": config.slot_minutes,
            }
        ]
    )
    out_dir.mkdir(exist_ok=True)
    metadata.to_csv(out_dir / "acn_fleet_metadata.csv", index=False)
    pd.DataFrame({"horizon_date": dates}).to_csv(out_dir / "acn_fleet_horizon_dates.csv", index=False)
    return metadata


def build_fleet_case_for_experiment(
    base: EVData,
    n_vehicles: int,
    seed: int,
    config: FleetExperimentConfig,
    acn_sessions: pd.DataFrame | None = None,
    horizon_dates: list[str] | None = None,
) -> FleetData:
    if not config.use_acn_sessions:
        return build_synthetic_fleet_case(base, n_vehicles=n_vehicles, seed=seed)

    if acn_sessions is None:
        cache = acn_cache_path(config.acn_site, config.acn_start, config.acn_end, config.acn_cache)
        acn_sessions = load_acn_sessions(
            config.acn_site,
            config.acn_start,
            config.acn_end,
            config.acn_min_kwh,
            cache,
        )
    if horizon_dates is None:
        horizon_dates = feasible_acn_horizon_dates(
            acn_sessions,
            base=base,
            site=config.acn_site,
            slot_minutes=config.slot_minutes,
            min_sessions=n_vehicles,
        )
    if not horizon_dates:
        raise ValueError(
            f"No ACN horizon dates contain at least {n_vehicles} feasible sessions "
            f"after filtering for site={config.acn_site}."
        )
    horizon_date = horizon_dates[seed % len(horizon_dates)]
    return build_fleet_from_acn_sessions(
        acn_sessions,
        horizon_date=horizon_date,
        base=base,
        n_vehicles=n_vehicles,
        slot_minutes=config.slot_minutes,
        seed=seed,
        site=config.acn_site,
        cache_path=acn_cache_path(config.acn_site, config.acn_start, config.acn_end, config.acn_cache),
    )


def load_acn_session_templates(horizon: int) -> list[tuple[int, int, float]]:
    """Load optional ACN-Data session templates from a local CSV export.

    Expected columns are flexible: connection/disconnect timestamps plus delivered
    energy, or explicit arrival/departure hour columns plus requested energy. If no
    local export is present, the fleet generator falls back to ACN-informed ranges.
    """
    candidates = [
        Path("data/acn_sessions.csv"),
        Path("data/acn_caltech_sessions.csv"),
        Path("data/acn_jpl_sessions.csv"),
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return []

    sessions = pd.read_csv(path)
    lower_columns = {column.lower(): column for column in sessions.columns}

    energy_col = next(
        (
            lower_columns[name]
            for name in ("kwhdelivered", "kwh_delivered", "energy_kwh", "requested_energy", "requested_kwh")
            if name in lower_columns
        ),
        None,
    )
    if energy_col is None:
        return []

    templates: list[tuple[int, int, float]] = []
    if "connectiontime" in lower_columns and "disconnecttime" in lower_columns:
        connected = pd.to_datetime(sessions[lower_columns["connectiontime"]], errors="coerce")
        disconnected = pd.to_datetime(sessions[lower_columns["disconnecttime"]], errors="coerce")
        arrivals = connected.dt.hour + connected.dt.minute / 60.0
        departures = disconnected.dt.hour + disconnected.dt.minute / 60.0
        overnight = disconnected.dt.date > connected.dt.date
        departures = departures.where(~overnight, departures + 24.0)
    else:
        arrival_col = next((lower_columns[name] for name in ("arrival", "arrival_hour", "connect_hour") if name in lower_columns), None)
        departure_col = next((lower_columns[name] for name in ("departure", "departure_hour", "disconnect_hour") if name in lower_columns), None)
        if arrival_col is None or departure_col is None:
            return []
        arrivals = pd.to_numeric(sessions[arrival_col], errors="coerce")
        departures = pd.to_numeric(sessions[departure_col], errors="coerce")

    energies = pd.to_numeric(sessions[energy_col], errors="coerce")
    for arrival_raw, departure_raw, energy_raw in zip(arrivals, departures, energies):
        if not np.isfinite(arrival_raw) or not np.isfinite(departure_raw) or not np.isfinite(energy_raw):
            continue
        arrival = int(np.clip(np.floor(arrival_raw), 0, horizon - 1))
        departure = int(np.clip(np.ceil(departure_raw), arrival + 1, horizon))
        energy = float(np.clip(energy_raw, 2.0, 35.0))
        if departure > arrival and energy > 0.0:
            templates.append((arrival, departure, energy))
    return templates


def build_synthetic_fleet_case(
    base: EVData,
    n_vehicles: int,
    seed: int = 20260506,
    grid_headroom_per_vehicle: float = FLEET_GRID_HEADROOM_PER_VEHICLE,
) -> FleetData:
    """Build a nested synthetic fleet extension from the single-EV public profile.

    Fleet sizes are nested for paper-readability: N=20 contains the same first
    10 EVs as N=10 plus 10 additional EVs, and so on.
    """
    rng = np.random.default_rng(seed)
    availability = np.zeros((n_vehicles, base.T), dtype=float)
    emax = np.empty(n_vehicles)
    emin = np.empty(n_vehicles)
    e0 = np.empty(n_vehicles)
    etarget = np.empty(n_vehicles)
    x_limit = np.empty(n_vehicles)
    y_limit = np.empty(n_vehicles)
    acn_templates = load_acn_session_templates(base.T)

    for i in range(n_vehicles):
        if acn_templates:
            arrival, departure, requested_energy = acn_templates[int(rng.integers(0, len(acn_templates)))]
        else:
            arrival = int(rng.integers(7, 16))
            dwell = int(rng.integers(7, 13))
            departure = min(base.T, arrival + dwell)
            requested_energy = float(rng.uniform(6.0, 18.0))
        availability[i, arrival:departure] = 1.0

        emax[i] = rng.uniform(55.0, 85.0)
        emin[i] = 0.12 * emax[i]
        e0[i] = rng.uniform(0.35, 0.75) * emax[i]
        etarget[i] = min(0.92 * emax[i], e0[i] + requested_energy)
        x_limit[i] = rng.uniform(7.0, 11.0)
        y_limit[i] = rng.uniform(4.0, 8.0)

    vehicle_grid_factors = rng.uniform(0.9, 1.1, size=(n_vehicles, 1))
    time_grid_factors = rng.uniform(0.95, 1.05, size=(1, base.T))
    u_nominal = base.u_nominal.reshape(1, base.T) * vehicle_grid_factors * time_grid_factors

    return FleetData(
        name=f"fleet_{n_vehicles}",
        profile_date=base.profile_date,
        N=n_vehicles,
        T=base.T,
        L=base.L,
        availability=availability,
        price_buy=base.price_buy,
        price_sell=base.price_sell,
        u_nominal=u_nominal,
        X=x_limit,
        Y=y_limit,
        M=base.M + grid_headroom_per_vehicle * n_vehicles,
        E0=e0,
        Emin=emin,
        Emax=emax,
        Etarget=etarget,
        eta_c=np.full(n_vehicles, base.eta_c),
        eta_d=np.full(n_vehicles, base.eta_d),
        alpha=base.alpha,
        rho=np.full(n_vehicles, base.rho),
    )


def profile_cache_path(profile_date: str) -> Path:
    if profile_date == PUBLIC_PROFILE_DATE:
        return OPSD_PROFILE_CACHE
    return Path(f"data/opsd_de_lu_{profile_date.replace('-', '_')}_profile.csv")


def load_public_profile(profile_date: str = PUBLIC_PROFILE_DATE) -> pd.DataFrame:
    cache_path = profile_cache_path(profile_date)
    if cache_path.exists():
        return pd.read_csv(cache_path)

    cache_path.parent.mkdir(exist_ok=True)
    data = load_opsd_year()
    day = data[data["utc_timestamp"].str.startswith(profile_date)].copy()
    if len(day) != 24:
        raise ValueError(f"Expected 24 hourly rows for {profile_date}, found {len(day)}")
    if day[OPSD_COLUMNS].isna().any().any():
        raise ValueError(f"Missing OPSD values in the selected profile for {profile_date}")

    raw_load = day["DE_LU_load_actual_entsoe_transparency"].astype(float).to_numpy()
    raw_price = day["DE_LU_price_day_ahead"].astype(float).to_numpy()
    station_baseload = affine_scale(raw_load, lower=390.0, upper=475.0)

    day_ahead_eur_per_kwh = np.maximum(raw_price, 1.0) / 1000.0
    buy_price = day_ahead_eur_per_kwh
    sell_price = 0.98 * buy_price

    profile = pd.DataFrame(
        {
            "hour": np.arange(1, 25),
            "utc_timestamp": day["utc_timestamp"].to_numpy(),
            "cet_cest_timestamp": day["cet_cest_timestamp"].to_numpy(),
            "raw_load_mw": raw_load,
            "raw_price_eur_per_mwh": raw_price,
            "station_baseload": station_baseload,
            "buy_price_eur_per_kwh": buy_price,
            "sell_price_eur_per_kwh": sell_price,
            "source": f"Open Power System Data time_series 2020-10-06, DE-LU, {profile_date}",
        }
    )
    profile.to_csv(cache_path, index=False)
    return profile


def available_profile_dates_2019() -> list[str]:
    year = load_opsd_year()
    complete_dates = []
    for date, frame in year.groupby(year["utc_timestamp"].str[:10]):
        if len(frame) == 24 and not frame[OPSD_COLUMNS].isna().any().any():
            complete_dates.append(date)
    return sorted(complete_dates)


def load_opsd_year() -> pd.DataFrame:
    if OPSD_YEAR_CACHE.exists():
        return pd.read_csv(OPSD_YEAR_CACHE)

    OPSD_YEAR_CACHE.parent.mkdir(exist_ok=True)
    start = pd.Timestamp("2019-01-01T00:00:00Z")
    end = pd.Timestamp("2020-01-01T00:00:00Z")
    rows = []
    for chunk in pd.read_csv(OPSD_URL, usecols=OPSD_COLUMNS, chunksize=5000):
        timestamps = pd.to_datetime(chunk["utc_timestamp"], utc=True)
        mask = (timestamps >= start) & (timestamps < end)
        if mask.any():
            rows.append(chunk.loc[mask].copy())
        if len(timestamps) and timestamps.iloc[-1] >= end:
            break
    if not rows:
        raise RuntimeError("Could not extract the OPSD DE-LU 2019 profiles")
    year = pd.concat(rows, ignore_index=True)
    year.to_csv(OPSD_YEAR_CACHE, index=False)
    return year


def load_opsd_month() -> pd.DataFrame:
    if OPSD_MONTH_CACHE.exists():
        return pd.read_csv(OPSD_MONTH_CACHE)

    OPSD_MONTH_CACHE.parent.mkdir(exist_ok=True)
    start = pd.Timestamp("2019-01-01T00:00:00Z")
    end = pd.Timestamp("2019-02-01T00:00:00Z")
    rows = []
    for chunk in pd.read_csv(OPSD_URL, usecols=OPSD_COLUMNS, chunksize=5000):
        timestamps = pd.to_datetime(chunk["utc_timestamp"], utc=True)
        mask = (timestamps >= start) & (timestamps < end)
        if mask.any():
            rows.append(chunk.loc[mask].copy())
        if len(timestamps) and timestamps.iloc[-1] >= end:
            break
    if not rows:
        raise RuntimeError("Could not extract the OPSD DE-LU January 2019 profile")
    month = pd.concat(rows, ignore_index=True)
    month.to_csv(OPSD_MONTH_CACHE, index=False)
    return month


def affine_scale(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if np.isclose(min_value, max_value):
        return np.full_like(values, (lower + upper) / 2.0)
    return lower + (values - min_value) * (upper - lower) / (max_value - min_value)


def uncertainty_box(data: EVData, delta: float) -> tuple[np.ndarray, np.ndarray]:
    lower = np.maximum(0.0, (1.0 - delta) * data.u_nominal)
    upper = (1.0 + delta) * data.u_nominal
    return lower, upper


def policy_u(data: EVData, delta: float, policy: Policy) -> np.ndarray:
    lower, upper = uncertainty_box(data, delta)
    if policy.name == "deterministic":
        return data.u_nominal
    if policy.chance_epsilon is not None:
        quantile = 1.0 - policy.chance_epsilon
        return lower + quantile * (upper - lower)
    if policy.uncertainty_quantile is None:
        raise ValueError(f"Policy {policy.name} must define either a quantile or a chance epsilon")
    return lower + policy.uncertainty_quantile * (upper - lower)


def fleet_uncertainty_box(data: FleetData, delta: float) -> tuple[np.ndarray, np.ndarray]:
    nominal = fleet_u_matrix(data)
    lower = np.maximum(0.0, (1.0 - delta) * nominal)
    upper = (1.0 + delta) * nominal
    return lower, upper


def fleet_policy_u(data: FleetData, delta: float, policy: Policy) -> np.ndarray:
    lower, upper = fleet_uncertainty_box(data, delta)
    if policy.name == "deterministic":
        return fleet_u_matrix(data)
    if policy.chance_epsilon is not None:
        quantile = 1.0 - policy.chance_epsilon
        return lower + quantile * (upper - lower)
    if policy.uncertainty_quantile is None:
        raise ValueError(f"Policy {policy.name} must define either a quantile or a chance epsilon")
    return lower + policy.uncertainty_quantile * (upper - lower)


def fleet_u_matrix(data: FleetData) -> np.ndarray:
    nominal = np.asarray(data.u_nominal, dtype=float)
    if nominal.ndim == 1:
        return np.tile(nominal.reshape(1, data.T), (data.N, 1))
    if nominal.shape != (data.N, data.T):
        raise ValueError(f"Fleet u_nominal must have shape ({data.N}, {data.T}) or ({data.T},)")
    return nominal


def solve_model(data: EVData, delta: float, policy: Policy) -> dict:
    u_used = policy_u(data, delta, policy)

    x = cp.Variable(data.T, nonneg=True, name="charge")
    y = cp.Variable(data.T, nonneg=True, name="discharge")
    E = cp.Variable(data.T + 1, name="state_of_charge")
    p = x - y

    objective = cp.sum(
        cp.multiply(data.price_buy, x)
        - cp.multiply(data.price_sell, y)
        + data.alpha * cp.square(data.L + p)
        + data.rho * (x + y)
    )

    constraints = [
        x <= data.X * data.availability,
        y <= data.Y * data.availability,
        E[0] == data.E0,
        E[1:] == E[:-1] + data.eta_c * x - y / data.eta_d,
        E[1:] >= data.Emin,
        E[1:] <= data.Emax,
        E[-1] >= data.Etarget,
        data.L + cp.multiply(u_used, x) <= data.M,
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    start_time = time.perf_counter()
    solve_with_fallback(problem)
    solve_time_sec = time.perf_counter() - start_time

    status = problem.status
    base_result = {
        "case": data.name,
        "policy": policy.name,
        "policy_label": policy.label,
        "delta": delta,
        "status": status,
        "objective": np.nan,
        "solve_time_sec": float(solve_time_sec),
    }
    if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        return base_result

    charge = clean_array(x.value)
    discharge = clean_array(y.value)
    soc = clean_array(E.value)
    net = charge - discharge
    station_load = data.L + net
    nominal_grid_margin = data.L + data.u_nominal * charge - data.M

    return {
        **base_result,
        "objective": float(problem.value),
        "x": charge,
        "y": discharge,
        "E": soc,
        "p": net,
        "station_load": station_load,
        "total_charge": float(charge.sum()),
        "total_discharge": float(discharge.sum()),
        "terminal_energy": float(soc[-1]),
        "peak_load": float(station_load.max()),
        "economic_cost": float(
            np.sum(data.price_buy * charge - data.price_sell * discharge + data.rho * (charge + discharge))
        ),
        "load_penalty": float(np.sum(data.alpha * station_load**2)),
        "paper_like_cost": float(np.sum((data.L + net) * net)),
        "max_simultaneous_flow": float(np.max(np.minimum(charge, discharge))),
        "nominal_grid_margin": float(np.max(nominal_grid_margin)),
    }


def build_small_validation_case(
    base: EVData,
    *,
    name: str = "miqp_peak_8",
    start: int = 10,
    stop: int = 18,
    grid_limit: float = 490.0,
    target: float = 60.0,
    initial_energy: float = 50.0,
) -> EVData:
    """Small quadratic-cost instance for binary complementarity validation."""
    return EVData(
        name=name,
        profile_date=base.profile_date,
        T=stop - start,
        L=base.L[start:stop],
        availability=np.ones(stop - start),
        price_buy=base.price_buy[start:stop],
        price_sell=base.price_sell[start:stop],
        u_nominal=base.u_nominal[start:stop],
        X=base.X,
        Y=base.Y,
        M=grid_limit,
        E0=initial_energy,
        Emin=base.Emin,
        Emax=base.Emax,
        Etarget=target,
        eta_c=base.eta_c,
        eta_d=base.eta_d,
        alpha=base.alpha,
        rho=base.rho,
    )


def build_miqp_validation_cases(base: EVData) -> list[EVData]:
    """Return several small cases where robust grid protection is sometimes binding."""
    configs = [
        ("miqp_peak_8", 10, 18, 490.0, 60.0),
        ("miqp_wide_binding_12", 10, 22, 485.0, 65.0),
        ("miqp_tight_binding_12", 10, 22, 480.0, 62.0),
        ("miqp_late_binding_10", 12, 22, 480.0, 64.0),
    ]
    return [
        build_small_validation_case(base, name=name, start=start, stop=stop, grid_limit=grid_limit, target=target)
        for name, start, stop, grid_limit, target in configs
    ]


def build_extended_miqp_validation_cases(base: EVData) -> list[EVData]:
    """Return a larger MIQP validation set with multiple horizon lengths."""
    configs = [
        ("miqp_ext_peak_T8", 10, 18, 490.0, 60.0),
        ("miqp_ext_late_T8", 12, 20, 485.0, 60.0),
        ("miqp_ext_peak_T12", 10, 22, 485.0, 65.0),
        ("miqp_ext_late_T12", 12, 24, 480.0, 64.0),
        ("miqp_ext_mid_T16", 6, 22, 485.0, 66.0),
        ("miqp_ext_late_T16", 8, 24, 480.0, 66.0),
        ("miqp_ext_mid_T20", 4, 24, 485.0, 68.0),
        ("miqp_ext_full_T20", 2, 22, 480.0, 67.0),
        ("miqp_ext_full_T24", 0, 24, 485.0, 68.0),
        ("miqp_ext_tight_T24", 0, 24, 480.0, 67.0),
    ]
    return [
        build_small_validation_case(base, name=name, start=start, stop=stop, grid_limit=grid_limit, target=target)
        for name, start, stop, grid_limit, target in configs
    ]


def solve_binary_complementarity_model(
    data: EVData,
    delta: float,
    policy: Policy,
    time_limit_sec: float | None = None,
) -> dict:
    """Solve a small MIQP validation model with explicit charge/discharge binaries."""
    u_used = policy_u(data, delta, policy)
    x = cp.Variable(data.T, nonneg=True, name="binary_charge")
    y = cp.Variable(data.T, nonneg=True, name="binary_discharge")
    E = cp.Variable(data.T + 1, name="binary_state_of_charge")
    z = cp.Variable(data.T, boolean=True, name="charge_mode")
    p = x - y

    objective = cp.sum(
        cp.multiply(data.price_buy, x)
        - cp.multiply(data.price_sell, y)
        + data.rho * (x + y)
    )
    if data.alpha > 0.0:
        objective += cp.sum(data.alpha * cp.square(data.L + p))

    constraints = [
        x <= cp.multiply(data.X * data.availability, z),
        y <= cp.multiply(data.Y * data.availability, 1.0 - z),
        E[0] == data.E0,
        E[1:] == E[:-1] + data.eta_c * x - y / data.eta_d,
        E[1:] >= data.Emin,
        E[1:] <= data.Emax,
        E[-1] >= data.Etarget,
        data.L + cp.multiply(u_used, x) <= data.M,
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    start_time = time.perf_counter()
    try:
        solve_mixed_integer_with_fallback(problem, time_limit_sec=time_limit_sec)
    except cp.SolverError as exc:
        return {
            "case": data.name,
            "policy": f"binary_{policy.name}",
            "policy_label": f"MIQP {policy.label}",
            "delta": delta,
            "status": f"solver_error: {exc}",
            "objective": np.nan,
            "solve_time_sec": float(time.perf_counter() - start_time),
        }
    solve_time_sec = time.perf_counter() - start_time

    base_result = {
        "case": data.name,
        "policy": f"binary_{policy.name}",
        "policy_label": f"MIQP {policy.label}",
        "delta": delta,
        "status": problem.status,
        "objective": np.nan,
        "solve_time_sec": float(solve_time_sec),
    }
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        return base_result

    charge = clean_array(x.value)
    discharge = clean_array(y.value)
    soc = clean_array(E.value)
    net = charge - discharge
    return {
        **base_result,
        "objective": float(problem.value),
        "x": charge,
        "y": discharge,
        "E": soc,
        "p": net,
        "station_load": data.L + net,
        "total_charge": float(charge.sum()),
        "total_discharge": float(discharge.sum()),
        "terminal_energy": float(soc[-1]),
        "peak_load": float((data.L + net).max()),
        "economic_cost": float(
            np.sum(data.price_buy * charge - data.price_sell * discharge + data.rho * (charge + discharge))
        ),
        "load_penalty": float(np.sum(data.alpha * (data.L + net) ** 2)),
        "paper_like_cost": float(np.sum((data.L + net) * net)),
        "max_simultaneous_flow": float(np.max(np.minimum(charge, discharge))),
        "nominal_grid_margin": float(np.max(data.L + data.u_nominal * charge - data.M)),
    }


def solve_fleet_model(data: FleetData, delta: float, policy: Policy) -> dict:
    """Solve the fleet-level convex QP extension with a shared station grid constraint."""
    u_used = fleet_policy_u(data, delta, policy)

    x = cp.Variable((data.N, data.T), nonneg=True, name="fleet_charge")
    y = cp.Variable((data.N, data.T), nonneg=True, name="fleet_discharge")
    E = cp.Variable((data.N, data.T + 1), name="fleet_state_of_charge")
    aggregate_charge = cp.sum(x, axis=0)
    aggregate_discharge = cp.sum(y, axis=0)
    station_load = data.L + aggregate_charge - aggregate_discharge

    price_buy = data.price_buy.reshape(1, data.T)
    price_sell = data.price_sell.reshape(1, data.T)
    throughput_penalty = data.rho.reshape(data.N, 1)

    objective = cp.sum(
        cp.multiply(price_buy, x)
        - cp.multiply(price_sell, y)
        + cp.multiply(throughput_penalty, x + y)
    ) + cp.sum(data.alpha * cp.square(station_load))

    constraints = [
        x <= data.X.reshape(data.N, 1) * data.availability,
        y <= data.Y.reshape(data.N, 1) * data.availability,
        E[:, 0] == data.E0,
        E[:, 1:] == E[:, :-1] + cp.multiply(data.eta_c.reshape(data.N, 1), x)
        - cp.multiply((1.0 / data.eta_d).reshape(data.N, 1), y),
        E[:, 1:] >= data.Emin.reshape(data.N, 1),
        E[:, 1:] <= data.Emax.reshape(data.N, 1),
        E[:, -1] >= data.Etarget,
        data.L + cp.sum(cp.multiply(u_used, x), axis=0) <= data.M,
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    start_time = time.perf_counter()
    solve_fleet_with_fallback(problem)
    solve_time_sec = time.perf_counter() - start_time

    status = problem.status
    base_result = {
        "case": data.name,
        "profile_date": data.profile_date,
        "n_vehicles": data.N,
        "policy": policy.name,
        "policy_label": policy.label,
        "delta": delta,
        "status": status,
        "objective": np.nan,
        "solve_time_sec": float(solve_time_sec),
        "decision_variables": int(2 * data.N * data.T + data.N * (data.T + 1)),
    }
    if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        return base_result

    charge = clean_array(x.value).reshape(data.N, data.T)
    discharge = clean_array(y.value).reshape(data.N, data.T)
    soc = clean_array(E.value).reshape(data.N, data.T + 1)
    aggregate_charge_value = charge.sum(axis=0)
    aggregate_discharge_value = discharge.sum(axis=0)
    station_load_value = data.L + aggregate_charge_value - aggregate_discharge_value
    nominal_grid_margin = data.L + np.sum(fleet_u_matrix(data) * charge, axis=0) - data.M

    return {
        **base_result,
        "objective": float(problem.value),
        "x": charge,
        "y": discharge,
        "E": soc,
        "aggregate_charge": aggregate_charge_value,
        "aggregate_discharge": aggregate_discharge_value,
        "station_load": station_load_value,
        "total_charge": float(charge.sum()),
        "total_discharge": float(discharge.sum()),
        "mean_terminal_energy": float(soc[:, -1].mean()),
        "peak_load": float(station_load_value.max()),
        "economic_cost": float(
            np.sum(data.price_buy * aggregate_charge_value - data.price_sell * aggregate_discharge_value)
            + np.sum(data.rho.reshape(data.N, 1) * (charge + discharge))
        ),
        "load_penalty": float(np.sum(data.alpha * station_load_value**2)),
        "max_simultaneous_flow": float(np.max(np.minimum(charge, discharge))),
        "nominal_grid_margin": float(np.max(nominal_grid_margin)),
    }


def solve_fleet_budgeted_model(data: FleetData, delta: float, gamma: float) -> dict:
    """Solve the Bertsimas-Sim budgeted robust fleet counterpart."""
    nominal = fleet_u_matrix(data)
    deviation = delta * nominal

    x = cp.Variable((data.N, data.T), nonneg=True, name="budgeted_charge")
    y = cp.Variable((data.N, data.T), nonneg=True, name="budgeted_discharge")
    E = cp.Variable((data.N, data.T + 1), name="budgeted_state_of_charge")
    theta = cp.Variable(data.T, nonneg=True, name="budgeted_theta")
    slack = cp.Variable((data.N, data.T), nonneg=True, name="budgeted_slack")

    aggregate_charge = cp.sum(x, axis=0)
    aggregate_discharge = cp.sum(y, axis=0)
    station_load = data.L + aggregate_charge - aggregate_discharge

    price_buy = data.price_buy.reshape(1, data.T)
    price_sell = data.price_sell.reshape(1, data.T)
    throughput_penalty = data.rho.reshape(data.N, 1)
    objective = cp.sum(
        cp.multiply(price_buy, x)
        - cp.multiply(price_sell, y)
        + cp.multiply(throughput_penalty, x + y)
    ) + cp.sum(data.alpha * cp.square(station_load))

    constraints = [
        x <= data.X.reshape(data.N, 1) * data.availability,
        y <= data.Y.reshape(data.N, 1) * data.availability,
        E[:, 0] == data.E0,
        E[:, 1:] == E[:, :-1] + cp.multiply(data.eta_c.reshape(data.N, 1), x)
        - cp.multiply((1.0 / data.eta_d).reshape(data.N, 1), y),
        E[:, 1:] >= data.Emin.reshape(data.N, 1),
        E[:, 1:] <= data.Emax.reshape(data.N, 1),
        E[:, -1] >= data.Etarget,
        slack + cp.reshape(theta, (1, data.T), order="C") >= cp.multiply(deviation, x),
        data.L + cp.sum(cp.multiply(nominal, x), axis=0) + gamma * theta + cp.sum(slack, axis=0) <= data.M,
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    start_time = time.perf_counter()
    solve_fleet_with_fallback(problem)
    return collect_fleet_result(
        data=data,
        problem=problem,
        x_var=x,
        y_var=y,
        E_var=E,
        start_time=start_time,
        delta=delta,
        policy=f"budgeted_gamma_{gamma:g}",
        policy_label=f"Budgeted B={gamma:g}",
        model_family="budgeted_robust",
        gamma=gamma,
        decision_variables=int(2 * data.N * data.T + data.N * (data.T + 1) + data.T + data.N * data.T),
    )


def solve_fleet_bernstein_model(data: FleetData, delta: float, epsilon: float) -> dict:
    """Solve the Bernstein-type SOCP fleet chance counterpart."""
    nominal = fleet_u_matrix(data)
    deviation = delta * nominal
    sigma = deviation / np.sqrt(3.0)
    per_slot_epsilon = epsilon / data.T
    omega = float(np.log(1.0 / per_slot_epsilon))

    x = cp.Variable((data.N, data.T), nonneg=True, name="bernstein_charge")
    y = cp.Variable((data.N, data.T), nonneg=True, name="bernstein_discharge")
    E = cp.Variable((data.N, data.T + 1), name="bernstein_state_of_charge")
    radius = cp.Variable(data.T, nonneg=True, name="bernstein_radius")

    aggregate_charge = cp.sum(x, axis=0)
    aggregate_discharge = cp.sum(y, axis=0)
    station_load = data.L + aggregate_charge - aggregate_discharge

    price_buy = data.price_buy.reshape(1, data.T)
    price_sell = data.price_sell.reshape(1, data.T)
    throughput_penalty = data.rho.reshape(data.N, 1)
    objective = cp.sum(
        cp.multiply(price_buy, x)
        - cp.multiply(price_sell, y)
        + cp.multiply(throughput_penalty, x + y)
    ) + cp.sum(data.alpha * cp.square(station_load))

    constraints = [
        x <= data.X.reshape(data.N, 1) * data.availability,
        y <= data.Y.reshape(data.N, 1) * data.availability,
        E[:, 0] == data.E0,
        E[:, 1:] == E[:, :-1] + cp.multiply(data.eta_c.reshape(data.N, 1), x)
        - cp.multiply((1.0 / data.eta_d).reshape(data.N, 1), y),
        E[:, 1:] >= data.Emin.reshape(data.N, 1),
        E[:, 1:] <= data.Emax.reshape(data.N, 1),
        E[:, -1] >= data.Etarget,
        radius >= cp.max(cp.multiply(deviation, x), axis=0),
    ]
    for t in range(data.T):
        constraints.append(
            data.L[t]
            + cp.sum(cp.multiply(nominal[:, t], x[:, t]))
            + np.sqrt(2.0 * omega) * cp.norm(cp.multiply(sigma[:, t], x[:, t]), 2)
            + (omega / 3.0) * radius[t]
            <= data.M
        )

    problem = cp.Problem(cp.Minimize(objective), constraints)
    start_time = time.perf_counter()
    solve_conic_with_fallback(problem)
    return collect_fleet_result(
        data=data,
        problem=problem,
        x_var=x,
        y_var=y,
        E_var=E,
        start_time=start_time,
        delta=delta,
        policy=f"bernstein_eps_{epsilon:g}",
        policy_label=f"Bernstein eps={epsilon:g}",
        model_family="bernstein_socp",
        epsilon=epsilon,
        decision_variables=int(2 * data.N * data.T + data.N * (data.T + 1) + data.T),
    )


def collect_fleet_result(
    data: FleetData,
    problem: cp.Problem,
    x_var: cp.Variable,
    y_var: cp.Variable,
    E_var: cp.Variable,
    start_time: float,
    delta: float,
    policy: str,
    policy_label: str,
    model_family: str,
    gamma: float = np.nan,
    epsilon: float = np.nan,
    seed: int | float = np.nan,
    decision_variables: int | None = None,
) -> dict:
    solve_time_sec = time.perf_counter() - start_time
    status = problem.status
    base_result = {
        "case": data.name,
        "profile_date": data.profile_date,
        "n_vehicles": data.N,
        "seed": seed,
        "policy": policy,
        "policy_label": policy_label,
        "model_family": model_family,
        "gamma": gamma,
        "epsilon": epsilon,
        "delta": delta,
        "status": status,
        "objective": np.nan,
        "solve_time_sec": float(solve_time_sec),
        "decision_variables": decision_variables
        if decision_variables is not None
        else int(2 * data.N * data.T + data.N * (data.T + 1)),
    }
    if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        return base_result

    charge = clean_array(x_var.value).reshape(data.N, data.T)
    discharge = clean_array(y_var.value).reshape(data.N, data.T)
    soc = clean_array(E_var.value).reshape(data.N, data.T + 1)
    aggregate_charge_value = charge.sum(axis=0)
    aggregate_discharge_value = discharge.sum(axis=0)
    station_load_value = data.L + aggregate_charge_value - aggregate_discharge_value
    nominal_grid_margin = data.L + np.sum(fleet_u_matrix(data) * charge, axis=0) - data.M
    return {
        **base_result,
        "objective": float(problem.value),
        "x": charge,
        "y": discharge,
        "E": soc,
        "aggregate_charge": aggregate_charge_value,
        "aggregate_discharge": aggregate_discharge_value,
        "station_load": station_load_value,
        "total_charge": float(charge.sum()),
        "total_discharge": float(discharge.sum()),
        "mean_terminal_energy": float(soc[:, -1].mean()),
        "peak_load": float(station_load_value.max()),
        "economic_cost": float(
            np.sum(data.price_buy * aggregate_charge_value - data.price_sell * aggregate_discharge_value)
            + np.sum(data.rho.reshape(data.N, 1) * (charge + discharge))
        ),
        "load_penalty": float(np.sum(data.alpha * station_load_value**2)),
        "max_simultaneous_flow": float(np.max(np.minimum(charge, discharge))),
        "nominal_grid_margin": float(np.max(nominal_grid_margin)),
    }


def clean_array(value: np.ndarray | None, tol: float = 1.0e-5) -> np.ndarray:
    arr = np.asarray(value, dtype=float).ravel()
    arr[np.abs(arr) < tol] = 0.0
    return arr


def solve_with_fallback(problem: cp.Problem) -> None:
    try:
        problem.solve(
            solver=cp.OSQP,
            eps_abs=1.0e-8,
            eps_rel=1.0e-8,
            max_iter=100000,
            polish=True,
            verbose=False,
        )
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            problem.solve(solver=cp.CLARABEL, verbose=False)
    except cp.SolverError:
        problem.solve(solver=cp.CLARABEL, verbose=False)


def solve_fleet_with_fallback(problem: cp.Problem) -> None:
    """Use CLARABEL first for the optional fleet QPs; it is more stable on tight fleet cases."""
    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
    except cp.SolverError:
        problem.solve(
            solver=cp.OSQP,
            eps_abs=1.0e-6,
            eps_rel=1.0e-6,
            max_iter=50000,
            polish=False,
            verbose=False,
        )


def solve_conic_with_fallback(problem: cp.Problem) -> None:
    installed = set(cp.installed_solvers())
    clarabel_status = None
    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
        clarabel_status = problem.status
        if problem.status == cp.OPTIMAL:
            return
    except cp.SolverError:
        clarabel_status = None

    if cp.GUROBI in installed:
        try:
            problem.solve(solver=cp.GUROBI, verbose=False)
            if problem.status == cp.OPTIMAL:
                return
        except cp.SolverError:
            pass

    if clarabel_status == cp.OPTIMAL_INACCURATE:
        return
    problem.solve(solver=cp.SCS, eps=1.0e-5, max_iters=50000, verbose=False)


def solve_mixed_integer_with_fallback(problem: cp.Problem, time_limit_sec: float | None = None) -> None:
    installed = set(cp.installed_solvers())
    for solver in (cp.GUROBI, cp.CPLEX, cp.SCIPY):
        if solver not in installed:
            continue
        try:
            if solver == cp.GUROBI:
                gurobi_options = {
                    "solver": solver,
                    "verbose": False,
                    "MIPGap": 1.0e-8,
                    "FeasibilityTol": 1.0e-8,
                    "IntFeasTol": 1.0e-8,
                    "OptimalityTol": 1.0e-8,
                }
                if time_limit_sec is not None:
                    gurobi_options["TimeLimit"] = float(time_limit_sec)
                problem.solve(**gurobi_options)
            elif solver == cp.CPLEX:
                problem.solve(solver=solver, verbose=False)
            else:
                problem.solve(solver=solver, verbose=False, scipy_options={"method": "highs"})
            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                return
        except cp.SolverError:
            continue
    raise cp.SolverError("No working mixed-integer solver found")


def monte_carlo_violation(data: EVData, solution: dict, delta: float, samples: int = 10000) -> dict:
    if "x" not in solution:
        return {"violation_probability": np.nan, "max_violation": np.nan, "mean_positive_violation": np.nan}

    lower, upper = uncertainty_box(data, delta)
    rng = np.random.default_rng(20260504)
    sampled_u = rng.uniform(lower, upper, size=(samples, data.T))
    grid_margin = data.L + sampled_u * solution["x"] - data.M
    per_sample_violation = np.max(grid_margin, axis=1)
    positive_violation = np.maximum(per_sample_violation, 0.0)

    return {
        "violation_probability": float(np.mean(per_sample_violation > 1.0e-6)),
        "max_violation": float(np.max(per_sample_violation)),
        "mean_positive_violation": float(np.mean(positive_violation)),
    }


def exact_violation_probability(data: EVData, solution: dict, delta: float, tol: float = 1.0e-5) -> float:
    """Exact probability of at least one grid violation for independent uniform u_t."""
    if "x" not in solution:
        return np.nan

    lower, upper = uncertainty_box(data, delta)
    no_violation_probability = 1.0
    for baseload, charge, lo, hi in zip(data.L, solution["x"], lower, upper):
        if charge <= 1.0e-9:
            slot_probability = 1.0 if baseload - data.M <= tol else 0.0
        elif np.isclose(hi, lo):
            slot_probability = 1.0 if baseload + hi * charge - data.M <= tol else 0.0
        else:
            threshold = (data.M + tol - baseload) / charge
            slot_probability = np.clip((threshold - lo) / (hi - lo), 0.0, 1.0)
        no_violation_probability *= float(slot_probability)
    return float(1.0 - no_violation_probability)


def fleet_monte_carlo_violation(
    data: FleetData,
    solution: dict,
    delta: float,
    samples: int = 10000,
    seed: int = 20260504,
) -> dict:
    if "x" not in solution:
        return {"violation_probability": np.nan, "max_violation": np.nan, "mean_positive_violation": np.nan}

    lower, upper = fleet_uncertainty_box(data, delta)
    rng = np.random.default_rng(seed)
    sampled_u = rng.uniform(lower, upper, size=(samples, data.N, data.T))
    grid_margin = data.L.reshape(1, data.T) + np.sum(sampled_u * solution["x"].reshape(1, data.N, data.T), axis=1) - data.M
    per_sample_violation = np.max(grid_margin, axis=1)
    positive_violation = np.maximum(per_sample_violation, 0.0)
    return {
        "violation_probability": float(np.mean(per_sample_violation > 1.0e-6)),
        "max_violation": float(np.max(per_sample_violation)),
        "mean_positive_violation": float(np.mean(positive_violation)),
    }


def fleet_exact_violation_probability(data: FleetData, solution: dict, delta: float, tol: float = 1.0e-5) -> float:
    """Exact grid violation probability for fixed aggregate fleet charging under uniform u_t."""
    if "aggregate_charge" not in solution:
        return np.nan

    lower, upper = fleet_uncertainty_box(data, delta)
    if lower.ndim != 1:
        return np.nan
    no_violation_probability = 1.0
    for baseload, charge, lo, hi in zip(data.L, solution["aggregate_charge"], lower, upper):
        if charge <= 1.0e-9:
            slot_probability = 1.0 if baseload - data.M <= tol else 0.0
        elif np.isclose(hi, lo):
            slot_probability = 1.0 if baseload + hi * charge - data.M <= tol else 0.0
        else:
            threshold = (data.M + tol - baseload) / charge
            slot_probability = np.clip((threshold - lo) / (hi - lo), 0.0, 1.0)
        no_violation_probability *= float(slot_probability)
    return float(1.0 - no_violation_probability)


def worst_case_grid_margin(data: EVData, solution: dict, delta: float) -> float:
    if "x" not in solution:
        return np.nan
    _, upper = uncertainty_box(data, delta)
    return float(np.max(data.L + upper * solution["x"] - data.M))


def fleet_worst_case_grid_margin(data: FleetData, solution: dict, delta: float) -> float:
    if "x" not in solution:
        return np.nan
    _, upper = fleet_uncertainty_box(data, delta)
    return float(np.max(data.L + np.sum(upper * solution["x"], axis=0) - data.M))


def row_from_solution(data: EVData, solution: dict, delta: float) -> dict:
    violation = monte_carlo_violation(data, solution, delta=delta)
    return {
        "case": data.name,
        "profile_date": data.profile_date,
        "policy": solution["policy"],
        "policy_label": solution["policy_label"],
        "delta": delta,
        "status": solution["status"],
        "objective": solution.get("objective", np.nan),
        "economic_cost": solution.get("economic_cost", np.nan),
        "load_penalty": solution.get("load_penalty", np.nan),
        "paper_like_cost": solution.get("paper_like_cost", np.nan),
        "total_charge": solution.get("total_charge", np.nan),
        "total_discharge": solution.get("total_discharge", np.nan),
        "terminal_energy": solution.get("terminal_energy", np.nan),
        "peak_load": solution.get("peak_load", np.nan),
        "nominal_grid_margin": solution.get("nominal_grid_margin", np.nan),
        "max_simultaneous_flow": solution.get("max_simultaneous_flow", np.nan),
        "solve_time_sec": solution.get("solve_time_sec", np.nan),
        "exact_violation_probability": exact_violation_probability(data, solution, delta=delta),
        "worst_case_margin": worst_case_grid_margin(data, solution, delta=delta),
        "violation_probability": violation["violation_probability"],
        "max_violation": violation["max_violation"],
        "mean_positive_violation": violation["mean_positive_violation"],
    }


def row_from_fleet_solution(data: FleetData, solution: dict, delta: float) -> dict:
    violation = fleet_monte_carlo_violation(data, solution, delta=delta)
    return {
        "case": data.name,
        "profile_date": data.profile_date,
        "fleet_source": data.source,
        "acn_site": data.acn_site,
        "acn_cache": data.acn_cache,
        "horizon_date": data.horizon_date,
        "source_sessions": data.source_sessions,
        "grid_limit": data.M,
        "n_vehicles": data.N,
        "seed": solution.get("seed", np.nan),
        "policy": solution["policy"],
        "policy_label": solution["policy_label"],
        "model_family": solution.get("model_family", "protected_box"),
        "gamma": solution.get("gamma", np.nan),
        "epsilon": solution.get("epsilon", np.nan),
        "delta": delta,
        "status": solution["status"],
        "objective": solution.get("objective", np.nan),
        "economic_cost": solution.get("economic_cost", np.nan),
        "load_penalty": solution.get("load_penalty", np.nan),
        "total_charge": solution.get("total_charge", np.nan),
        "total_discharge": solution.get("total_discharge", np.nan),
        "mean_terminal_energy": solution.get("mean_terminal_energy", np.nan),
        "peak_load": solution.get("peak_load", np.nan),
        "nominal_grid_margin": solution.get("nominal_grid_margin", np.nan),
        "max_simultaneous_flow": solution.get("max_simultaneous_flow", np.nan),
        "solve_time_sec": solution.get("solve_time_sec", np.nan),
        "decision_variables": solution.get("decision_variables", np.nan),
        "exact_violation_probability": fleet_exact_violation_probability(data, solution, delta=delta),
        "violation_probability": violation["violation_probability"],
        "max_violation": violation["max_violation"],
        "mean_positive_violation": violation["mean_positive_violation"],
        "worst_case_margin": fleet_worst_case_grid_margin(data, solution, delta=delta),
    }

def run_delta_sensitivity(data: EVData, deltas: list[float]) -> tuple[pd.DataFrame, dict]:
    rows = []
    solutions = {}
    for delta in deltas:
        for policy in POLICIES:
            solution = solve_model(data, delta=delta, policy=policy)
            solutions[(data.name, policy.name, delta)] = solution
            rows.append(row_from_solution(data, solution, delta))
    return pd.DataFrame(rows), solutions


def run_case_benchmark(cases: list[EVData], delta: float) -> tuple[pd.DataFrame, dict]:
    rows = []
    solutions = {}
    for data in cases:
        for policy in POLICIES:
            solution = solve_model(data, delta=delta, policy=policy)
            solutions[(data.name, policy.name, delta)] = solution
            rows.append(row_from_solution(data, solution, delta))
    return pd.DataFrame(rows), solutions


def run_parameter_sensitivity(data: EVData, delta: float) -> pd.DataFrame:
    rows = []
    experiments = [
        ("M", [475.0, 477.5, 480.0, 485.0, 490.0]),
        ("Etarget", [55.0, 60.0, 65.0, 70.0, 75.0]),
    ]
    for parameter, values in experiments:
        for value in values:
            varied_data = replace(data, name=f"{parameter}_{value:g}", **{parameter: value})
            for policy in POLICIES:
                solution = solve_model(varied_data, delta=delta, policy=policy)
                row = row_from_solution(varied_data, solution, delta)
                row["parameter"] = parameter
                row["parameter_value"] = value
                rows.append(row)
    return pd.DataFrame(rows)


def run_multi_day_benchmark(dates: list[str] | tuple[str, ...] | None, delta: float) -> pd.DataFrame:
    rows = []
    for data in build_multi_day_cases(dates):
        for policy in POLICIES:
            solution = solve_model(data, delta=delta, policy=policy)
            rows.append(row_from_solution(data, solution, delta))
    return pd.DataFrame(rows)


def run_multi_ev_scaling(
    base: EVData,
    vehicle_counts: list[int] | tuple[int, ...] = (10, 20, 30),
    delta: float = 0.5,
    policies: list[Policy] | tuple[Policy, ...] | None = None,
    fleet_config: FleetExperimentConfig | None = None,
    acn_sessions: pd.DataFrame | None = None,
    horizon_dates: list[str] | None = None,
) -> pd.DataFrame:
    """Run an optional fleet-level extension without changing the single-EV pipeline."""
    if policies is None:
        policies = tuple(policy for policy in POLICIES if policy.name in {"deterministic", "quantile_75", "robust"})
    fleet_config = fleet_config or FleetExperimentConfig()

    rows = []
    for n_vehicles in vehicle_counts:
        data = build_fleet_case_for_experiment(
            base,
            n_vehicles=n_vehicles,
            seed=FLEET_SEEDS[0],
            config=fleet_config,
            acn_sessions=acn_sessions,
            horizon_dates=horizon_dates,
        )
        data = replace(data, M=FLEET_FIXED_GRID_LIMIT)
        for policy in policies:
            solution = solve_fleet_model(data, delta=delta, policy=policy)
            rows.append(row_from_fleet_solution(data, solution, delta=delta))
    return pd.DataFrame(rows)


def run_budgeted_robust_grid(
    base: EVData,
    vehicle_counts: list[int] | tuple[int, ...] = FLEET_COUNTS,
    seeds: list[int] | tuple[int, ...] = FLEET_SEEDS,
    gammas: list[float] | tuple[float, ...] = GAMMA_LEVELS,
    delta: float = 0.5,
    fleet_config: FleetExperimentConfig | None = None,
    acn_sessions: pd.DataFrame | None = None,
    horizon_dates: list[str] | None = None,
) -> pd.DataFrame:
    fleet_config = fleet_config or FleetExperimentConfig()
    rows = []
    for seed in seeds:
        for n_vehicles in vehicle_counts:
            data = build_fleet_case_for_experiment(
                base,
                n_vehicles=n_vehicles,
                seed=seed,
                config=fleet_config,
                acn_sessions=acn_sessions,
                horizon_dates=horizon_dates,
            )
            data = replace(data, M=FLEET_FIXED_GRID_LIMIT)
            for gamma in gammas:
                solution = solve_fleet_budgeted_model(data, delta=delta, gamma=min(gamma, n_vehicles))
                solution["seed"] = seed
                rows.append(row_from_fleet_solution(data, solution, delta=delta))
    return pd.DataFrame(rows)


def run_bernstein_chance_grid(
    base: EVData,
    vehicle_counts: list[int] | tuple[int, ...] = FLEET_COUNTS,
    seeds: list[int] | tuple[int, ...] = FLEET_SEEDS,
    epsilons: list[float] | tuple[float, ...] = EPSILON_LEVELS,
    delta: float = 0.5,
    fleet_config: FleetExperimentConfig | None = None,
    acn_sessions: pd.DataFrame | None = None,
    horizon_dates: list[str] | None = None,
) -> pd.DataFrame:
    fleet_config = fleet_config or FleetExperimentConfig()
    rows = []
    for seed in seeds:
        for n_vehicles in vehicle_counts:
            data = build_fleet_case_for_experiment(
                base,
                n_vehicles=n_vehicles,
                seed=seed,
                config=fleet_config,
                acn_sessions=acn_sessions,
                horizon_dates=horizon_dates,
            )
            data = replace(data, M=FLEET_FIXED_GRID_LIMIT)
            for epsilon in epsilons:
                solution = solve_fleet_bernstein_model(data, delta=delta, epsilon=epsilon)
                solution["seed"] = seed
                rows.append(row_from_fleet_solution(data, solution, delta=delta))
    return pd.DataFrame(rows)


def run_fleet_multi_seed_benchmark(
    base: EVData,
    vehicle_counts: list[int] | tuple[int, ...] = FLEET_COUNTS,
    seeds: list[int] | tuple[int, ...] = FLEET_SEEDS,
    delta: float = 0.5,
    fleet_config: FleetExperimentConfig | None = None,
    acn_sessions: pd.DataFrame | None = None,
    horizon_dates: list[str] | None = None,
) -> pd.DataFrame:
    fleet_config = fleet_config or FleetExperimentConfig()
    rows = []
    policies = tuple(policy for policy in POLICIES if policy.name in {"deterministic", "quantile_75", "robust"})
    for seed in seeds:
        for n_vehicles in vehicle_counts:
            data = build_fleet_case_for_experiment(
                base,
                n_vehicles=n_vehicles,
                seed=seed,
                config=fleet_config,
                acn_sessions=acn_sessions,
                horizon_dates=horizon_dates,
            )
            data = replace(data, M=FLEET_FIXED_GRID_LIMIT)
            for policy in policies:
                solution = solve_fleet_model(data, delta=delta, policy=policy)
                solution["seed"] = seed
                rows.append(row_from_fleet_solution(data, solution, delta=delta))
    return pd.DataFrame(rows)


def run_binary_validation(
    base: EVData,
    delta: float = 0.5,
    cases: list[EVData] | None = None,
    time_limit_sec: float | None = None,
) -> pd.DataFrame:
    rows = []
    validation_policies = tuple(policy for policy in POLICIES if policy.name in {"deterministic", "quantile_75", "robust"})
    validation_cases = build_miqp_validation_cases(base) if cases is None else cases
    for data in validation_cases:
        for policy in validation_policies:
            relaxed = solve_model(data, delta=delta, policy=policy)
            binary = solve_binary_complementarity_model(
                data,
                delta=delta,
                policy=policy,
                time_limit_sec=time_limit_sec,
            )
            relaxed_row = row_from_solution(data, relaxed, delta=delta)
            binary_row = row_from_solution(data, binary, delta=delta)
            relaxed_row["validation_model"] = "convex_relaxation"
            binary_row["validation_model"] = "miqp_complementarity"
            relaxed_row["validation_policy"] = policy.name
            binary_row["validation_policy"] = policy.name
            if np.isfinite(relaxed_row["objective"]) and np.isfinite(binary_row["objective"]):
                objective_gap = binary_row["objective"] - relaxed_row["objective"]
                charge_l1_gap = np.sum(np.abs(binary["x"] - relaxed["x"]))
                discharge_l1_gap = np.sum(np.abs(binary["y"] - relaxed["y"]))
            else:
                objective_gap = np.nan
                charge_l1_gap = np.nan
                discharge_l1_gap = np.nan
            relaxed_row["objective_gap_vs_relaxation"] = 0.0
            relaxed_row["charge_l1_gap_vs_relaxation"] = 0.0
            relaxed_row["discharge_l1_gap_vs_relaxation"] = 0.0
            binary_row["objective_gap_vs_relaxation"] = objective_gap
            binary_row["charge_l1_gap_vs_relaxation"] = charge_l1_gap
            binary_row["discharge_l1_gap_vs_relaxation"] = discharge_l1_gap
            rows.extend([relaxed_row, binary_row])
    return pd.DataFrame(rows)


def summarize_binary_validation(validation: pd.DataFrame) -> pd.DataFrame:
    """Compact table with one row per case-policy MIQP comparison."""
    relaxed = validation[validation["validation_model"] == "convex_relaxation"].copy()
    miqp = validation[validation["validation_model"] == "miqp_complementarity"].copy()
    relaxed = relaxed.rename(
        columns={
            "objective": "relaxed_objective",
            "max_simultaneous_flow": "relaxed_max_simultaneous_flow",
            "worst_case_margin": "relaxed_worst_case_margin",
            "solve_time_sec": "relaxed_solve_time_sec",
        }
    )
    miqp = miqp.rename(
        columns={
            "objective": "miqp_objective",
            "max_simultaneous_flow": "miqp_max_simultaneous_flow",
            "solve_time_sec": "miqp_solve_time_sec",
            "status": "miqp_status",
        }
    )
    summary = miqp.merge(
        relaxed[
            [
                "case",
                "validation_policy",
                "relaxed_objective",
                "relaxed_max_simultaneous_flow",
                "relaxed_worst_case_margin",
                "relaxed_solve_time_sec",
            ]
        ],
        on=["case", "validation_policy"],
        how="left",
    )
    return summary[
        [
            "case",
            "validation_policy",
            "policy_label",
            "miqp_status",
            "relaxed_objective",
            "miqp_objective",
            "objective_gap_vs_relaxation",
            "charge_l1_gap_vs_relaxation",
            "discharge_l1_gap_vs_relaxation",
            "relaxed_max_simultaneous_flow",
            "miqp_max_simultaneous_flow",
            "relaxed_worst_case_margin",
            "relaxed_solve_time_sec",
            "miqp_solve_time_sec",
        ]
    ].sort_values(["case", "validation_policy"])


def run_q1_sensitivity(
    base: EVData,
    delta: float = 0.5,
    fleet_config: FleetExperimentConfig | None = None,
    acn_sessions: pd.DataFrame | None = None,
    horizon_dates: list[str] | None = None,
) -> pd.DataFrame:
    fleet_config = fleet_config or FleetExperimentConfig()
    rows = []
    scalar_experiments = [
        ("rho", SENSITIVITY_RHO),
        ("alpha", SENSITIVITY_ALPHA),
        ("delta", SENSITIVITY_DELTAS),
        ("epsilon", EPSILON_LEVELS),
    ]
    for parameter, values in scalar_experiments:
        for value in values:
            varied_data = base
            varied_delta = delta
            policy = next(p for p in POLICIES if p.name == "quantile_75")
            if parameter == "rho":
                varied_data = replace(base, name=f"rho_{value:g}", rho=float(value))
                policy = next(p for p in POLICIES if p.name == "robust")
            elif parameter == "alpha":
                varied_data = replace(base, name=f"alpha_{value:g}", alpha=float(value))
                policy = next(p for p in POLICIES if p.name == "robust")
            elif parameter == "delta":
                varied_data = replace(base, name=f"delta_{value:g}")
                varied_delta = float(value)
                policy = next(p for p in POLICIES if p.name == "robust")
            elif parameter == "epsilon":
                varied_data = replace(base, name=f"epsilon_{value:g}")
                policy = Policy(
                    f"chance_eps_{value:g}",
                    f"Slotwise quantile eps={value:g}",
                    chance_epsilon=float(value),
                )
            solution = solve_model(varied_data, delta=varied_delta, policy=policy)
            row = row_from_solution(varied_data, solution, delta=varied_delta)
            row["parameter"] = parameter
            row["parameter_value"] = value
            rows.append(row)

    for gamma in GAMMA_LEVELS:
        gamma_rows = []
        for seed in FLEET_SEEDS:
            fleet_base = build_fleet_case_for_experiment(
                base,
                n_vehicles=20,
                seed=seed,
                config=fleet_config,
                acn_sessions=acn_sessions,
                horizon_dates=horizon_dates,
            )
            solution = solve_fleet_budgeted_model(fleet_base, delta=delta, gamma=min(gamma, fleet_base.N))
            solution["seed"] = seed
            gamma_rows.append(row_from_fleet_solution(fleet_base, solution, delta=delta))
        gamma_frame = pd.DataFrame(gamma_rows)
        row = gamma_frame.iloc[0].copy()
        numeric_columns = gamma_frame.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            row[column] = gamma_frame[column].mean()
        row["status"] = "optimal" if set(gamma_frame["status"]) == {cp.OPTIMAL} else "mixed"
        row["seed"] = np.nan
        row["parameter"] = "gamma"
        row["parameter_value"] = gamma
        rows.append(row)
    return pd.DataFrame(rows)


def run_theorem_condition_ablation(base: EVData, delta: float = 0.5) -> pd.DataFrame:
    """Lightweight scalar ablation for objective and protection-condition robustness."""
    rows = []
    experiments: list[tuple[str, str, float, EVData, float]] = []

    for rho in SENSITIVITY_RHO:
        experiments.append(("rho", rf"$\rho={rho:g}$", float(rho), replace(base, name=f"ablation_rho_{rho:g}", rho=float(rho)), delta))

    for eta_c, eta_d in ABLATION_EFFICIENCY_PAIRS:
        product = float(eta_c * eta_d)
        experiments.append(
            (
                "efficiency_product",
                rf"$\eta_c\eta_d={product:.3f}$",
                product,
                replace(base, name=f"ablation_eff_{product:.3f}", eta_c=float(eta_c), eta_d=float(eta_d)),
                delta,
            )
        )

    for spread in ABLATION_PRICE_SPREADS:
        experiments.append(
            (
                "price_spread",
                rf"$\pi^d/\pi^c={spread:.2f}$",
                float(spread),
                replace(base, name=f"ablation_spread_{spread:.2f}", price_sell=float(spread) * base.price_buy),
                delta,
            )
        )

    for capacity in ABLATION_CAPACITY_LEVELS:
        experiments.append(
            (
                "capacity",
                rf"$M={capacity:.0f}$",
                float(capacity),
                replace(base, name=f"ablation_M_{capacity:.0f}", M=float(capacity)),
                delta,
            )
        )

    for radius in SENSITIVITY_DELTAS:
        experiments.append(("uncertainty_radius", rf"$\delta={radius:.1f}$", float(radius), replace(base, name=f"ablation_delta_{radius:.1f}"), float(radius)))

    for target in ABLATION_TERMINAL_TARGETS:
        experiments.append(
            (
                "terminal_target",
                rf"$E^{{tar}}={target:.0f}$",
                float(target),
                replace(base, name=f"ablation_target_{target:.0f}", Etarget=float(target)),
                delta,
            )
        )

    for parameter, level_label, level_value, varied_data, varied_delta in experiments:
        for policy in POLICIES:
            solution = solve_model(varied_data, delta=varied_delta, policy=policy)
            row = row_from_solution(varied_data, solution, delta=varied_delta)
            row["ablation_parameter"] = parameter
            row["ablation_level_label"] = level_label
            row["ablation_level_value"] = level_value
            row["efficiency_product"] = varied_data.eta_c * varied_data.eta_d
            row["price_spread_ratio"] = float(np.mean(varied_data.price_sell / np.maximum(varied_data.price_buy, 1.0e-9)))
            row["capacity_M"] = varied_data.M
            row["terminal_target"] = varied_data.Etarget
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_multi_day_results(multi_day_summary: pd.DataFrame) -> pd.DataFrame:
    policy_order = {policy.name: idx for idx, policy in enumerate(POLICIES)}
    summary = (
        multi_day_summary.groupby(["policy", "policy_label"], as_index=False)
        .agg(
            days=("profile_date", "nunique"),
            mean_net_cost=("economic_cost", "mean"),
            std_net_cost=("economic_cost", "std"),
            mean_total_discharge=("total_discharge", "mean"),
            mean_exact_violation_probability=("exact_violation_probability", "mean"),
            max_exact_violation_probability=("exact_violation_probability", "max"),
            max_worst_case_margin=("worst_case_margin", "max"),
            mean_solve_time_sec=("solve_time_sec", "mean"),
        )
        .sort_values("policy", key=lambda column: column.map(policy_order))
        .reset_index(drop=True)
    )
    return summary


def schedule_frame(data: EVData, solution: dict) -> pd.DataFrame:
    hours = np.arange(1, data.T + 1)
    return pd.DataFrame(
        {
            "hour": hours,
            "available": data.availability,
            "baseload": data.L,
            "buy_price": data.price_buy,
            "sell_price": data.price_sell,
            "charge": solution["x"],
            "discharge": solution["y"],
            "net_charge": solution["p"],
            "state_of_charge_start": solution["E"][:-1],
            "state_of_charge_end": solution["E"][1:],
            "station_load": solution["station_load"],
        }
    )


def plot_schedule(data: EVData, solutions: dict, delta: float, out_dir: Path) -> None:
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "legend.fontsize": 9})
    hours = np.arange(1, data.T + 1)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10.5), sharex=True)

    axes[0].plot(hours, data.L, color="0.25", label="baseload")
    axes[0].axhline(data.M, color="black", linewidth=1.0, label="grid limit")

    plotted = ["deterministic", "quantile_75", "robust"]
    styles = {"deterministic": "--", "quantile_75": "-.", "robust": "-"}
    for policy_name in plotted:
        solution = solutions[(data.name, policy_name, delta)]
        axes[1].plot(hours, solution["x"], styles[policy_name], label=f"{policy_name} charge")
        axes[1].plot(hours, solution["y"], styles[policy_name], label=f"{policy_name} discharge")
        axes[2].plot(np.arange(0, data.T + 1), solution["E"], styles[policy_name], label=policy_name)
        axes[3].plot(hours, solution["station_load"], styles[policy_name], label=policy_name)

    axes[0].set_ylabel("Baseload")
    axes[1].set_ylabel("Energy (kWh)")
    axes[2].set_ylabel("SOC (kWh)")
    axes[3].set_ylabel("Station load")
    axes[3].set_xlabel("Hour")
    axes[3].axhline(data.M, color="black", linewidth=1.0, label="grid limit")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle(f"EV charging and V2G schedules at delta={delta:.0%}")
    fig.tight_layout()
    fig.savefig(out_dir / f"schedules_delta_{int(delta * 100):02d}.png", dpi=200)
    plt.close(fig)


def plot_delta_sensitivity(summary: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12})
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    axes = axes.ravel()
    labels = {
        "deterministic": "Deterministic",
        "quantile_75": "Slotwise Q75",
        "robust": "Robust",
    }

    metrics = [
        ("economic_cost", "Net EUR-equivalent cost"),
        ("exact_violation_probability", "Exact violation probability"),
    ]

    for ax, (column, label) in zip(axes, metrics):
        for policy in [p.name for p in POLICIES]:
            frame = summary[summary["policy"] == policy].sort_values("delta")
            if frame.empty:
                continue
            ax.plot(frame["delta"], frame[column], marker="o", markersize=6.5, linewidth=2.8, label=labels.get(policy, policy))
        ax.set_xlabel(r"Uncertainty level $\delta$")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.28, linewidth=0.8)
        ax.legend(loc="upper left", ncol=1, frameon=False, handlelength=1.8)

    fig.tight_layout()
    fig.savefig(out_dir / "delta_sensitivity_summary.png", dpi=200)
    plt.close(fig)


def plot_case_benchmark(case_summary: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "legend.fontsize": 9})
    complete = case_summary[case_summary["status"].isin([cp.OPTIMAL, cp.OPTIMAL_INACCURATE])].copy()
    policies = [p.name for p in POLICIES]
    cases = list(dict.fromkeys(complete["case"]))
    x = np.arange(len(cases))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharex=True)
    for i, policy in enumerate(policies):
        frame = complete[complete["policy"] == policy].set_index("case").reindex(cases)
        offset = (i - (len(policies) - 1) / 2) * width
        axes[0].bar(x + offset, frame["economic_cost"], width=width, label=policy)
        axes[1].bar(x + offset, frame["exact_violation_probability"], width=width, label=policy)

    axes[0].set_ylabel("Net EUR-equivalent cost")
    axes[1].set_ylabel("Exact violation probability")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([display_case_name(case) for case in cases], rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "case_benchmark_delta_50.png", dpi=200)
    plt.close(fig)


def plot_multi_day_benchmark(multi_day_summary: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10})
    complete = multi_day_summary[multi_day_summary["status"].isin([cp.OPTIMAL, cp.OPTIMAL_INACCURATE])].copy()
    policy_order = [p.name for p in POLICIES]
    policy_labels = ["Deterministic", "Slotwise Q75", "Robust"]
    positions = np.arange(len(policy_order))
    rng = np.random.default_rng(20260512)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2), sharex=True)
    metrics = [
        ("economic_cost", "Net EUR-equivalent cost"),
        ("exact_violation_probability", "Exact violation probability"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        data_by_policy = [
            complete[complete["policy"] == policy][metric].dropna().to_numpy()
            for policy in policy_order
        ]
        ax.boxplot(
            data_by_policy,
            positions=positions,
            widths=0.58,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "0.90", "edgecolor": "0.25"},
            medianprops={"color": "black", "linewidth": 1.4},
        )
        for pos, values in zip(positions, data_by_policy):
            if len(values) == 0:
                continue
            sample_size = min(len(values), 120)
            sampled = rng.choice(values, size=sample_size, replace=False)
            jitter = rng.normal(0.0, 0.055, size=sample_size)
            ax.scatter(np.full(sample_size, pos) + jitter, sampled, s=13, alpha=0.20, color="tab:blue", linewidths=0)
        ax.set_ylabel(ylabel)
        ax.set_xticks(positions)
        ax.set_xticklabels(policy_labels, rotation=0)
        ax.grid(True, axis="y", alpha=0.28, linewidth=0.8)

    axes[0].set_xlabel("Policy")
    axes[1].set_xlabel("Policy")
    axes[1].set_yscale("symlog", linthresh=1.0e-3)
    fig.tight_layout()
    fig.savefig(out_dir / "multi_day_benchmark_delta_50.png", dpi=200)
    plt.close(fig)


def plot_parameter_sensitivity(parameter_summary: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "legend.fontsize": 9})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    axes = axes.ravel()
    specs = [
        ("M", "exact_violation_probability", "Grid limit M", "Exact violation probability"),
        ("M", "economic_cost", "Grid limit M", "Net EUR-equivalent cost"),
        ("Etarget", "total_discharge", "Terminal target", "Total V2G discharge"),
        ("Etarget", "economic_cost", "Terminal target", "Net EUR-equivalent cost"),
    ]

    for ax, (parameter, metric, xlabel, ylabel) in zip(axes, specs):
        frame = parameter_summary[parameter_summary["parameter"] == parameter]
        for policy, policy_frame in frame.groupby("policy"):
            ax.plot(policy_frame["parameter_value"], policy_frame[metric], marker="o", linewidth=2.0, label=policy)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "parameter_sensitivity_delta_50.png", dpi=200)
    plt.close(fig)


def plot_multi_ev_scaling(fleet_summary: pd.DataFrame, out_dir: Path) -> None:
    if fleet_summary.empty:
        return
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "legend.fontsize": 9})
    complete = fleet_summary[fleet_summary["status"].isin([cp.OPTIMAL, cp.OPTIMAL_INACCURATE])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)

    for policy, frame in complete.groupby("policy"):
        frame = frame.sort_values("n_vehicles")
        reliability = frame["exact_violation_probability"].fillna(frame["violation_probability"])
        axes[0].plot(frame["n_vehicles"], frame["solve_time_sec"], marker="o", linewidth=2.0, label=policy)
        axes[1].plot(
            frame["n_vehicles"],
            reliability,
            marker="o",
            linewidth=2.0,
            label=policy,
        )

    axes[0].set_ylabel("Solve time (s)")
    axes[1].set_ylabel("Exact violation probability")
    for ax in axes:
        ax.set_xlabel("Number of EVs")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "multi_ev_scaling_delta_50.png", dpi=200)
    plt.close(fig)


def summarize_fleet_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = frame.copy()
    work["reported_violation_probability"] = work["exact_violation_probability"].fillna(work["violation_probability"])
    group_cols = ["model_family", "policy", "policy_label", "n_vehicles"]
    optional_cols = [col for col in ("gamma", "epsilon") if col in work.columns and work[col].notna().any()]
    group_cols += optional_cols
    return (
        work.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_objective=("objective", "mean"),
            std_objective=("objective", "std"),
            mean_net_cost=("economic_cost", "mean"),
            std_net_cost=("economic_cost", "std"),
            mean_total_discharge=("total_discharge", "mean"),
            mean_violation_probability=("reported_violation_probability", "mean"),
            max_violation_probability=("reported_violation_probability", "max"),
            mean_worst_case_margin=("worst_case_margin", "mean"),
            mean_solve_time_sec=("solve_time_sec", "mean"),
        )
        .reset_index(drop=True)
    )


def plot_q1_fleet_multiseed(fleet_summary: pd.DataFrame, out_dir: Path) -> None:
    if fleet_summary.empty:
        return
    work = fleet_summary.copy()
    if "mean_violation_probability" in work.columns:
        grouped = work.rename(
            columns={
                "mean_violation_probability": "mean_violation",
                "max_violation_probability": "max_violation",
            }
        ).copy()
    else:
        work["reported_violation_probability"] = work["exact_violation_probability"].fillna(work["violation_probability"])
        grouped = (
            work.groupby(["policy", "n_vehicles"], as_index=False)
            .agg(
                mean_net_cost=("economic_cost", "mean"),
                std_net_cost=("economic_cost", "std"),
                mean_total_discharge=("total_discharge", "mean"),
                mean_violation=("reported_violation_probability", "mean"),
                max_violation=("reported_violation_probability", "max"),
            )
            .sort_values(["policy", "n_vehicles"])
        )

    plt.rcParams.update({"font.size": 15, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 11})
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.0), sharex=True)
    labels = {"deterministic": "Deterministic", "quantile_75": "Slotwise Q75", "robust": "Robust"}
    for policy in ("deterministic", "quantile_75", "robust"):
        frame = grouped[grouped["policy"] == policy].sort_values("n_vehicles")
        if frame.empty:
            continue
        label = labels.get(policy, policy)
        axes[0].plot(frame["n_vehicles"], frame["mean_net_cost"], marker="o", markersize=6.5, linewidth=2.6, label=label)
        axes[1].plot(frame["n_vehicles"], frame["mean_total_discharge"], marker="o", markersize=6.5, linewidth=2.6, label=label)
        axes[2].plot(frame["n_vehicles"], frame["mean_violation"], marker="o", markersize=6.5, linewidth=2.6, label=label)
        axes[2].fill_between(frame["n_vehicles"], frame["mean_violation"], frame["max_violation"], alpha=0.12)

    axes[0].set_ylabel("Mean EUR-equivalent cost")
    axes[1].set_ylabel("Mean V2G discharge (kWh)")
    axes[2].set_ylabel("Violation probability")
    axes[2].set_ylim(bottom=0.0)
    for ax in axes:
        ax.set_xlabel("Number of EVs")
        ax.grid(True, alpha=0.28, linewidth=0.8)
        ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    fig.savefig(out_dir / "q1_fleet_multiseed_validation.png", dpi=200)
    plt.close(fig)


def plot_q1_budgeted_gamma(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12})
    frame = summary[summary["n_vehicles"] == 20].copy()
    if frame.empty:
        frame = summary.copy()
    frame["beta"] = frame["gamma"] / frame["n_vehicles"]
    frame = frame.sort_values("beta")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    axes[0].plot(frame["beta"], frame["mean_violation_probability"], marker="o", linewidth=2.2, color="#2B5C8A")
    axes[0].set_title("A. Reliability")
    axes[0].set_xlabel(r"Protection budget $\beta=B/N$")
    axes[0].set_ylabel("Mean violation probability")
    viol_top = max(0.30, float(frame["mean_violation_probability"].max()) * 1.15)
    axes[0].set_ylim(0.0, viol_top)
    axes[0].grid(True, alpha=0.28, linewidth=0.8)

    axes[1].plot(frame["beta"], frame["mean_net_cost"], marker="s", linewidth=2.2, color="#3A7D44")
    axes[1].set_title("B. Economic cost")
    axes[1].set_xlabel(r"Protection budget $\beta=B/N$")
    axes[1].set_ylabel("Mean EUR-equivalent cost")
    axes[1].grid(True, alpha=0.28, linewidth=0.8)

    fig.savefig(out_dir / "q1_budgeted_cost_reliability_frontier.png", dpi=200)
    plt.close(fig)


def plot_q1_bernstein_epsilon(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "legend.fontsize": 9})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True)
    for n_vehicles, frame in summary.groupby("n_vehicles"):
        frame = frame.sort_values("epsilon")
        axes[0].plot(frame["epsilon"], frame["mean_net_cost"], marker="o", linewidth=2.0, label=f"N={n_vehicles}")
        axes[1].plot(
            frame["epsilon"],
            frame["mean_total_discharge"],
            marker="o",
            linewidth=2.0,
            label=f"N={n_vehicles}",
        )
        axes[2].plot(
            frame["epsilon"],
            frame["mean_worst_case_margin"],
            marker="o",
            linewidth=2.0,
            label=f"N={n_vehicles}",
        )

    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[2].set_xscale("log")
    axes[0].invert_xaxis()
    axes[1].invert_xaxis()
    axes[2].invert_xaxis()
    axes[0].set_ylabel("Mean EUR-equivalent cost")
    axes[1].set_ylabel("Mean V2G discharge (kWh)")
    axes[2].set_ylabel("Mean full-box margin")
    for ax in axes:
        ax.set_xlabel("Joint risk level epsilon")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "q1_bernstein_epsilon_tradeoff.png", dpi=200)
    plt.close(fig)


def plot_q1_parameter_sensitivity(sensitivity: pd.DataFrame, out_dir: Path) -> None:
    if sensitivity.empty:
        return
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "legend.fontsize": 9})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    axes = axes.ravel()

    specs = [
        ("rho", "total_discharge", "Throughput penalty rho", "V2G discharge (kWh)"),
        ("alpha", "total_discharge", "Load penalty alpha", "V2G discharge (kWh)"),
        ("epsilon", "exact_violation_probability", "Joint risk level epsilon", "Exact violation probability"),
        ("gamma", "violation_probability", "Budget B", "Sampled violation probability"),
    ]
    for ax, (parameter, metric, xlabel, ylabel) in zip(axes, specs):
        frame = sensitivity[sensitivity["parameter"] == parameter].sort_values("parameter_value")
        ax.plot(frame["parameter_value"], frame[metric], marker="o", linewidth=2.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if parameter == "epsilon":
            ax.invert_xaxis()

    fig.tight_layout()
    fig.savefig(out_dir / "q1_parameter_sensitivity.png", dpi=200)
    plt.close(fig)


def print_table(df: pd.DataFrame, columns: list[str], title: str) -> None:
    print(f"\n{title}")
    print(df[columns].to_string(index=False, float_format=lambda v: f"{v:8.4f}"))


def package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not installed"


def runtime_environment_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("OS", platform.platform()),
            ("Python", sys.version.split()[0]),
            ("CVXPY", cp.__version__),
            ("NumPy", np.__version__),
            ("Pandas", pd.__version__),
            ("OSQP", package_version("osqp")),
            ("CLARABEL", package_version("clarabel")),
            ("SCS", package_version("scs")),
            ("Installed solvers", ", ".join(cp.installed_solvers())),
            ("Processor", platform.processor() or "not reported"),
        ],
        columns=["item", "value"],
    )


def main() -> None:
    base = build_base_case()
    cases = build_cases()
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    deltas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    summary, delta_solutions = run_delta_sensitivity(base, deltas)
    summary.to_csv(out_dir / "summary.csv", index=False)

    selected_delta = 0.5
    case_summary, case_solutions = run_case_benchmark(cases, delta=selected_delta)
    case_summary.to_csv(out_dir / "case_benchmark_delta_50.csv", index=False)

    parameter_summary = run_parameter_sensitivity(base, delta=selected_delta)
    parameter_summary.to_csv(out_dir / "parameter_sensitivity_delta_50.csv", index=False)

    multi_day_summary = run_multi_day_benchmark(None, delta=selected_delta)
    multi_day_summary.to_csv(out_dir / "multi_day_benchmark_delta_50.csv", index=False)
    multi_day_policy_summary = summarize_multi_day_results(multi_day_summary)
    multi_day_policy_summary.to_csv(out_dir / "multi_day_policy_summary_delta_50.csv", index=False)

    runtime_environment_frame().to_csv(out_dir / "runtime_environment.csv", index=False)

    for policy_name in ["deterministic", "quantile_75", "robust"]:
        solution = delta_solutions[(base.name, policy_name, selected_delta)]
        schedule_frame(base, solution).to_csv(
            out_dir / f"schedule_{policy_name}_delta_{int(selected_delta * 100):02d}.csv",
            index=False,
        )

    plot_schedule(base, delta_solutions, delta=selected_delta, out_dir=out_dir)
    plot_delta_sensitivity(summary, out_dir=out_dir)
    plot_case_benchmark(case_summary, out_dir=out_dir)
    plot_multi_day_benchmark(multi_day_summary, out_dir=out_dir)
    plot_parameter_sensitivity(parameter_summary, out_dir=out_dir)

    display_columns = [
        "policy",
        "delta",
        "status",
        "economic_cost",
        "total_discharge",
        "terminal_energy",
        "max_simultaneous_flow",
        "solve_time_sec",
        "exact_violation_probability",
        "violation_probability",
        "max_violation",
    ]
    print_table(summary, display_columns, "Base-case delta sensitivity")

    case_columns = [
        "case",
        "policy",
        "status",
        "economic_cost",
        "total_discharge",
        "solve_time_sec",
        "exact_violation_probability",
        "violation_probability",
        "max_violation",
    ]
    print_table(case_summary, case_columns, "Scenario benchmark at delta=0.5")

    multi_day_columns = [
        "policy",
        "days",
        "mean_net_cost",
        "mean_total_discharge",
        "mean_exact_violation_probability",
        "max_exact_violation_probability",
        "max_worst_case_margin",
    ]
    print_table(multi_day_policy_summary, multi_day_columns, "Multi-day benchmark at delta=0.5")

    print(f"\nSaved results to: {out_dir.resolve()}")


def load_fleet_experiment_context(
    base: EVData,
    fleet_config: FleetExperimentConfig,
    out_dir: Path,
) -> tuple[pd.DataFrame | None, list[str] | None]:
    if not fleet_config.use_acn_sessions:
        return None, None

    cache = acn_cache_path(
        fleet_config.acn_site,
        fleet_config.acn_start,
        fleet_config.acn_end,
        fleet_config.acn_cache,
    )
    sessions = load_acn_sessions(
        fleet_config.acn_site,
        fleet_config.acn_start,
        fleet_config.acn_end,
        fleet_config.acn_min_kwh,
        cache,
    )
    metadata = prepare_acn_metadata(sessions, base, fleet_config, cache, out_dir)
    if metadata.empty or int(metadata.loc[0, "valid_horizon_days_for_max_fleet"]) == 0:
        raise ValueError(
            "No ACN calendar day contains enough feasible sessions for the maximum "
            f"fleet size {max(FLEET_COUNTS)} after filtering. Try a wider date range, "
            "a smaller min_kwh, or --use-acn-sessions false."
        )
    horizon_dates = pd.read_csv(out_dir / "acn_fleet_horizon_dates.csv")["horizon_date"].dropna().astype(str).tolist()
    return sessions, horizon_dates


def load_or_run_fleet_multi_seed_benchmark(
    base: EVData,
    out_dir: Path,
    *,
    delta: float,
    fleet_config: FleetExperimentConfig,
    acn_sessions: pd.DataFrame | None,
    horizon_dates: list[str] | None,
    vehicle_counts: list[int] | tuple[int, ...] = FLEET_COUNTS,
    seeds: list[int] | tuple[int, ...] = FLEET_SEEDS,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_path = out_dir / "q1_fleet_multi_seed.csv"
    summary_path = out_dir / "q1_fleet_multi_seed_summary.csv"
    if not force and raw_path.exists() and summary_path.exists():
        raw = pd.read_csv(raw_path)
        summary = pd.read_csv(summary_path)
        required_policies = {"deterministic", "quantile_75", "robust"}
        cache_has_fixed_limit = "grid_limit" in raw.columns and np.allclose(raw["grid_limit"].dropna(), FLEET_FIXED_GRID_LIMIT)
        if cache_has_fixed_limit and required_policies.issubset(set(raw.get("policy", pd.Series(dtype=str)).dropna())) and required_policies.issubset(
            set(summary.get("policy", pd.Series(dtype=str)).dropna())
        ):
            return raw, summary

    raw = run_fleet_multi_seed_benchmark(
        base,
        vehicle_counts=vehicle_counts,
        seeds=seeds,
        delta=delta,
        fleet_config=fleet_config,
        acn_sessions=acn_sessions,
        horizon_dates=horizon_dates,
    )
    raw.to_csv(raw_path, index=False)
    summary = summarize_fleet_results(raw)
    summary.to_csv(summary_path, index=False)
    return raw, summary


def load_or_run_budgeted_robust_grid(
    base: EVData,
    out_dir: Path,
    *,
    delta: float,
    fleet_config: FleetExperimentConfig,
    acn_sessions: pd.DataFrame | None,
    horizon_dates: list[str] | None,
    vehicle_counts: list[int] | tuple[int, ...] = FLEET_COUNTS,
    seeds: list[int] | tuple[int, ...] = FLEET_SEEDS,
    gammas: list[float] | tuple[float, ...] = GAMMA_LEVELS,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_path = out_dir / "q1_budgeted_robust_gamma_grid.csv"
    summary_path = out_dir / "q1_budgeted_robust_gamma_summary.csv"
    if not force and raw_path.exists() and summary_path.exists():
        raw = pd.read_csv(raw_path)
        if "grid_limit" in raw.columns and np.allclose(raw["grid_limit"].dropna(), FLEET_FIXED_GRID_LIMIT):
            return raw, pd.read_csv(summary_path)

    raw = run_budgeted_robust_grid(
        base,
        vehicle_counts=vehicle_counts,
        seeds=seeds,
        gammas=gammas,
        delta=delta,
        fleet_config=fleet_config,
        acn_sessions=acn_sessions,
        horizon_dates=horizon_dates,
    )
    raw.to_csv(raw_path, index=False)
    summary = summarize_fleet_results(raw)
    summary.to_csv(summary_path, index=False)
    return raw, summary


def build_fleet_config_from_args(args: argparse.Namespace) -> FleetExperimentConfig:
    return FleetExperimentConfig(
        use_acn_sessions=args.use_acn_sessions,
        acn_site=args.acn_site,
        acn_start=args.acn_start,
        acn_end=args.acn_end,
        acn_min_kwh=args.acn_min_kwh,
        acn_cache=args.acn_cache,
        slot_minutes=60,
    )


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reliability-aware EV/V2G scheduling experiments.")
    parser.add_argument("--q1-experiments", action="store_true", help="Run the full paper/Q1 experiment suite.")
    parser.add_argument("--only-multi-ev", action="store_true", help="Run only the fleet scaling demo.")
    parser.add_argument("--multi-ev", action="store_true", help="Run the default scalar experiments and then the fleet demo.")
    parser.add_argument(
        "--use-acn-sessions",
        type=parse_bool,
        default=True,
        help="Use ACN-Data session-level records for fleet experiments. Use false for the legacy synthetic fallback.",
    )
    parser.add_argument("--acn-site", choices=sorted(ACN_SITE_TIMEZONES), default="caltech")
    parser.add_argument("--acn-start", default="2019-01-01", help="Inclusive local ACN session start date, YYYY-MM-DD.")
    parser.add_argument("--acn-end", default="2019-03-31", help="Inclusive local ACN session end date, YYYY-MM-DD.")
    parser.add_argument("--acn-min-kwh", type=float, default=2.0, help="Minimum delivered energy for ACN sessions.")
    parser.add_argument(
        "--acn-cache",
        default=None,
        help="CSV path for session-level ACN records. Defaults to <site>.csv when present, otherwise data/acn_sessions_<site>_<start>_<end>.csv.",
    )
    return parser.parse_args(argv)


def multi_ev_demo(fleet_config: FleetExperimentConfig | None = None) -> None:
    base = build_base_case()
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    fleet_config = fleet_config or FleetExperimentConfig()
    acn_sessions, horizon_dates = load_fleet_experiment_context(base, fleet_config, out_dir)

    selected_delta = 0.5
    fleet_summary = run_multi_ev_scaling(
        base,
        delta=selected_delta,
        fleet_config=fleet_config,
        acn_sessions=acn_sessions,
        horizon_dates=horizon_dates,
    )
    fleet_summary.to_csv(out_dir / "multi_ev_scaling_delta_50.csv", index=False)
    plot_multi_ev_scaling(fleet_summary, out_dir=out_dir)

    display_columns = [
        "n_vehicles",
        "policy",
        "status",
        "economic_cost",
        "total_discharge",
        "solve_time_sec",
        "decision_variables",
        "exact_violation_probability",
        "worst_case_margin",
        "max_simultaneous_flow",
    ]
    print_table(fleet_summary, display_columns, "Optional multi-EV scaling benchmark at delta=0.5")
    print(f"\nSaved multi-EV results to: {out_dir.resolve()}")


def q1_experiments(fleet_config: FleetExperimentConfig | None = None) -> None:
    base = build_base_case()
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    fleet_config = fleet_config or FleetExperimentConfig()
    acn_sessions, horizon_dates = load_fleet_experiment_context(base, fleet_config, out_dir)
    selected_delta = 0.5

    summary, delta_solutions = run_delta_sensitivity(base, list(SENSITIVITY_DELTAS))
    summary.to_csv(out_dir / "q1_baseline_policy_rerun.csv", index=False)

    budgeted = run_budgeted_robust_grid(
        base,
        delta=selected_delta,
        fleet_config=fleet_config,
        acn_sessions=acn_sessions,
        horizon_dates=horizon_dates,
    )
    budgeted.to_csv(out_dir / "q1_budgeted_robust_gamma_grid.csv", index=False)
    budgeted_summary = summarize_fleet_results(budgeted)
    budgeted_summary.to_csv(out_dir / "q1_budgeted_robust_gamma_summary.csv", index=False)

    bernstein = run_bernstein_chance_grid(
        base,
        delta=selected_delta,
        fleet_config=fleet_config,
        acn_sessions=acn_sessions,
        horizon_dates=horizon_dates,
    )
    bernstein.to_csv(out_dir / "q1_bernstein_socp_epsilon_grid.csv", index=False)
    bernstein_summary = summarize_fleet_results(bernstein)
    bernstein_summary.to_csv(out_dir / "q1_bernstein_socp_epsilon_summary.csv", index=False)

    binary_validation = run_binary_validation(base, delta=selected_delta)
    binary_validation.to_csv(out_dir / "q1_binary_complementarity_validation.csv", index=False)
    binary_validation_summary = summarize_binary_validation(binary_validation)
    binary_validation_summary.to_csv(out_dir / "q1_binary_complementarity_summary.csv", index=False)

    fleet_multi_seed = run_fleet_multi_seed_benchmark(
        base,
        delta=selected_delta,
        fleet_config=fleet_config,
        acn_sessions=acn_sessions,
        horizon_dates=horizon_dates,
    )
    fleet_multi_seed.to_csv(out_dir / "q1_fleet_multi_seed.csv", index=False)
    fleet_multi_seed_summary = summarize_fleet_results(fleet_multi_seed)
    fleet_multi_seed_summary.to_csv(out_dir / "q1_fleet_multi_seed_summary.csv", index=False)

    sensitivity = run_q1_sensitivity(
        base,
        delta=selected_delta,
        fleet_config=fleet_config,
        acn_sessions=acn_sessions,
        horizon_dates=horizon_dates,
    )
    sensitivity.to_csv(out_dir / "q1_parameter_sensitivity.csv", index=False)

    ablation = run_theorem_condition_ablation(base, delta=selected_delta)
    ablation.to_csv(out_dir / "q1_theorem_condition_ablation.csv", index=False)

    plot_q1_fleet_multiseed(fleet_multi_seed, out_dir=out_dir)
    plot_q1_budgeted_gamma(budgeted_summary, out_dir=out_dir)
    plot_q1_bernstein_epsilon(bernstein_summary, out_dir=out_dir)
    plot_q1_parameter_sensitivity(sensitivity, out_dir=out_dir)

    runtime_environment_frame().to_csv(out_dir / "runtime_environment.csv", index=False)
    for policy_name in ["deterministic", "quantile_75", "robust"]:
        solution = delta_solutions[(base.name, policy_name, selected_delta)]
        schedule_frame(base, solution).to_csv(
            out_dir / f"q1_schedule_{policy_name}_delta_{int(selected_delta * 100):02d}.csv",
            index=False,
        )

    print_table(
        summary,
        [
            "policy",
            "delta",
            "status",
            "economic_cost",
            "total_discharge",
            "max_simultaneous_flow",
            "exact_violation_probability",
            "worst_case_margin",
        ],
        "Q1 baseline policy rerun",
    )
    print_table(
        budgeted_summary,
        [
            "n_vehicles",
            "gamma",
            "mean_net_cost",
            "mean_total_discharge",
            "mean_violation_probability",
            "mean_solve_time_sec",
        ],
        "Q1 budgeted robust B summary",
    )
    print_table(
        bernstein_summary,
        [
            "n_vehicles",
            "epsilon",
            "mean_net_cost",
            "mean_total_discharge",
            "mean_violation_probability",
            "mean_solve_time_sec",
        ],
        "Q1 Bernstein/SOCP eps summary",
    )
    print_table(
        binary_validation,
        [
            "validation_model",
            "policy",
            "status",
            "objective",
            "max_simultaneous_flow",
            "objective_gap_vs_relaxation",
            "charge_l1_gap_vs_relaxation",
        ],
        "Q1 binary complementarity validation",
    )
    print_table(
        binary_validation_summary,
        [
            "case",
            "validation_policy",
            "miqp_status",
            "relaxed_objective",
            "miqp_objective",
            "objective_gap_vs_relaxation",
            "charge_l1_gap_vs_relaxation",
            "relaxed_worst_case_margin",
        ],
        "Q1 MIQP complementarity summary",
    )
    print(f"\nSaved Q1 experiment results to: {out_dir.resolve()}")


if __name__ == "__main__":
    cli_args = parse_cli_args()
    cli_fleet_config = build_fleet_config_from_args(cli_args)
    if cli_args.q1_experiments:
        q1_experiments(cli_fleet_config)
    elif cli_args.only_multi_ev:
        multi_ev_demo(cli_fleet_config)
    else:
        main()
        if cli_args.multi_ev:
            multi_ev_demo(cli_fleet_config)
