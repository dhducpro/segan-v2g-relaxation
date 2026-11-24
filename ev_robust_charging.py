"""Read the station load, tariff and charging-session inputs for the revised study."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PUBLIC_PROFILE_DATE = "2019-01-24"
OPSD_COLUMNS = [
    "utc_timestamp",
    "cet_cest_timestamp",
    "DE_LU_load_actual_entsoe_transparency",
    "DE_LU_price_day_ahead",
]
ACN_SITE_TIMEZONES = {
    "caltech": "America/Los_Angeles",
    "jpl": "America/Los_Angeles",
    "office001": "America/Los_Angeles",
}


@dataclass(frozen=True)
class EVData:
    """Single-vehicle inputs, with power in kW and battery energy in kWh."""

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


def load_acn_sessions(
    site: str,
    start_date: str,
    end_date: str,
    min_kwh: float,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Read sessions connected on the requested local dates, including both endpoints."""
    site = site.lower()
    if site not in ACN_SITE_TIMEZONES:
        sites = ", ".join(sorted(ACN_SITE_TIMEZONES))
        raise ValueError(f"Unsupported ACN site {site!r}. Expected one of {sites}.")

    timezone_name = ACN_SITE_TIMEZONES[site]
    path = Path(cache_path) if cache_path else acn_cache_path(site, start_date, end_date)
    if not path.is_file():
        raise RuntimeError(
            f"ACN session export missing at {path}. Put {site}.csv in the repository "
            "root or pass its path to load_acn_sessions()."
        )

    try:
        source = pd.read_csv(path)
    except PermissionError:
        # Read the same file with the Python parser if the Windows C parser cannot open it.
        source = pd.read_csv(path, engine="python")
    sessions = standardize_acn_session_frame(source, site, timezone_name, min_kwh)
    timezone = ZoneInfo(timezone_name)
    start = datetime.fromisoformat(start_date).replace(tzinfo=timezone)
    stop = datetime.fromisoformat(end_date).replace(tzinfo=timezone) + timedelta(days=1)
    connections = parse_acn_timestamp(sessions["connectionTime"], timezone_name)
    return sessions.loc[(connections >= start) & (connections < stop)].copy()


def standardize_acn_session_frame(
    frame: pd.DataFrame,
    site: str,
    timezone_name: str,
    min_kwh: float,
) -> pd.DataFrame:
    """Accept API or compact CSV column names and retain sessions delivering min_kwh."""
    aliases = {
        "Start": "connectionTime",
        "End": "disconnectTime",
        "Energy": "kWhDelivered",
    }
    sessions = frame.rename(
        columns={old: new for old, new in aliases.items() if new not in frame.columns}
    ).copy()
    missing = sorted(set(aliases.values()).difference(sessions.columns))
    if missing:
        raise ValueError(
            "ACN session CSV must contain connectionTime/disconnectTime/kWhDelivered "
            f"or Start/End/Energy. Missing columns: {missing}."
        )

    # Preserve the source row numbering when an export contains no session identifiers.
    if "session_id" not in sessions:
        sessions["session_id"] = np.arange(len(sessions))
    defaults = {
        "site": site,
        "timezone": timezone_name,
        "stationID": "",
        "spaceID": "",
        "siteID": site,
        "doneChargingTime": "",
        "userInput_kWhRequested": np.nan,
        "userInput_WhPerMile": np.nan,
    }
    for name, default in defaults.items():
        if name not in sessions:
            sessions[name] = default

    sessions["kWhDelivered"] = pd.to_numeric(sessions["kWhDelivered"], errors="coerce")
    columns = [
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
    return sessions.loc[sessions["kWhDelivered"] >= float(min_kwh), columns].copy()


def parse_acn_timestamp(series: pd.Series, timezone_name: str) -> pd.Series:
    """Convert timestamped observations to local time, localizing naive observations."""
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(timezone_name, nonexistent="shift_forward", ambiguous="NaT")
    return parsed.dt.tz_convert(timezone_name)


def acn_cache_path(site: str, start_date: str, end_date: str) -> Path:
    """One export per site covers the full period. Date filtering occurs on read."""
    return ROOT / f"{site}.csv"


def load_opsd_year() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "opsd_de_lu_2019.csv")


def _complete_opsd_day(day: pd.DataFrame) -> bool:
    return len(day) == 24 and day[OPSD_COLUMNS].notna().all().all()


def available_profile_dates_2019() -> list[str]:
    hourly = load_opsd_year()
    utc_dates = hourly["utc_timestamp"].str[:10]
    return [date for date, day in hourly.groupby(utc_dates) if _complete_opsd_day(day)]


def affine_scale(values, lower: float, upper: float) -> np.ndarray:
    """Scale a daily profile to the stated range, using its midpoint for constant loads."""
    values = np.asarray(values, dtype=float)
    minimum, maximum = values.min(), values.max()
    if np.isclose(maximum, minimum):
        return np.full_like(values, (lower + upper) / 2)
    return lower + (values - minimum) * (upper - lower) / (maximum - minimum)


def load_public_profile(profile_date: str = PUBLIC_PROFILE_DATE) -> pd.DataFrame:
    """Return the recorded signed prices and the tariffs used for the reference case."""
    hourly = load_opsd_year()
    day = hourly.loc[hourly["utc_timestamp"].str.startswith(profile_date)]
    if not _complete_opsd_day(day):
        raise ValueError(f"A complete 24-hour load-and-price day is required for {profile_date}.")

    raw_price = day["DE_LU_price_day_ahead"].to_numpy(dtype=float)
    raw_load = day["DE_LU_load_actual_entsoe_transparency"].to_numpy(dtype=float)
    # Only the positive-price control floors prices at 1 EUR/MWh.
    control_buy_price = np.maximum(raw_price, 1.0) / 1000
    return pd.DataFrame(
        {
            "hour": np.arange(24),
            "utc_timestamp": day["utc_timestamp"].to_numpy(),
            "raw_load_mw": raw_load,
            "raw_price_eur_per_mwh": raw_price,
            "station_baseload": affine_scale(raw_load, 390.0, 475.0),
            "buy_price_eur_per_kwh": control_buy_price,
            "sell_price_eur_per_kwh": 0.98 * control_buy_price,
        }
    )


def build_case_from_profile(profile_date: str, name: str | None = None) -> EVData:
    """Construct the reference single-EV case with availability in slots 11 through 21."""
    profile = load_public_profile(profile_date)
    availability = np.zeros(24)
    availability[10:21] = 1.0
    return EVData(
        name=name or profile_date,
        profile_date=profile_date,
        T=24,
        L=profile["station_baseload"].to_numpy(),
        availability=availability,
        price_buy=profile["buy_price_eur_per_kwh"].to_numpy(),
        price_sell=profile["sell_price_eur_per_kwh"].to_numpy(),
        u_nominal=np.full(24, 1.5),
        X=30.0,
        Y=25.0,
        M=475.0,
        E0=50.0,
        Emin=10.0,
        Emax=75.0,
        Etarget=60.0,
        eta_c=0.95,
        eta_d=0.9,
        alpha=2e-4,
        rho=0.001,
    )


def build_base_case() -> EVData:
    return build_case_from_profile(PUBLIC_PROFILE_DATE, "base")
