"""Public-data loaders for the revised V2G study.

Optimization lives in revision_model.py; experiments in run_revision_experiments.py.
The superseded optimization implementation is preserved under revision/original/.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
OPSD_COLUMNS = ["utc_timestamp", "cet_cest_timestamp", "DE_LU_load_actual_entsoe_transparency", "DE_LU_price_day_ahead"]
ACN_SITE_TIMEZONES = {"caltech":"America/Los_Angeles", "jpl":"America/Los_Angeles", "office001":"America/Los_Angeles"}
PUBLIC_PROFILE_DATE = "2019-01-24"

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

def acn_cache_path(site, start_date, end_date):
    return ROOT / (site + ".csv")


def load_opsd_year():
    return pd.read_csv(ROOT / "data/opsd_de_lu_2019.csv")


def available_profile_dates_2019():
    data=load_opsd_year()
    return [date for date,g in data.groupby(data.utc_timestamp.str[:10])
            if len(g)==24 and not g[OPSD_COLUMNS].isna().any().any()]


def affine_scale(values, lower, upper):
    values=np.asarray(values,float)
    if np.isclose(values.max(),values.min()):return np.full_like(values,(lower+upper)/2)
    return lower+(values-values.min())*(upper-lower)/(values.max()-values.min())


def load_public_profile(profile_date=PUBLIC_PROFILE_DATE):
    data=load_opsd_year();day=data[data.utc_timestamp.str.startswith(profile_date)].copy()
    if len(day)!=24 or day[OPSD_COLUMNS].isna().any().any():
        raise ValueError("A complete 24-hour load-and-price day is required")
    raw=day.DE_LU_price_day_ahead.to_numpy(float)
    load=day.DE_LU_load_actual_entsoe_transparency.to_numpy(float)
    # Positive-price CONTROL only. The historical arm explicitly replaces these with raw prices.
    positive=np.maximum(raw,1.)/1000
    return pd.DataFrame({"hour":np.arange(24),"utc_timestamp":day.utc_timestamp.to_numpy(),
        "raw_load_mw":load,"raw_price_eur_per_mwh":raw,"station_baseload":affine_scale(load,390.,475.),
        "buy_price_eur_per_kwh":positive,"sell_price_eur_per_kwh":.98*positive})


def build_case_from_profile(profile_date, name=None):
    p=load_public_profile(profile_date);a=np.zeros(24);a[10:21]=1
    return EVData(name or profile_date,profile_date,24,p.station_baseload.to_numpy(),a,
        p.buy_price_eur_per_kwh.to_numpy(),p.sell_price_eur_per_kwh.to_numpy(),np.full(24,1.5),
        30.,25.,475.,50.,10.,75.,60.,.95,.9,2e-4,.001)


def build_base_case():
    return build_case_from_profile(PUBLIC_PROFILE_DATE,"base")
