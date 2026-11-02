"""
build_dataset.py
================
Builds the model-ready NYC yellow-taxi demand table used in the project.

This one script reproduces everything in data/ and the one-page dataset_summary.pdf from public sources.

Pipeline (same logic as eda.ipynb, made reproducible end-to-end):
  1. Download monthly yellow-taxi trip files (Parquet) + taxi zone lookup and zone shapes from TLC.
  2. Aggregate raw trips to pickups per taxi zone per hour (DuckDB, never loads raw trips into pandas).
  3. Densify to a full zone x hour grid (hours with no trips -> 0 pickups).
  4. Keep only busy zones (avg >= MIN_AVG pickups/hour over [MIN_AVG_SINCE, TEST_START), i.e. recent
     dev data only; default 20/h since congestion pricing start).
  5. Fetch hourly weather for Midtown from the Open-Meteo Historical Forecast API (one fixed model,
     default GFS) so the same source/model can be queried live via the Forecast API at deployment.
     Weather is for the target hour itself, i.e. it assumes the next-day forecast ~ what happened.
  6. Holiday calendar from the `holidays` package (offline, known years ahead -> no leakage).
  7. NWS weather alerts (watches/warnings/advisories) from the Iowa Environmental Mesonet archive, kept
     only if already issued H hours before the target hour. Live equivalent: api.weather.gov/alerts.
  8. Build forecasting features that only use information available H hours before the target hour.
  9. Write one row per (zone, hour) to a gzip CSV (or Parquet), with a `split` column (dev/test).
     dev is for model selection / cross-validation; hist_avg-style target encodings are NOT in the table,
     compute them per CV fold from that fold's training rows (and from all of dev for the final test run).
 10. Write dataset_summary.pdf: a one-page description for teammates (columns, sources, caveats, a map of
     the modeled zones). Every number in it is computed from the table just built.

Outputs
  data/model_table.csv.gz         the modeling table
  data/model_table.features.txt   the feature column names
  dataset_summary.pdf             one-page summary
  data/raw/                       cached downloads (trips, zones, weather, alerts)
  data/taxi.duckdb                intermediate tables (scratch; rebuilt every run)

Output columns
  identifiers : zone_id, zone_name, borough, ts_hour, split
  target      : pickups                       (pickups in that zone during that hour)
  features    : see FEATURES list below      (also written to <output>.features.txt)

Usage
  pip install -r requirements.txt     # duckdb, pandas, holidays, geopandas, matplotlib, reportlab
  python build_dataset.py                                  # defaults below (first run downloads ~2.5 GB)
  python build_dataset.py --horizon 24 --end 2026-08 --out data/model_table.csv.gz
  python build_dataset.py --format parquet                 # much faster to write/read
  python build_dataset.py --skip-download                  # reuse files already in data/raw
  python build_dataset.py --min-avg 1 --min-avg-since 2023-01-01   # old behaviour: all zones >= 1/h over full dev period
  python build_dataset.py --weather-model ecmwf_ifs025     # another Open-Meteo model (check its archive start)
  python build_dataset.py --no-report                      # skip the PDF

"""

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

import duckdb

BASE_URL = "https://d37ci6vzurychx.cloudfront.net"
CBD_PRICING_START = "2025-01-05"  # congestion pricing start date
# Hours missing from the TLC files themselves (near-zero trips citywide, incl. JFK/LGA). Rows whose target
# falls in a gap, or whose lag features reach into it (up to 672h later), are dropped.
DATA_GAPS = [("2023-09-21 13:00", "2023-09-25 00:00")]

# Open-Meteo: archive of past forecasts. Live equivalent (deployment) = https://api.open-meteo.com/v1/forecast
WEATHER_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
WEATHER_LAT, WEATHER_LON = 40.754, -73.984  # Midtown Manhattan; one point is enough for 47 Manhattan zones
# NWS alerts: IEM archive of every watch/warning/advisory issued by the Upton NY office (OKX).
ALERTS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/watchwarn.py"
ALERT_UGC = {132: "NYZ178", 138: "NYZ176"}  # JFK -> Southern Queens, LGA -> Northern Queens
ALERT_UGC_DEFAULT = "NYZ072"                # all Manhattan zones -> New York (Manhattan)
ALERT_GROUPS = {  # VTEC phenomena code -> feature group
    "WS": "winter", "BZ": "winter", "WW": "winter", "IS": "winter",
    "HT": "heat", "EH": "heat", "XH": "heat",
    "CW": "cold", "EC": "cold", "WC": "cold",
    "WI": "wind", "HW": "wind", "TR": "wind", "HU": "wind",
    "FA": "flood", "FF": "flood", "FL": "flood", "CF": "flood",
}
ALERT_LEVEL = {"Y": 1, "A": 2, "W": 3}      # advisory < watch < warning (statements ignored)

WEATHER_VARS = {  # Open-Meteo variable -> column name
    "temperature_2m": "temp_c",
    "apparent_temperature": "feels_like_c",
    "precipitation": "precip_mm",     # rain + snow; rain/snowfall/weather_code are not archived for GFS before 2024-12
    "snow_depth": "snow_depth_m",
    "wind_speed_10m": "wind_kmh",
    "cloud_cover": "cloud_cover_pct",
}

FEATURES = [
    "zone_code", "borough_code",                      # location (integer-encoded categoricals)
    "lag_h", "lag_48h", "roll_24h", "roll_7d",        # recent demand (all >= H hours old)
    "lag_168h", "lag_336h", "same_hour_4wk",          # weekly pattern
    "duration_lag_h", "fare_lag_h",                   # trip context H hours earlier (NaN if no trips)
    "hr", "dow", "month", "is_weekend",               # calendar (dow: 0 = Sunday ... 6 = Saturday)
    "post_cbd_pricing",                               # 1 on/after congestion pricing start
    *WEATHER_VARS.values(), "is_precip", "likely_snow",             # weather at the target hour (stands in for the forecast)
    "is_holiday", "is_minor_holiday", "is_holiday_season",          # holidays (see step_holidays)
    "days_to_holiday", "days_since_holiday",
    *(f"alert_{g}" for g in dict.fromkeys(ALERT_GROUPS.values())),  # NWS alert level known H hours ahead (0-3)
]
TARGET = "pickups"


# ----------------------------------------------------------------------------- helpers
def month_list(start: str, end: str):
    """'2023-01', '2026-08' -> ['2023-01', ..., '2026-08'] (inclusive)."""
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def next_month_start(ym: str) -> str:
    y, m = map(int, ym.split("-"))
    y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return f"{y}-{m:02d}-01"


def download(url: str, dest: pathlib.Path, retries: int = 3) -> bool:
    """Download to a temp file, then rename, so an interrupted download never leaves a broken file."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):  # month not published yet
                print(f"  not available ({e.code}): {url}")
                return False
            print(f"  attempt {attempt} failed ({e}); retrying...")
        except Exception as e:  # network hiccup
            print(f"  attempt {attempt} failed ({e}); retrying...")
        time.sleep(3 * attempt)
    tmp.unlink(missing_ok=True)
    print(f"  giving up on {url}")
    return False


# ----------------------------------------------------------------------------- steps
def step_download(raw: pathlib.Path, months, skip: bool):
    raw.mkdir(parents=True, exist_ok=True)
    lookup = raw / "taxi_zone_lookup.csv"
    if skip:
        print("[1/9] Skipping download (using existing files).")
    else:
        print(f"[1/9] Downloading {len(months)} monthly files + zone lookup and shapes...")
        download(f"{BASE_URL}/misc/taxi_zone_lookup.csv", lookup)
        download(f"{BASE_URL}/misc/taxi_zones.zip", raw / "taxi_zones.zip")
        for ym in months:
            f = raw / f"yellow_{ym}.parquet"
            if not f.exists():
                print(f"  {ym}")
            download(f"{BASE_URL}/trip-data/yellow_tripdata_{ym}.parquet", f)

    have = [ym for ym in months if (raw / f"yellow_{ym}.parquet").exists()]
    missing = sorted(set(months) - set(have))
    if not lookup.exists():
        sys.exit("Zone lookup CSV missing; cannot continue.")
    if not have:
        sys.exit("No trip files found; cannot continue.")
    if missing:
        print(f"  WARNING: missing months will be dropped from the panel: {missing}")
    return have


def step_aggregate(con, raw: pathlib.Path, months, start_ts: str, end_ts: str):
    print("[2/9] Aggregating trips to zone x hour...")
    files = [str(raw / f"yellow_{ym}.parquet") for ym in months]
    con.execute(f"""
    CREATE OR REPLACE TABLE zone_hour AS
    SELECT PULocationID AS zone_id,
           date_trunc('hour', tpep_pickup_datetime) AS ts_hour,
           COUNT(*) AS pickups,
           AVG(trip_distance) AS avg_distance,
           AVG(CAST(total_amount AS DECIMAL(14, 2))) AS avg_fare_total,  -- exact sum -> identical output every run
           AVG(date_diff('minute', tpep_pickup_datetime, tpep_dropoff_datetime)) AS avg_duration_min
    FROM read_parquet({files}, union_by_name = true)
    WHERE tpep_pickup_datetime >= TIMESTAMP '{start_ts}'
      AND tpep_pickup_datetime <  TIMESTAMP '{end_ts}'
      AND tpep_dropoff_datetime >= tpep_pickup_datetime      -- drop negative durations
      AND PULocationID BETWEEN 1 AND 263                     -- 264/265 = unknown / outside NYC
    GROUP BY 1, 2
    """)

    # Sanity check: flag months whose volume jumps vs. neighbours (usually a broken download).
    m = con.sql("""SELECT date_trunc('month', ts_hour) AS month, SUM(pickups) AS p
                   FROM zone_hour GROUP BY 1 ORDER BY 1""").df()
    ratio = m["p"] / ((m["p"].shift(1) + m["p"].shift(-1)) / 2)
    odd = m[(ratio < 0.8) | (ratio > 1.2)]
    total = int(m["p"].sum())
    print(f"  {total:,} pickups across {len(m)} months")
    if len(odd):
        print("  WARNING: months with unusual volume (check these files):")
        print(odd.to_string(index=False))

    # Same idea per day (a few missing days barely move a monthly total): compare each day with the
    # median of the same weekday in the surrounding +-4 weeks. Holidays and big storms show up here too;
    # anything else is likely missing data and belongs in DATA_GAPS.
    d = con.sql("""SELECT CAST(ts_hour AS DATE) AS day, SUM(pickups) AS p
                   FROM zone_hour GROUP BY 1 ORDER BY 1""").df()
    d["day"] = d["day"].astype("datetime64[ns]")
    d = d.set_index("day").asfreq("D", fill_value=0)
    typical = d.groupby(d.index.day_of_week)["p"].transform(
        lambda s: s.rolling(9, center=True, min_periods=5).median())
    d["ratio"] = (d["p"] / typical).round(2)
    odd = d[(d["ratio"] < 0.5) | (d["ratio"] > 1.5)]
    if len(odd):
        print(f"  WARNING: {len(odd)} days below 50% / above 150% of a typical same weekday "
              "(holiday/storm, or missing data -> add to DATA_GAPS):")
        print("   ", ", ".join(f"{i:%Y-%m-%d} ({r:.2f})" for i, r in odd["ratio"].items()))


def step_panel(con, raw: pathlib.Path, start_ts: str, end_ts: str):
    print("[3/9] Building dense zone x hour panel...")
    lookup = raw / "taxi_zone_lookup.csv"
    con.execute(f"""
    CREATE OR REPLACE TABLE panel AS
    WITH hours AS (
        SELECT range AS ts_hour
        FROM range(TIMESTAMP '{start_ts}', TIMESTAMP '{end_ts}', INTERVAL 1 HOUR)),
    zones AS (
        SELECT LocationID AS zone_id, Borough AS borough, Zone AS zone_name
        FROM read_csv_auto('{lookup}')
        WHERE LocationID BETWEEN 1 AND 263)
    SELECT z.zone_id, z.zone_name, z.borough, h.ts_hour,
           COALESCE(a.pickups, 0) AS pickups,
           a.avg_fare_total, a.avg_duration_min
    FROM zones z
    CROSS JOIN hours h
    LEFT JOIN zone_hour a ON a.zone_id = z.zone_id AND a.ts_hour = h.ts_hour
    """)
    n = con.execute("SELECT COUNT(*) FROM panel").fetchall()[0][0]
    print(f"  {n:,} zone-hours")


def step_weather(con, raw: pathlib.Path, start_ts: str, end_ts: str, model: str, skip: bool):
    """Hourly weather in UTC (cached per year), converted to NYC local time to match the TLC timestamps."""
    import pandas as pd

    print(f"[4/9] Weather (Open-Meteo historical forecast, model = {model})...")
    start = pd.Timestamp(start_ts) - pd.Timedelta(days=1)  # 1-day buffer each side for the UTC shift
    end = pd.Timestamp(end_ts)                             # exclusive end + ~1 day buffer (end_date is inclusive)
    frames = []
    for year in range(start.year, end.year + 1):
        f = raw / f"weather_{model}_{year}.json"
        lo, hi = max(start, pd.Timestamp(f"{year}-01-01")), min(end, pd.Timestamp(f"{year}-12-31"))
        stale = f.exists() and json.loads(f.read_text())["hourly"]["time"][-1] < f"{hi:%Y-%m-%d}T23:00"
        if not f.exists() or (stale and not skip):
            print(f"  fetching {lo:%Y-%m-%d} -> {hi:%Y-%m-%d}")
            url = (f"{WEATHER_URL}?latitude={WEATHER_LAT}&longitude={WEATHER_LON}"
                   f"&start_date={lo:%Y-%m-%d}&end_date={hi:%Y-%m-%d}"
                   f"&hourly={','.join(WEATHER_VARS)}&models={model}&timezone=GMT")
            if not download(url, f):
                sys.exit(f"Weather download failed for {year}; cannot continue.")
        frames.append(pd.DataFrame(json.loads(f.read_text())["hourly"]))

    w = pd.concat(frames)[["time", *WEATHER_VARS]].rename(columns=WEATHER_VARS)
    # UTC -> naive NYC local. Fall-back DST hour appears twice (average it); spring-forward hour is
    # missing locally but exists in the panel, so reindex and forward-fill.
    w["ts_hour"] = (pd.to_datetime(w.pop("time"), utc=True)
                    .dt.tz_convert("America/New_York").dt.tz_localize(None))
    w = w.groupby("ts_hour").mean()
    w = w.reindex(pd.date_range(start_ts, end_ts, freq="h", inclusive="left"), method="ffill")
    w = w.rename_axis("ts_hour").reset_index()
    n_missing = int(w.drop(columns="ts_hour").isna().any(axis=1).sum())
    if n_missing:
        print(f"  WARNING: {n_missing} hours with missing weather values (left as NaN)")
    con.register("weather_df", w)
    con.execute("CREATE OR REPLACE TABLE weather AS SELECT * FROM weather_df")
    con.unregister("weather_df")
    print(f"  {len(w):,} hours, {w.ts_hour.min()} -> {w.ts_hour.max()}")


def step_holidays(con, start_ts: str, end_ts: str):
    """Daily holiday table.
    is_holiday        : offices close (NYSE calendar, minus one-off closures), on both the actual and the
                        observed date (July 4 fireworks happen on the 4th even when the day off is the 3rd)
    is_minor_holiday  : other NY public holidays where most offices stay open (Columbus, Veterans, Election Day...)
    is_holiday_season : Dec 24 - Jan 1 and the day after Thanksgiving
    days_to/since_holiday : distance to the nearest is_holiday date, clipped at 7
    """
    import holidays
    import pandas as pd

    print("[5/9] Holiday calendar...")
    days = pd.date_range(pd.Timestamp(start_ts) - pd.Timedelta(days=7),
                         pd.Timestamp(end_ts) + pd.Timedelta(days=7), freq="D")
    years = range(days[0].year, days[-1].year + 1)
    nyse = holidays.financial_holidays("NYSE", years=years)
    ny = holidays.US(subdiv="NY", years=years)  # includes actual + observed dates
    base = lambda name: name.replace(" (observed)", "")
    major_names = {base(n) for n in nyse.values() if "Mourning" not in n}
    major = {d for d, n in nyse.items() if base(n) in major_names} | \
            {d for d, n in ny.items() if base(n) in major_names}
    minor = {d for d, n in ny.items() if d not in major and n != "Susan B. Anthony Day"}
    thanksgiving = {d for d, n in ny.items() if n == "Thanksgiving Day"}

    h = pd.DataFrame({"day": days})
    d = h["day"].dt.date
    h["is_holiday"] = d.isin(major).astype(int)
    h["is_minor_holiday"] = d.isin(minor).astype(int)
    h["is_holiday_season"] = (((h.day.dt.month == 12) & (h.day.dt.day >= 24)) |
                              ((h.day.dt.month == 1) & (h.day.dt.day == 1)) |
                              (d - pd.Timedelta(days=1)).isin(thanksgiving)).astype(int)
    hol = h.day.where(h.is_holiday == 1)
    h["days_to_holiday"] = (hol.bfill() - h.day).dt.days.clip(upper=7).fillna(7).astype(int)
    h["days_since_holiday"] = (h.day - hol.ffill()).dt.days.clip(upper=7).fillna(7).astype(int)
    con.register("holidays_df", h)
    con.execute("CREATE OR REPLACE TABLE holiday_cal AS SELECT * FROM holidays_df")
    con.unregister("holidays_df")
    shown = h[(h.is_holiday == 1) & (h.day >= start_ts) & (h.day < end_ts)]
    print(f"  {len(shown)} holiday days in range, e.g. {', '.join(shown.day.dt.strftime('%Y-%m-%d')[-3:])}")


def step_alerts(con, raw: pathlib.Path, start_ts: str, end_ts: str, H: int, skip: bool):
    """Hourly NWS alert level per alert group and NWS zone, as it was known H hours before each hour.
    An alert counts for target hour t (decision time T0 = t - H, in UTC) if it was first sent at or before T0,
    had not yet expired / been cancelled / been upgraded by T0, and its originally planned window covers t.
    Short-fuse alerts (thunderstorm, flash-flood warnings...) are issued too late and drop out automatically.
    """
    import pandas as pd

    print(f"[6/9] NWS alerts (IEM archive, known >= {H}h ahead)...")
    ugcs = sorted({ALERT_UGC_DEFAULT, *ALERT_UGC.values()})
    frames = []
    for year in range(pd.Timestamp(start_ts).year, pd.Timestamp(end_ts).year + 1):
        f = raw / f"nws_alerts_OKX_{year}.csv"
        if not f.exists() or (not skip and year >= pd.Timestamp.now().year - 1):
            url = (f"{ALERTS_URL}?sts={year - 1}-12-25T00:00Z&ets={year + 1}-01-01T00:00Z"
                   "&wfo%5B%5D=OKX&accept=csv")
            f.unlink(missing_ok=True)
            if not download(url, f):
                sys.exit(f"NWS alert download failed for {year}; cannot continue.")
        frames.append(pd.read_csv(f, dtype=str))
    a = pd.concat(frames)
    a = a[a.ugc.isin(ugcs) & a.phenomena.isin(ALERT_GROUPS) & a.significance.isin(ALERT_LEVEL)]
    a = a.drop_duplicates(["ugc", "phenomena", "significance", "eventid", "vtec_year"])
    utc = lambda c, fmt=None: pd.to_datetime(a[c], format=fmt, utc=True)
    a = a.assign(sent=utc("utc_prodissue"), begin=utc("utc_issue"), end=utc("utc_init_expire", "%Y%m%d%H%M"),
                 gone=utc("utc_expire"), group=a.phenomena.map(ALERT_GROUPS), level=a.significance.map(ALERT_LEVEL))

    # target hours in UTC (the panel is naive NYC local time)
    local = pd.date_range(start_ts, end_ts, freq="h", inclusive="left")
    t_utc = local.tz_localize("America/New_York", ambiguous=[True] * len(local),  # fall-back hour: 1st one
                              nonexistent="shift_forward").tz_convert("UTC")
    T0 = t_utc - pd.Timedelta(hours=H)

    groups = list(dict.fromkeys(ALERT_GROUPS.values()))
    out = []
    for ugc in ugcs:
        tab = pd.DataFrame(0, index=range(len(local)), columns=[f"alert_{g}" for g in groups])
        for r in a[a.ugc == ugc].itertuples():
            on = (r.sent <= T0) & (r.gone > T0) & (r.begin <= t_utc) & (t_utc < r.end)
            col = f"alert_{r.group}"
            tab.loc[on, col] = tab.loc[on, col].clip(lower=r.level)
        out.append(tab.assign(ugc=ugc, ts_hour=local))
    al = pd.concat(out, ignore_index=True)
    con.register("alerts_df", al)
    con.execute("CREATE OR REPLACE TABLE alerts AS SELECT * FROM alerts_df")
    con.unregister("alerts_df")
    on = al[al.ugc == ALERT_UGC_DEFAULT].filter(like="alert_").gt(0).sum()
    print(f"  {len(a)} alerts for {', '.join(ugcs)}; Manhattan hours with an alert: "
          + ", ".join(f"{k.removeprefix('alert_')} {v}" for k, v in on.items()))


def step_features(con, H: int, test_start: str, min_avg: float, min_avg_since: str):
    print(f"[7/9] Building features (horizon = {H}h, zones avg >= {min_avg}/h over "
          f"{min_avg_since} -> {test_start})...")
    if H > 168:
        sys.exit("Horizon > 168h would make the weekly lags leak future data; use H <= 168.")
    if min_avg_since >= test_start:
        sys.exit("--min-avg-since must be before --test-start (zone selection uses dev data only).")
    groups = list(dict.fromkeys(ALERT_GROUPS.values()))
    alert_cols = ", ".join(f"COALESCE(al.alert_{g}, 0) AS alert_{g}" for g in groups)
    ugc_case = " ".join(f"WHEN {z} THEN '{u}'" for z, u in ALERT_UGC.items())
    gap_filter = "".join(
        f"      AND NOT (b.ts_hour >= TIMESTAMP '{lo}' AND b.ts_hour < TIMESTAMP '{hi}' + INTERVAL 672 HOUR)\n"
        for lo, hi in DATA_GAPS)
    con.execute(f"""
    CREATE OR REPLACE TABLE features AS
    WITH active AS (   -- zone selection on recent dev data only (no peeking at test)
        SELECT zone_id FROM panel
        WHERE ts_hour >= TIMESTAMP '{min_avg_since}' AND ts_hour < TIMESTAMP '{test_start}'
        GROUP BY zone_id HAVING AVG(pickups) >= {min_avg}),
    codes AS (   -- small integer codes for sklearn categorical features
        SELECT a.zone_id,
               DENSE_RANK() OVER (ORDER BY a.zone_id) - 1 AS zone_code,
               DENSE_RANK() OVER (ORDER BY p.borough) - 1 AS borough_code
        FROM active a
        JOIN (SELECT DISTINCT zone_id, borough FROM panel) p USING (zone_id)),
    base AS (
        SELECT zone_id, zone_name, borough, ts_hour, pickups,
            LAG(pickups, {H})  OVER w AS lag_h,
            LAG(pickups, 48)   OVER w AS lag_48h,
            LAG(pickups, 168)  OVER w AS lag_168h,
            LAG(pickups, 336)  OVER w AS lag_336h,
            LAG(pickups, 504)  OVER w AS lag_504h,
            LAG(pickups, 672)  OVER w AS lag_672h,
            AVG(pickups) OVER (w ROWS BETWEEN {H + 23}  PRECEDING AND {H} PRECEDING) AS roll_24h,
            AVG(pickups) OVER (w ROWS BETWEEN {H + 167} PRECEDING AND {H} PRECEDING) AS roll_7d,
            LAG(avg_duration_min, {H}) OVER w AS duration_lag_h,
            LAG(avg_fare_total, {H})   OVER w AS fare_lag_h
        FROM panel
        WINDOW w AS (PARTITION BY zone_id ORDER BY ts_hour))
    SELECT
        b.zone_id, b.zone_name, b.borough, b.ts_hour,
        CASE WHEN b.ts_hour < TIMESTAMP '{test_start}' THEN 'dev' ELSE 'test' END AS split,
        b.pickups,
        c.zone_code, c.borough_code,
        b.lag_h, b.lag_48h,
        ROUND(b.roll_24h, 3) AS roll_24h,
        ROUND(b.roll_7d, 3)  AS roll_7d,
        b.lag_168h, b.lag_336h,
        ROUND((b.lag_168h + b.lag_336h + b.lag_504h + b.lag_672h) / 4.0, 3) AS same_hour_4wk,
        ROUND(b.duration_lag_h, 2) AS duration_lag_h,
        ROUND(b.fare_lag_h, 2)     AS fare_lag_h,
        hour(b.ts_hour)  AS hr,
        dayofweek(b.ts_hour) AS dow,
        month(b.ts_hour) AS month,
        CAST(dayofweek(b.ts_hour) IN (0, 6) AS INT) AS is_weekend,
        CAST(b.ts_hour >= TIMESTAMP '{CBD_PRICING_START}' AS INT) AS post_cbd_pricing,
        ROUND(w.temp_c, 1) AS temp_c, ROUND(w.feels_like_c, 1) AS feels_like_c,
        ROUND(w.precip_mm, 2) AS precip_mm, ROUND(w.snow_depth_m, 3) AS snow_depth_m,
        ROUND(w.wind_kmh, 1) AS wind_kmh, ROUND(w.cloud_cover_pct) AS cloud_cover_pct,
        CAST(w.precip_mm > 0.1 AS INT) AS is_precip,
        CAST(w.precip_mm > 0.1 AND w.temp_c <= 1 AS INT) AS likely_snow,
        hc.is_holiday, hc.is_minor_holiday, hc.is_holiday_season,
        hc.days_to_holiday, hc.days_since_holiday,
        {alert_cols}
    FROM base b
    JOIN codes c USING (zone_id)
    LEFT JOIN weather w ON w.ts_hour = b.ts_hour
    JOIN holiday_cal hc ON CAST(hc.day AS DATE) = CAST(b.ts_hour AS DATE)
    LEFT JOIN alerts al ON al.ts_hour = b.ts_hour
        AND al.ugc = CASE b.zone_id {ugc_case} ELSE '{ALERT_UGC_DEFAULT}' END
    WHERE b.lag_672h IS NOT NULL          -- first 4 weeks have no history
{gap_filter}    ORDER BY b.ts_hour, b.zone_id
    """)
    summary = con.sql("""SELECT split, COUNT(*) AS rows, MIN(ts_hour) AS first, MAX(ts_hour) AS last,
                                COUNT(DISTINCT zone_id) AS zones
                         FROM features GROUP BY 1 ORDER BY first""").df()
    print(summary.to_string(index=False))


def step_export(con, out: pathlib.Path, fmt: str):
    print(f"[8/9] Writing {out} ...")
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        con.execute(f"COPY features TO '{out}' (FORMAT PARQUET)")
    else:
        compression = "GZIP" if str(out).endswith(".gz") else "NONE"
        con.execute(f"""COPY features TO '{out}'
                        (FORMAT CSV, HEADER, COMPRESSION {compression},
                         TIMESTAMPFORMAT '%Y-%m-%d %H:%M:%S')""")
    feat_file = out.with_name(out.name.split(".")[0] + ".features.txt")
    feat_file.write_text("\n".join(FEATURES) + "\n")
    print(f"  done: {out.stat().st_size / 1e6:.1f} MB; feature list in {feat_file}")


def zone_map_png(con, raw: pathlib.Path) -> "io.BytesIO":
    """Choropleth of NYC taxi zones: modeled zones shaded by average pickups/hour, all others gray."""
    import io

    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch

    zones = gpd.read_file(f"zip://{raw / 'taxi_zones.zip'}!taxi_zones/taxi_zones.shp")
    avg = con.sql("SELECT zone_id, AVG(pickups) AS avg FROM features GROUP BY 1").df()
    zones = zones.merge(avg, left_on="LocationID", right_on="zone_id", how="left")
    on = zones[zones.avg.notna()]
    # one-hue sequential ramp (blue 200 -> 700); the lightest steps are left for "not modeled"
    ramp = LinearSegmentedColormap.from_list("blue", ["#9ec5f4", "#5598e7", "#256abf", "#104281", "#0d366b"])

    fig, ax = plt.subplots(figsize=(3.0, 2.75), dpi=300)
    zones[zones.avg.isna()].plot(ax=ax, color="#e7e6e2", edgecolor="white", linewidth=0.25)
    on.plot(ax=ax, column="avg", cmap=ramp, edgecolor="white", linewidth=0.3)
    x0, y0, x1, y1 = on.total_bounds
    pad = 0.06 * max(x1 - x0, y1 - y0)
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_axis_off()
    for loc_id, label, dx, dy in [(132, "JFK", -16000, -4000), (138, "LGA", 3000, 7000)]:
        c = zones.loc[zones.LocationID == loc_id].geometry.iloc[0].centroid
        ax.annotate(label, (c.x, c.y), (c.x + dx, c.y + dy), fontsize=6.5, color="#1f2933", ha="center",
                    arrowprops=dict(arrowstyle="-", color="#5b6670", lw=0.5))
    ax.text(x0 + 0.5 * pad, y1 - 0.5 * pad, "Manhattan", fontsize=6.5, color="#1f2933", ha="left")

    sm = plt.cm.ScalarMappable(cmap=ramp, norm=plt.Normalize(on.avg.min(), on.avg.max()))
    cax = fig.add_axes([0.04, 0.075, 0.5, 0.03])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.5, length=2, pad=1, colors="#5b6670")
    fig.text(0.04, 0.125, "Modeled zones: avg pickups per hour", fontsize=6, color="#1f2933")
    fig.legend(handles=[Patch(facecolor="#e7e6e2", edgecolor="white", label="Not modeled")],
               loc="lower right", bbox_to_anchor=(1.0, 0.06), fontsize=6, frameon=False, handlelength=1.2)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0.17)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf


def step_report(con, raw: pathlib.Path, pdf: pathlib.Path, out: pathlib.Path, a):
    """One-page dataset summary for teammates; all numbers come from the tables built above."""
    import datetime

    import holidays
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    print(f"[9/9] Writing {pdf} ...")
    H = a.horizon
    q = lambda sql: con.sql(sql).fetchall()[0]
    n, first, last, n_zones, n_manh, mean, med, mx, zero = q("""
        SELECT COUNT(*), MIN(ts_hour), MAX(ts_hour), COUNT(DISTINCT zone_id),
               COUNT(DISTINCT zone_id) FILTER (WHERE borough = 'Manhattan'),
               AVG(pickups), MEDIAN(pickups), MAX(pickups), AVG(CASE WHEN pickups = 0 THEN 1 ELSE 0 END)
        FROM features""")
    splits = {s: (lo, hi, c) for s, lo, hi, c in con.sql(
        "SELECT split, MIN(ts_hour), MAX(ts_hour), COUNT(*) FROM features GROUP BY 1").fetchall()}
    others = [z for (z,) in con.sql("""SELECT DISTINCT zone_name FROM features
                                        WHERE borough <> 'Manhattan' ORDER BY 1""").fetchall()]
    trips = q("SELECT SUM(pickups) FROM zone_hour")[0]
    hours_with_history = q("SELECT COUNT(DISTINCT ts_hour) FROM panel")[0] - 672
    gap_rows = n_zones * hours_with_history - n
    winter = q(f"""SELECT SUM(pickups) / SUM(same_hour_4wk) FROM features
                   WHERE borough = 'Manhattan' AND alert_winter = 3 AND is_holiday + is_holiday_season = 0""")[0]
    d = lambda ts: f"{ts:%-d %b %Y}"
    pct = lambda x: f"{100 * x:.0f}%"
    gaps = "; ".join(f"{lo} to {hi}" for lo, hi in DATA_GAPS)

    INK, MUTED, ACCENT = colors.HexColor("#1f2933"), colors.HexColor("#5b6670"), colors.HexColor("#1d4e89")
    CORE_BG, TRY_BG, LABEL_BG, RULE = (colors.HexColor("#e8f0fa"), colors.HexColor("#fff4dc"),
                                       colors.HexColor("#f3f5f7"), colors.HexColor("#c9d2db"))
    ss = getSampleStyleSheet()
    base = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=7.3, leading=8.9, textColor=INK)
    small = ParagraphStyle("s", parent=base, fontSize=6.8, leading=8.2)
    muted = ParagraphStyle("m", parent=base, textColor=MUTED, fontSize=6.9, leading=8.4)
    h1 = ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=14.5, leading=17)
    h2 = ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=9, leading=11, textColor=ACCENT,
                        spaceBefore=4, spaceAfter=2)
    li = ParagraphStyle("li", parent=base, leftIndent=9, bulletIndent=0, spaceAfter=1)
    cellh = ParagraphStyle("ch", parent=small, fontName="Helvetica-Bold", textColor=colors.white)
    P = lambda t, s=small: Paragraph(t, s)
    code = lambda t: f'<font name="Courier" size="6.6">{t}</font>'
    grid = [("GRID", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8)]
    header = [("BACKGROUND", (0, 0), (-1, 0), ACCENT)]

    # --- top: task + key facts (left), zone map (right)
    dev_lo, dev_hi, dev_n = splits["dev"]
    test_lo, test_hi, test_n = splits["test"]
    facts = Table([
        [P("<b>Rows</b>"), P(f"{n:,}")],
        [P("<b>Period</b>"), P(f"{d(first)} – {d(last)} (hourly, NYC local time)")],
        [P("<b>Zones</b>"), P(f"{n_zones}: {n_manh} in Manhattan + {' + '.join(others)} (map)")],
        [P("<b>Pickups / row</b>"), P(f"mean {mean:.0f}, median {med:.0f}, max {mx:,}; {100 * zero:.1f}% of rows are 0")],
        [P("<b>dev</b> split"), P(f"{d(dev_lo)} – {d(dev_hi)}, {dev_n:,} rows: model selection with "
                                  "time-ordered CV")],
        [P("<b>test</b> split"), P(f"{d(test_lo)} – {d(test_hi)}, {test_n:,} rows: used once, at the end "
                                   "(a full year)")],
    ], colWidths=[0.95 * inch, 3.55 * inch])
    facts.setStyle(TableStyle(grid + [("BACKGROUND", (0, 0), (0, -1), LABEL_BG)]))
    left = [P("<b>Task.</b> Predict how many yellow-taxi pickups happen in each busy NYC taxi zone during each "
              f"hour, <b>{H} hours in advance</b>. <b>One row = one zone × one hour.</b> The target column is "
              "<b>pickups</b> (trips starting in that zone in that hour).", base), Spacer(1, 5), facts]
    top = Table([[left, Image(zone_map_png(con, raw), width=2.75 * inch, height=2.52 * inch)]],
                colWidths=[4.6 * inch, 2.9 * inch])
    top.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                             ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story = [P("NYC Yellow-Taxi Demand: Model Table at a Glance", h1),
             P(f"{code(out.as_posix())} &nbsp;·&nbsp; built by {code('build_dataset.py')} "
               f"&nbsp;·&nbsp; snapshot of {datetime.date.today():%-d %b %Y} &nbsp;·&nbsp; reproduce everything "
               f"with {code('python build_dataset.py')}", muted),
             Spacer(1, 5), top]

    # --- columns
    feat = [
        (None, "IDs (not features)", "zone_id, zone_name, borough, ts_hour, split",
         "Row identifiers; <i>split</i> = dev / test", "—"),
        (CORE_BG, "Location", "zone_code, borough_code", "Zone and borough as integer categories", "TLC zone lookup"),
        (CORE_BG, "Recent demand", "lag_h, lag_48h, roll_24h, roll_7d",
         f"Pickups {H}h and 48h earlier; average over the 24h / 7 days ending {H}h earlier", "TLC trips"),
        (CORE_BG, "Weekly pattern", "lag_168h, lag_336h, same_hour_4wk",
         "Same hour 1 and 2 weeks earlier; average of the same hour over the last 4 weeks", "TLC trips"),
        (CORE_BG, "Trip context", "duration_lag_h, fare_lag_h",
         f"Average trip duration (min) and fare ($) in the zone {H}h earlier; empty if there were no trips",
         "TLC trips"),
        (CORE_BG, "Calendar", "hr, dow, month, is_weekend",
         "Hour 0–23, day of week (0 = Sunday), month, weekend flag", "Timestamp"),
        (CORE_BG, "Weather", ", ".join([*WEATHER_VARS.values(), "is_precip", "likely_snow"]),
         "Midtown weather for the target hour; is_precip = precip &gt; 0.1 mm; "
         "likely_snow = precip at 1 °C or colder", f"Open-Meteo ({a.weather_model})"),
        (CORE_BG, "Holidays", "is_holiday, is_holiday_season, days_to_holiday, days_since_holiday",
         "Offices closed (NYSE calendar); Dec 24 – Jan 1 and the day after Thanksgiving; "
         "days to the next / since the last holiday (capped at 7)", "holidays pkg"),
        (CORE_BG, "Weather alerts", ", ".join(f"alert_{g}" for g in dict.fromkeys(ALERT_GROUPS.values())),
         f"Highest NWS alert in effect for the hour, counted only if it was already issued {H}h earlier: "
         f"0 none, 1 advisory, 2 watch, 3 warning. In Manhattan, winter-warning hours ran at {pct(winter)} "
         "of normal demand", "NWS via IEM"),
        (TRY_BG, "Test in CV", "post_cbd_pricing, is_minor_holiday",
         f"Congestion-pricing period (from {d(datetime.date.fromisoformat(CBD_PRICING_START))}); holidays most "
         "offices ignore (Columbus, Veterans, Election Day). Possibly redundant, so compare models with and "
         "without them", "—"),
    ]
    listed = {c.strip() for _, _, cols, _, _ in feat for c in cols.split(",")}
    missing = [f for f in FEATURES if f not in listed]
    if missing:  # keep the PDF honest when FEATURES changes
        print(f"  WARNING: features not described in the PDF: {missing}")
    rows = [[P(h, cellh) for h in ["Group", "Columns", "What it means", "Source"]]]
    style = grid + header
    for i, (bg, g, cols, desc, src) in enumerate(feat, start=1):
        rows.append([P(f"<b>{g}</b>"), P(code(cols)), P(desc), P(src)])
        if bg is not None:
            style.append(("BACKGROUND", (0, i), (-1, i), bg))
    t = Table(rows, colWidths=[0.9 * inch, 2.35 * inch, 3.05 * inch, 1.2 * inch])
    t.setStyle(TableStyle(style))
    story += [P("Columns", h2), t, Spacer(1, 2),
              P('<font backColor="#e8f0fa">&nbsp;&nbsp;&nbsp;&nbsp;</font> blue = planned model features &nbsp;&nbsp; '
                '<font backColor="#fff4dc">&nbsp;&nbsp;&nbsp;&nbsp;</font> amber = to be tested in CV &nbsp;&nbsp; '
                f"white = identifiers. <b>Not in the file:</b> {code('hist_avg')} (average pickups per zone × "
                "weekday × hour). Compute it inside each CV fold from that fold's training rows only, otherwise "
                "it leaks the answer.", muted)]

    # --- sources
    src = [
        [P(h, cellh) for h in ["Source", "What we take", "Access"]],
        [P("<b>NYC TLC trip records</b>"),
         P(f"Every yellow-taxi trip, {first:%b %Y} – {last:%b %Y} ({trips / 1e6:.0f}M valid trips), "
           "aggregated to pickups per zone per hour. Zones with no trips in an hour get 0. Zones kept only if "
           f"they averaged at least {a.min_avg:g} pickups/h from {d(datetime.date.fromisoformat(a.min_avg_since))} "
           f"to {d(dev_hi)} (dev data only)."),
         P("Free monthly Parquet files from the TLC website")],
        [P("<b>Open-Meteo Historical Forecast API</b>"),
         P(f"Hourly weather at one Midtown point from the <b>{a.weather_model}</b> weather model. This is an "
           "archive of past forecasts, so it is very close to the observed weather. Project 2 uses the live "
           "Open-Meteo forecast with the same model and point, so training and deployment come from one source."),
         P("Free, no API key; cached in " + code("data/raw/"))],
        [P("<b>NWS weather alerts</b>"),
         P("Every watch, warning and advisory for Manhattan, Southern Queens (JFK) and Northern Queens (LGA), "
           "with the time it was first issued. The Iowa Environmental Mesonet archive provides the history; "
           "Project 2 uses the live api.weather.gov feed, which carries the same alert codes."),
         P("Free, no API key; cached in " + code("data/raw/"))],
        [P("<b>Python <i>holidays</i> package</b>"),
         P("Stock-exchange (NYSE) holiday calendar as the \"offices closed\" list, plus other NY public "
           "holidays. Known years ahead, so no leakage."),
         P(f"Open source, offline (v{holidays.__version__})")],
    ]
    t = Table(src, colWidths=[1.45 * inch, 4.45 * inch, 1.6 * inch])
    t.setStyle(TableStyle(grid + header))
    story += [P("Where the data comes from", h2), t]

    # --- caveats
    notes = [
        f"<b>No look-ahead.</b> Every demand feature is at least {H}h old, so each row only uses what was known "
        f"{H}h before.",
        "<b>Weather is treated as a perfect forecast.</b> Rows hold the weather that actually occurred, not the "
        "forecast from the day before, so test scores are slightly optimistic (mostly because of rain). "
        "Project 2 will measure the gap.",
        "<b>Cross-validate in time order.</b> Use folds ordered by time (e.g. by month), not shuffled K-fold, and "
        "build the folds on timestamps so all zones for an hour stay in the same fold.",
        "<b>Holiday echo.</b> Weekly lags carry a holiday's low demand into the following week (e.g. lag_168h on "
        "the Thursday after Thanksgiving). Watch for this in the per-day errors.",
        f"<b>Data gap removed.</b> TLC's raw data is missing most trips from {gaps}. Those hours and the 4 weeks "
        f"after them (whose lags look back into the gap) are dropped, {gap_rows:,} dev rows.",
    ]
    story += [P("Good to know before modeling", h2)] + [Paragraph(x, li, bulletText="•") for x in notes]

    doc = SimpleDocTemplate(str(pdf), pagesize=letter, leftMargin=0.5 * inch, rightMargin=0.5 * inch,
                            topMargin=0.45 * inch, bottomMargin=0.4 * inch,
                            title="NYC Taxi Demand Model Table: Data Summary", author="Project 1 team")
    doc.build(story)
    pages = doc.page if hasattr(doc, "page") else "?"
    print(f"  done: {pdf} ({pages} page{'s' if pages != 1 else ''})")


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Build the NYC taxi zone-hour demand modeling table.")
    p.add_argument("--start", default="2023-01", help="first month (YYYY-MM)")
    p.add_argument("--end", default="2026-08", help="last month, inclusive (YYYY-MM)")
    p.add_argument("--horizon", type=int, default=24, help="forecast horizon in hours (<= 168)")
    p.add_argument("--test-start", default="2025-09-01", help="dev = before this date; test = from this date on")
    p.add_argument("--min-avg", type=float, default=20.0,
                   help="keep zones averaging >= this many pickups/hour (see --min-avg-since)")
    p.add_argument("--min-avg-since", default=CBD_PRICING_START,
                   help="start of the window (up to --test-start) used to compute each zone's average")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--db", default="data/taxi.duckdb")
    p.add_argument("--out", default="data/model_table.csv.gz")
    p.add_argument("--format", choices=["csv", "parquet"], default="csv")
    p.add_argument("--weather-model", default="gfs_global",
                   help="Open-Meteo model; must be archived back to --start and forecast >= 1 day ahead")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--report", default="dataset_summary.pdf", help="one-page summary PDF")
    p.add_argument("--no-report", action="store_true")
    a = p.parse_args()

    out = pathlib.Path(a.out)
    if a.format == "parquet" and out.suffix != ".parquet":
        out = out.with_name(out.name.split(".")[0] + ".parquet")

    raw = pathlib.Path(a.raw_dir)
    pathlib.Path(a.db).parent.mkdir(parents=True, exist_ok=True)
    start_ts = f"{a.start}-01"
    end_ts = next_month_start(a.end)

    t0 = time.time()
    months = step_download(raw, month_list(a.start, a.end), a.skip_download)
    con = duckdb.connect(a.db)
    step_aggregate(con, raw, months, start_ts, end_ts)
    step_panel(con, raw, start_ts, end_ts)
    step_weather(con, raw, start_ts, end_ts, a.weather_model, a.skip_download)
    step_holidays(con, start_ts, end_ts)
    step_alerts(con, raw, start_ts, end_ts, a.horizon, a.skip_download)
    step_features(con, a.horizon, a.test_start, a.min_avg, a.min_avg_since)
    step_export(con, out, a.format)
    if not a.no_report:
        step_report(con, raw, pathlib.Path(a.report), out, a)
    print(f"Finished in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()