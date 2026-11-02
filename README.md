# NYC Yellow-Taxi Demand Forecasting

Predict how many yellow-taxi pickups happen in each busy NYC taxi zone during each hour, **24 hours in advance**.

This repo holds the code that builds the modeling table from public sources. The data itself is not committed:
you rebuild it with one command. [`dataset_summary.pdf`](dataset_summary.pdf) is a one-page overview of the
table (columns, sources, caveats, and a map of the modeled zones).

## Setup

Requires **Python 3.11+** (tested on 3.13).

```bash
git clone <repo-url>
cd <repo-folder>
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Build the dataset

```bash
python build_dataset.py
```

The first run downloads about 2.5 GB of TLC trip files into `data/raw/` and takes a few minutes, depending on
your connection. Later runs reuse the cached files; add `--skip-download` to skip the download checks
entirely (a rebuild then takes about 30 seconds).

Every run produces the same table: the build is deterministic.

| Output | What it is |
|---|---|
| `data/model_table.csv.gz` | **The modeling table** (one row per zone × hour, ~1.5M rows) |
| `data/model_table.features.txt` | Names of the 34 feature columns |
| `dataset_summary.pdf` | One-page summary, regenerated from the table |
| `data/raw/` | Cached downloads (trips, zone shapes, weather, weather alerts) |
| `data/taxi.duckdb` | Intermediate tables; scratch space, safe to delete |

Run `python build_dataset.py --help` for options (forecast horizon, date range, test start, zone threshold).

## Use the table

```python
import pandas as pd

df = pd.read_csv("data/model_table.csv.gz", parse_dates=["ts_hour"])
features = open("data/model_table.features.txt").read().split()

dev, test = df[df.split == "dev"], df[df.split == "test"]
X_dev, y_dev = dev[features], dev["pickups"]
```

- **Target:** `pickups`. **Identifiers** (not features): `zone_id`, `zone_name`, `borough`, `ts_hour`, `split`.
- **`dev`** (29 Jan 2023 – 31 Aug 2025) is for model selection with **time-ordered** cross-validation. Build the
  folds on `ts_hour`, not shuffled rows, so all zones for an hour stay in the same fold.
- **`test`** (1 Sep 2025 – 31 Aug 2026, a full year) is for the final score only.
- **`hist_avg`** (average pickups per zone × weekday × hour) is deliberately not in the file. Compute it inside
  each CV fold from that fold's training rows only, otherwise it leaks the answer.

## Data sources

All free, with no API keys:

- **NYC TLC trip records**: monthly yellow-taxi Parquet files and the taxi zone lookup and shapes
- **Open-Meteo Historical Forecast API**: hourly weather for Midtown from the GFS model
- **Iowa Environmental Mesonet**: archive of National Weather Service watches, warnings and advisories
- **Python `holidays` package**: NYSE and New York holiday calendars

See [`dataset_summary.pdf`](dataset_summary.pdf) for how each one is used and what to watch out for.
