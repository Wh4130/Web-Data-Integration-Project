# Web Data Integration Project — Airports

Integration of airport data from three sources: [OurAirports](https://ourairports.com/data/),
[Wikidata](https://www.wikidata.org/), and a Kaggle airports dataset.

See [docs/idea_abstract.md](docs/idea_abstract.md) for the project idea and approach.

## Project Requirements

You should integrate:

1. **3 different data sets**
2. **At least 2,500 entities** described in total (in joint dataset)
   - but more are better; good: >10,000 but <100,000
3. **At least 1,000 entities** should be contained in at least **two datasets**
   - please estimate based on a small sample
4. **At least 8 attributes** in joint dataset
   - entities should be identifiable by attribute combinations of at least two attributes, e.g. name+birthdate
5. **At least 5 attributes** should be contained in at least **two datasets**
   - some attributes (other than name) should be contained in **three datasets** (for fusion by voting)
6. Ideally, **at least one attribute is a list attribute**
   - e.g. actors of a movie, directors of a company, songs on a CD

## Project Structure

```
data/
  raw/            # untouched downloads per source
    ourairports/
    wikidata/
    kaggle/
  interim/        # cleaned / profiled intermediate data
  processed/       # final fused/integrated dataset
docs/             # idea abstract, reports, notes
notebooks/        # per-source pipeline notebooks: <source>_profiling (attribute quality),
                  # <source>_extraction (fetch, resolve conflicts, save), <source>_eda
src/wdi_airports/ # reusable Python code (profiling, mapping, matching, fusion)
tests/            # tests for src/wdi_airports
```

## Data Sources & Notebooks

### Wikidata

Queried via SPARQL against a [QLever](https://qlever.dev/wikidata) mirror (`https://qlever.dev/api/wikidata`), scoped to `wdt:P31 wd:Q1248784` (instance of: airport). Three notebooks (please visit sequentially):

- [`wikidata_profiling.ipynb`](notebooks/wikidata_profiling.ipynb) measures missing/conflict rates per attribute (and documents why `GROUP_CONCAT` is broken on this endpoint, so profiling uses `COUNT`-based histograms instead);
- [`wikidata_extraction.ipynb`](notebooks/wikidata_extraction.ipynb) fetches raw values, resolves per-airport conflicts in pandas, fixes a feet/metres unit mixup in elevation, resolves entity-valued attributes (country, located_in, ...) to readable labels, and saves the final dataset;
- [`wikidata_eda.ipynb`](notebooks/wikidata_eda.ipynb) explores the resulting dataset (surfaced a duplicate-entity case and the elevation unit bug before it was fixed).

### OurAirports

[`ourairports_data_cleansing.ipynb`](notebooks/ourairports_data_cleansing.ipynb) merges `airports-ourairports.csv` with `runways.csv`, aggregating each airport's runways into a list-of-dicts column (`runways`) — this is the project's list attribute (requirement 6).

### Kaggle

[Global Airports (IATA, ICAO, timezone, geo)](https://www.kaggle.com/datasets/samvelkoch/global-airports-iata-icao-timezone-geo) — a static CSV download. Its page lists the data source as *FlightRank 2025: Aeroclub RecSys Cup, 2025*, a Kaggle competition on business-travel flight recommendation. In this project it is the only source of `timezone` / UTC offset next to Wikidata, and it adds city and country names.

**Provenance caveat.** The competition's airport object only carries `iata`, `icao`, the city IATA code and the country codes (A2/A3), which match the identifier columns of this CSV. It has no airport name, coordinates or timezone, so those columns were added from a source the dataset page does not document. We therefore check empirically that the file is not a copy of one of our other sources. On airports matched via `ICAO` / `icao_code` (so only the ~80% of Kaggle rows that have an ICAO; see [`source_data_heterogeneity_check.ipynb`](notebooks/source_data_heterogeneity_check.ipynb)):

| Compared pair | Matched airports | Identical name | Identical coordinates |
|---|---|---|---|
| Kaggle vs OurAirports | 4,700 | 22.7% | 1.0% |
| Kaggle vs Wikidata | 4,524 | 23.3% | 0.0% |
| OurAirports vs Wikidata (reference) | 7,758 | 66.5% | 0.0% |

Kaggle names follow a different style (abbreviations such as `Intl` / `Muni` / `Rgnl` in 12.4% of names, versus 0.01% in the other two sources; often no "Airport" suffix), so the source is heterogeneous enough for matching.

[`Kaggle_data_cleaning.ipynb`](notebooks/Kaggle_data_cleaning.ipynb) validates the file and exports `data/interim/airports-kaggle.csv`. It drops two rows that reuse another airport's ICAO code (the old Berlin Schönefeld `SXF`, which shares `EDDB` with BER, and a railway station that shares `EDLW` with Dortmund), leaving 6,391 airports. Known data-quality issues kept for later steps: 319 rows (5.0%) sit at coordinates (0, 0) and should be treated as missing, and 88 rows (1.4%) have no UTC offset.

### Feasibility / Requirement Analysis

[`requirement_analysis.ipynb`](notebooks/requirement_analysis.ipynb) checks the [Project Requirements](#project-requirements) above against the raw/interim data from all three sources — entity counts, cross-source overlap, attribute coverage, and the list attribute.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management, and requires Python 3.13+.

### Getting started

```bash
git clone <repo-url>
cd Web-Data-Integration-Project
uv sync
```

Then, in VS Code / Jupyter: when opening a notebook, select the kernel pointing at **this project's `.venv`** (not a global/leftover kernel from another project — picking the wrong one is the most common source of `ModuleNotFoundError` here). `uv sync` also registers this env's ipykernel, so it should show up as an option.

### Data

`data/raw/` and `data/interim/` are committed directly to the repo — after cloning, you already have all the data any notebook needs; there's no separate download/setup step. You only need to re-run a notebook if you're regenerating data after a code change (e.g. tweaking the conflict-resolution logic in `wikidata_extraction.ipynb`). For Wikidata specifically, re-running the extraction/profiling notebooks requires network access to the QLever SPARQL endpoint.

### Workflow

Each notebook is meant to run top-to-bottom without manual steps in between (`jupyter nbconvert --execute` should always succeed — if it doesn't, that's a bug, not an expected manual-intervention point). For Wikidata specifically, run the three notebooks in order: `wikidata_profiling` → `wikidata_extraction` → `wikidata_eda`.

Suggested git workflow for the team — don't commit directly to `main`, every change goes through a reviewed PR:

1. **Switch to `main`**
   ```bash
   git checkout main
   ```
2. **Pull the latest changes**
   ```bash
   git pull origin main
   ```
3. **Create a branch** for your task (naming: `<yourname>/<short-task>`)
   ```bash
   git checkout -b alice/kaggle-profiling
   ```
4. **Develop** — make your changes, and if you touched a notebook, re-run it top-to-bottom (`jupyter nbconvert --execute --inplace <notebook>.ipynb`) before committing, so the committed outputs aren't stale. Then commit:
   ```bash
   git add <files>
   git commit -m "..."
   ```
5. **Push your branch** to the remote
   ```bash
   git push -u origin alice/kaggle-profiling
   ```
6. **Open a PR** into `main` (GitHub web UI, or `gh pr create --base main`) and get at least one teammate's approval before merging — this is enforced by branch protection on `main`.
