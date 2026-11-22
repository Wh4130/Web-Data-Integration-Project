from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 1. Setup
# ------------------------------------------------------------------
ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

norm_dir = ROOT / "data" / "normalized_outputs"
out_dir = ROOT / "data" / "data_fusion" / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

DIST_KM = 20          # max distance inside a group before it is flagged
SAMPLE_SIZE = 30      # groups exported for manual review
RANDOM_SEED = 42
PLACEHOLDERS = {"", "nan", "none", "null", "n/a", "na", "unknown", "-", "--"}

# Country codes that legitimately differ (territory vs. sovereign state).
# Extend this set if the examples file still shows such pairs.
TERRITORY_PAIRS = {
    ("PR", "US"), ("GU", "US"), ("VI", "US"), ("MP", "US"), ("AS", "US"),
    ("UM", "US"), ("TC", "GB"), ("FK", "GB"), ("KY", "GB"), ("VG", "GB"),
    ("BM", "GB"), ("GI", "GB"), ("AI", "GB"), ("MS", "GB"), ("SH", "GB"),
    ("IM", "GB"), ("JE", "GB"), ("GG", "GB"), ("IO", "GB"), ("GL", "DK"),
    ("FO", "DK"), ("AW", "NL"), ("CW", "NL"), ("SX", "NL"), ("BQ", "NL"),
    ("NC", "FR"), ("PF", "FR"), ("GP", "FR"), ("MQ", "FR"), ("RE", "FR"),
    ("GF", "FR"), ("YT", "FR"), ("PM", "FR"), ("WF", "FR"), ("HK", "CN"),
    ("MO", "CN"),
}

# ------------------------------------------------------------------
# 2. Load groups and attribute values
# ------------------------------------------------------------------
r2g = pd.read_csv(out_dir / "record_to_group.csv")
r2g = r2g[r2g["group_id"].str.startswith("g")]          # matched groups only

cols = ["id", "name", "icao", "iata", "countryCode", "lat", "lon"]
recs = pd.concat(
    [pd.read_csv(norm_dir / f, low_memory=False).reindex(columns=cols)
     for f in ("ourairportsNormalized.csv", "wikidataNormalized.csv", "kaggleNormalized.csv")],
    ignore_index=True,
)
recs["id"] = recs["id"].astype(str)
for c in ["name", "icao", "iata", "countryCode"]:
    s = recs[c].astype("string").str.strip()
    recs[c] = s.mask(s.str.lower().isin(PLACEHOLDERS))
for c in ["icao", "iata", "countryCode"]:
    recs[c] = recs[c].str.upper()
for c in ["lat", "lon"]:
    recs[c] = pd.to_numeric(recs[c], errors="coerce")

df = r2g.merge(recs, left_on="record_id", right_on="id", how="left")
print(f"Matched groups: {df['group_id'].nunique()}  records: {len(df)}")


# ------------------------------------------------------------------
# 3. Helpers
# ------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def max_pair_distance(g):
    pts = g[["lat", "lon"]].dropna().to_numpy()
    best = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            best = max(best, haversine_km(pts[i, 0], pts[i, 1], pts[j, 0], pts[j, 1]))
    return best


def code_conflict(g, col):
    """Returns (conflict?, dataset of the odd-one-out or '')."""
    sub = g.dropna(subset=[col])
    if sub[col].nunique() <= 1:
        return False, ""
    counts = sub.groupby(col).size()
    if len(counts) == 2 and counts.max() >= 2 and counts.min() == 1:
        odd_val = counts.idxmin()
        return True, sub.loc[sub[col] == odd_val, "dataset"].iloc[0]
    return True, "ambiguous"


def country_conflict(g):
    codes = list(g["countryCode"].dropna().unique())
    odd = ""
    hard = False
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            a, b = codes[i], codes[j]
            if (a, b) not in TERRITORY_PAIRS and (b, a) not in TERRITORY_PAIRS:
                hard = True
    if hard:
        _, odd = code_conflict(g, "countryCode")
    return hard, odd


# ------------------------------------------------------------------
# 4. Analyse every group
# ------------------------------------------------------------------
rows = []
for gid, g in df.groupby("group_id"):
    icao_c, icao_odd = code_conflict(g, "icao")
    iata_c, iata_odd = code_conflict(g, "iata")
    cc_c, cc_odd = country_conflict(g)
    dist = max_pair_distance(g)
    members = " | ".join(
        f"{r.record_id}:{r.name}:{r.icao}:{r.iata}:{r.countryCode}"
        for r in g.itertuples()
    )
    rows.append({
        "group_id": gid,
        "size": len(g),
        "icao_conflict": icao_c, "icao_odd": icao_odd,
        "iata_conflict": iata_c, "iata_odd": iata_odd,
        "country_conflict": cc_c, "country_odd": cc_odd,
        "max_dist_km": round(dist, 2),
        "dist_flag": dist > DIST_KM,
        "members": members,
    })

res = pd.DataFrame(rows)
flag_cols = ["icao_conflict", "iata_conflict", "country_conflict", "dist_flag"]
res["n_flags"] = res[flag_cols].sum(axis=1)
res["suspicious"] = res["n_flags"] > 0

# ------------------------------------------------------------------
# 5. Summary
# ------------------------------------------------------------------
n = len(res)
print("\n=== Suspicious Group Summary ===")
for c in flag_cols:
    print(f"{c:18s} {int(res[c].sum()):6d}  ({res[c].mean():.2%})")
print(f"{'any flag':18s} {int(res['suspicious'].sum()):6d}  ({res['suspicious'].mean():.2%}) of {n} groups")
print(f"{'>= 2 flags':18s} {int((res['n_flags'] >= 2).sum()):6d}")

print("\nGroups by max distance (km):")
for t in (1, 5, 20, 50, 100, 500):
    print(f"  > {t:4d} km: {int((res['max_dist_km'] > t).sum())}")

print("\nOdd-one-out dataset (icao / iata / country):")
for c in ["icao_odd", "iata_odd", "country_odd"]:
    print(f"  {c}: {res.loc[res[c] != '', c].value_counts().to_dict()}")

by_size = res.groupby("size")["suspicious"].agg(["sum", "count"])
by_size["rate"] = (by_size["sum"] / by_size["count"]).round(4)
print("\nSuspicious rate by group size:")
print(by_size.to_string())

# ------------------------------------------------------------------
# 6. Save outputs
# ------------------------------------------------------------------
res.to_csv(out_dir / "group_quality_all.csv", index=False)
sus = res[res["suspicious"]].sort_values("n_flags", ascending=False)
sus.to_csv(out_dir / "suspicious_clusters.csv", index=False)

# Manual review sample: random suspicious groups with an empty label column
sample = sus.sample(min(SAMPLE_SIZE, len(sus)), random_state=RANDOM_SEED).copy()
sample["manual_label"] = ""     # fill in: wrong_match / right_match_data_error / unsure
sample.to_csv(out_dir / "suspicious_sample_for_review.csv", index=False)

# Optional experiment: split the odd-one-out record into its own entity
odd_ids = []
for r in sus.itertuples():
    odd_ds = {r.icao_odd, r.iata_odd, r.country_odd} - {"", "ambiguous"}
    if len(odd_ds) == 1:
        ds = odd_ds.pop()
        g = df[(df["group_id"] == r.group_id) & (df["dataset"] == ds)]
        odd_ids.extend(g["record_id"].tolist())

r2g_all = pd.read_csv(out_dir / "record_to_group.csv")
clean = r2g_all.copy()
mask = clean["record_id"].isin(odd_ids)
clean.loc[mask, "group_id"] = ["x" + str(i) for i in range(int(mask.sum()))]
clean.to_csv(out_dir / "record_to_group_split_odd.csv", index=False)
print(f"\nRecords split off in the optional cleaned mapping: {len(odd_ids)}")
print(f"Saved to: {out_dir}")