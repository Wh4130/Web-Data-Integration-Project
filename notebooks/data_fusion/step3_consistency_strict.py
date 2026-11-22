from pathlib import Path
import re
import pandas as pd
import networkx as nx

# ------------------------------------------------------------------
# 1. Setup paths
# ------------------------------------------------------------------
ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

norm_dir = ROOT / "data" / "normalized_outputs"
ir_dir = ROOT / "data" / "identity_resolution" / "outputs"
out_dir = ROOT / "data" / "data_fusion" / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

target_cols = [
    "name", "iata", "icao", "localCode", "lat", "lon",
    "elevation", "countryCode", "countryName", "city", "timezone",
]

# Tolerances for the relaxed version
TOL = {"lat": 0.01, "lon": 0.01, "elevation": 10}   # degrees, degrees, metres
NUMERIC_COLS = set(TOL)
FUZZY_STR_COLS = {"name", "city", "countryName"}     # normalized before comparing
PLACEHOLDERS = {"", "nan", "none", "null", "n/a", "na", "unknown", "-", "--"}

# ------------------------------------------------------------------
# 2. Load datasets
# ------------------------------------------------------------------
oa = pd.read_csv(norm_dir / "ourairportsNormalized.csv", low_memory=False)
wd = pd.read_csv(norm_dir / "wikidataNormalized.csv", low_memory=False)
kg = pd.read_csv(norm_dir / "kaggleNormalized.csv", low_memory=False)

all_records = pd.concat(
    [d.reindex(columns=["id"] + target_cols) for d in (oa, wd, kg)],
    ignore_index=True,
)
all_records["id"] = all_records["id"].astype(str)
all_records = all_records.set_index("id")

# Treat placeholders as missing; make numeric columns numeric
for c in target_cols:
    if c in NUMERIC_COLS:
        all_records[c] = pd.to_numeric(all_records[c], errors="coerce")
    else:
        s = all_records[c].astype("string").str.strip()
        all_records[c] = s.mask(s.str.lower().isin(PLACEHOLDERS))

# ------------------------------------------------------------------
# 3. Build clusters from correspondences
# ------------------------------------------------------------------
c_oa_kg = pd.read_csv(ir_dir / "correspondences_oa_kg.csv")[["id1", "id2"]]
c_wd_oa = pd.read_csv(ir_dir / "wd_oa_correspondences.csv")[["id1", "id2"]]
all_corr = pd.concat([c_oa_kg, c_wd_oa], ignore_index=True).astype(str)

G = nx.Graph()
G.add_edges_from(zip(all_corr["id1"], all_corr["id2"]))
clusters = list(nx.connected_components(G))

id_to_cluster = {rid: cid for cid, members in enumerate(clusters) for rid in members}
matched_df = all_records.loc[all_records.index.isin(id_to_cluster)].copy()
matched_df["cluster_id"] = matched_df.index.map(id_to_cluster)
total_clusters = len(clusters)
print(f"Matched clusters: {total_clusters}")

# ------------------------------------------------------------------
# 4. Conflict definitions
# ------------------------------------------------------------------
def norm_str(s):
    s = str(s).lower()
    s = re.sub(r"\b(international|intl|airport)\b", " ", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def strict_conflict(vals):
    """More than one distinct non-null value."""
    return vals.dropna().nunique() > 1


def relaxed_conflict(vals, col):
    vals = vals.dropna()
    if len(vals) < 2:
        return False
    if col in TOL:
        return (vals.max() - vals.min()) > TOL[col]
    if col in FUZZY_STR_COLS:
        return len({norm_str(v) for v in vals}) > 1
    return len({str(v).strip().lower() for v in vals}) > 1


# ------------------------------------------------------------------
# 5. Consistency per attribute (strict and relaxed)
# consistency_on_overlap = (overlapping - conflicting) / overlapping
#   overlapping  = clusters where >= 2 records have a value
#   conflicting  = clusters where those values disagree
# ------------------------------------------------------------------
rows = []
conflict_cluster_ids = {}   # for the examples export

for col in target_cols:
    g = matched_df.groupby("cluster_id")[col]
    n_values = g.count()
    overlapping = int((n_values >= 2).sum())

    strict_flags = g.apply(strict_conflict)
    relaxed_flags = g.apply(lambda v: relaxed_conflict(v, col))
    n_strict = int(strict_flags.sum())
    n_relaxed = int(relaxed_flags.sum())
    conflict_cluster_ids[col] = list(relaxed_flags[relaxed_flags].index)

    def cons(n_conf):
        return round((overlapping - n_conf) / overlapping, 4) if overlapping else 1.0

    rows.append({
        "attribute": col,
        "total_matched_clusters": total_clusters,
        "overlapping_clusters": overlapping,
        "strict_conflicts": n_strict,
        "strict_consistency_on_overlap": cons(n_strict),
        "relaxed_conflicts": n_relaxed,
        "relaxed_consistency_on_overlap": cons(n_relaxed),
    })

df_cons = pd.DataFrame(rows)
df_cons.to_csv(out_dir / "pre_fusion_consistency.csv", index=False)

print("\n=== Pre-Fusion Consistency (on overlap) ===")
print(df_cons.to_string(index=False))
print(f"\nMean strict:  {df_cons['strict_consistency_on_overlap'].mean():.4f}")
print(f"Mean relaxed: {df_cons['relaxed_consistency_on_overlap'].mean():.4f}")

# ------------------------------------------------------------------
# 6. Export conflict examples for manual inspection / error analysis
# ------------------------------------------------------------------
EXAMPLE_COLS = ["iata", "icao", "countryCode", "countryName", "timezone", "name", "city"]
for col in EXAMPLE_COLS:
    ids = conflict_cluster_ids.get(col, [])[:50]
    sub = matched_df[matched_df["cluster_id"].isin(ids)][["cluster_id", col]].dropna()
    sub = sub.reset_index().sort_values(["cluster_id", "id"])
    sub.to_csv(out_dir / f"conflict_examples_{col}.csv", index=False)

print(f"\nConflict examples saved to: {out_dir}")