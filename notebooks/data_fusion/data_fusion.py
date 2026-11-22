from pathlib import Path
from collections import Counter
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import PyDI.fusion as fusion

# ------------------------------------------------------------------
# 0. Configuration
# ------------------------------------------------------------------
ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

norm_dir = ROOT / "data" / "normalized_outputs"
ir_dir = ROOT / "data" / "identity_resolution" / "outputs"
out_dir = ROOT / "data" / "data_fusion" / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

# Name of the ID column in each normalized file.
# CHECK THIS: the script prints the columns if the name is wrong.
ID_COL = {"OurAirports": "id", "Wikidata": "id", "Kaggle": "id"}

# Optional: drop weak correspondences (None = keep all)
SCORE_THRESHOLD = None

# Values that look filled but really mean "missing"
PLACEHOLDERS = {"", "nan", "none", "null", "n/a", "na", "unknown", "-", "--"}

target_cols = [
    "name", "iata", "icao", "localCode", "lat", "lon",
    "elevation", "countryCode", "countryName", "city", "timezone",
]

# ------------------------------------------------------------------
# 1. Load datasets and validate IDs
# ------------------------------------------------------------------
files = {
    "OurAirports": "ourairportsNormalized.csv",
    "Wikidata": "wikidataNormalized.csv",
    "Kaggle": "kaggleNormalized.csv",
}
datasets = {k: pd.read_csv(norm_dir / v, low_memory=False) for k, v in files.items()}

id2ds = {}          # global map: record id -> dataset name
for ds_name, df in datasets.items():
    col = ID_COL[ds_name]
    if col not in df.columns:
        raise KeyError(
            f"[{ds_name}] ID column '{col}' not found. Columns: {list(df.columns)}"
        )
    ids = df[col].astype(str)
    if ids.duplicated().any():
        raise ValueError(f"[{ds_name}] duplicate IDs inside the dataset")
    df[col] = ids
    for i in ids:
        if i in id2ds:
            raise ValueError(
                f"ID '{i}' appears in both {id2ds[i]} and {ds_name}: "
                "IDs are not globally unique (add a dataset prefix)."
            )
        id2ds[i] = ds_name
print("ID check passed: IDs are unique inside and across datasets.")

# ------------------------------------------------------------------
# 2. Pre-fusion density (placeholders count as missing)
# ------------------------------------------------------------------
def clean_missing(df):
    out = df.reindex(columns=target_cols).copy()
    for c in out.columns:
        if out[c].dtype == object:
            s = out[c].astype(str).str.strip().str.lower()
            out.loc[s.isin(PLACEHOLDERS), c] = pd.NA
    return out

overall_rows, col_density = [], {}
for ds_name, df in datasets.items():
    aligned = clean_missing(df)
    dens = aligned.notna().sum().sum() / (aligned.shape[0] * aligned.shape[1])
    overall_rows.append({
        "dataset": ds_name,
        "num_entities": len(df),
        "target_attributes": len(target_cols),
        "overall_density": round(dens, 4),
    })
    col_density[ds_name] = aligned.notna().mean().round(4)

df_overall_density = pd.DataFrame(overall_rows)
df_col_density = pd.DataFrame(col_density)
df_col_density.index.name = "attribute"
df_overall_density.to_csv(out_dir / "pre_fusion_density_overall.csv", index=False)
df_col_density.to_csv(out_dir / "pre_fusion_density_by_column.csv")

print("\n=== 1. Pre-Fusion Density ===")
print(df_overall_density.to_string(index=False))

# ------------------------------------------------------------------
# 3. Correspondences -> graph -> groups
# ------------------------------------------------------------------
c_oa_kg = pd.read_csv(ir_dir / "correspondences_oa_kg.csv")[["id1", "id2", "score"]]
c_wd_oa = pd.read_csv(ir_dir / "wd_oa_correspondences.csv")[["id1", "id2", "score"]]
all_corr = pd.concat([c_oa_kg, c_wd_oa], ignore_index=True)
all_corr["id1"] = all_corr["id1"].astype(str)
all_corr["id2"] = all_corr["id2"].astype(str)

# Every correspondence ID must exist in some dataset
missing = set(all_corr["id1"]) | set(all_corr["id2"])
missing = {i for i in missing if i not in id2ds}
if missing:
    raise ValueError(
        f"{len(missing)} correspondence IDs not found in any dataset, e.g. {list(missing)[:5]}"
    )

n_raw = len(all_corr)
if SCORE_THRESHOLD is not None:
    all_corr = all_corr[all_corr["score"] >= SCORE_THRESHOLD]
all_corr = all_corr[all_corr["id1"] != all_corr["id2"]]
all_corr = all_corr.drop_duplicates(subset=["id1", "id2"])
print(f"\nCorrespondences: {n_raw} raw -> {len(all_corr)} after cleaning")

G = nx.Graph()
G.add_edges_from(zip(all_corr["id1"], all_corr["id2"]))
clusters = list(nx.connected_components(G))

# ------------------------------------------------------------------
# 4. Singletons and total number of entities
# ------------------------------------------------------------------
matched_ids = set(G.nodes)
singletons = [i for i in id2ds if i not in matched_ids]
single_by_ds = Counter(id2ds[i] for i in singletons)

n_groups = len(clusters)
n_entities = n_groups + len(singletons)
largest_name = max(datasets, key=lambda k: len(datasets[k]))
largest_size = len(datasets[largest_name])

print("\n=== 2. Entity Counts ===")
print(f"Matched groups:          {n_groups}")
print(f"Unmatched singletons:    {len(singletons)}  {dict(single_by_ds)}")
print(f"Fused entities (total):  {n_entities}")
print(f"Largest input ({largest_name}): {largest_size}")
print(f"Additional entities:     {n_entities - largest_size}")

pd.DataFrame([{
    "matched_groups": n_groups,
    "singletons": len(singletons),
    "fused_entities": n_entities,
    "largest_input": largest_name,
    "largest_input_size": largest_size,
    "additional_entities": n_entities - largest_size,
}]).to_csv(out_dir / "entity_counts.csv", index=False)

# ------------------------------------------------------------------
# 5. Group size distribution (with and without singletons)
# ------------------------------------------------------------------
size_dist = Counter(len(c) for c in clusters)
size_dist_all = Counter(size_dist)
size_dist_all[1] += len(singletons)

df_group_dist = pd.DataFrame(
    [{"group_size": s, "frequency": size_dist[s]} for s in sorted(size_dist)]
)
df_group_dist_all = pd.DataFrame(
    [{"group_size": s, "frequency": size_dist_all[s]} for s in sorted(size_dist_all)]
)
df_group_dist.to_csv(out_dir / "group_size_distribution.csv", index=False)
df_group_dist_all.to_csv(out_dir / "group_size_distribution_with_singletons.csv", index=False)

print("\n=== 3. Group Size Distribution (matched groups only) ===")
print(df_group_dist.to_string(index=False))

plt.figure(figsize=(7, 4))
plt.bar(df_group_dist["group_size"].astype(str), df_group_dist["frequency"], color="#4A90E2")
plt.xlabel("Group Size")
plt.ylabel("Frequency")
plt.title("Correspondences Group Size Distribution")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(out_dir / "group_size_distribution.png", dpi=300)
plt.close()

# ------------------------------------------------------------------
# 6. Group composition: which datasets are in each group?
# ------------------------------------------------------------------
short = {"OurAirports": "OA", "Wikidata": "WD", "Kaggle": "KG"}
comp_rows, suspicious = [], []
for gid, members in enumerate(clusters):
    cnt = Counter(id2ds[m] for m in members)
    comp = "+".join(f"{short[d]}x{n}" if n > 1 else short[d] for d, n in sorted(cnt.items()))
    comp_rows.append({"group_id": gid, "size": len(members), "composition": comp})
    if any(n > 1 for n in cnt.values()):
        suspicious.append({"group_id": gid, "composition": comp, "members": "|".join(sorted(members))})

df_comp = pd.DataFrame(comp_rows).groupby("composition").size().reset_index(name="groups")
df_comp = df_comp.sort_values("groups", ascending=False)
df_comp.to_csv(out_dir / "group_composition.csv", index=False)
pd.DataFrame(suspicious).to_csv(out_dir / "suspicious_groups.csv", index=False)

print("\n=== 4. Group Composition ===")
print(df_comp.to_string(index=False))
print(f"\nSuspicious groups (same dataset more than once): {len(suspicious)}"
      f"  -> {out_dir / 'suspicious_groups.csv'}")

# ------------------------------------------------------------------
# 7. Save record -> group mapping for the fusion step
#    Singletons get their own group id so every record is covered.
# ------------------------------------------------------------------
rows = []
for gid, members in enumerate(clusters):
    for m in members:
        rows.append({"record_id": m, "dataset": id2ds[m], "group_id": f"g{gid}"})
for k, m in enumerate(singletons):
    rows.append({"record_id": m, "dataset": id2ds[m], "group_id": f"s{k}"})
pd.DataFrame(rows).to_csv(out_dir / "record_to_group.csv", index=False)

# ------------------------------------------------------------------
# 8. PyDI.fusion interfaces (for the next step)
# ------------------------------------------------------------------
print("\n=== 5. Available PyDI.fusion Interfaces ===")
print([x for x in dir(fusion) if not x.startswith("_")])
print(f"\nAll results saved to: {out_dir}")
