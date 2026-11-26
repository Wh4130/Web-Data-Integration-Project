from pathlib import Path
import inspect
import pandas as pd
import networkx as nx
import PyDI.fusion as fusion

# 1. Setup paths
ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

norm_dir = ROOT / "data" / "normalized_outputs"
ir_dir = ROOT / "data" / "identity_resolution" / "outputs"
out_dir = ROOT / "data" / "data_fusion" / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

# 2. Load datasets and align to the 11 target attributes
target_cols = [
    "name", "iata", "icao", "localCode", "lat", "lon",
    "elevation", "countryCode", "countryName", "city", "timezone"
]

oa = pd.read_csv(norm_dir / "ourairportsNormalized.csv", low_memory=False)
wd = pd.read_csv(norm_dir / "wikidataNormalized.csv", low_memory=False)
kg = pd.read_csv(norm_dir / "kaggleNormalized.csv", low_memory=False)

# Combine all records indexed by unique record ID for fast lookup
all_records = pd.concat([
    oa.reindex(columns=["id"] + target_cols),
    wd.reindex(columns=["id"] + target_cols),
    kg.reindex(columns=["id"] + target_cols),
], ignore_index=True).set_index("id")

# 3. Build record groups (clusters) from correspondences
c_oa_kg = pd.read_csv(ir_dir / "correspondences_oa_kg.csv")[["id1", "id2", "score"]]
c_wd_oa = pd.read_csv(ir_dir / "wd_oa_correspondences.csv")[["id1", "id2", "score"]]
all_corr = pd.concat([c_oa_kg, c_wd_oa], ignore_index=True)

G = nx.Graph()
G.add_edges_from(zip(all_corr["id1"], all_corr["id2"]))
clusters = list(nx.connected_components(G))

# Map each record ID to its cluster ID
id_to_cluster = {}
for cid, members in enumerate(clusters):
    for rec_id in members:
        id_to_cluster[rec_id] = cid

matched_df = all_records.loc[all_records.index.isin(id_to_cluster)].copy()
matched_df["cluster_id"] = matched_df.index.map(id_to_cluster)

# 4. Calculate pre-fusion Consistency per attribute across matched clusters
# A cluster is consistent on an attribute if it has <= 1 unique non-null value
consistency_rows = []
total_clusters = len(clusters)

for col in target_cols:
    non_null = matched_df[["cluster_id", col]].dropna()
    nunique_per_cluster = non_null.groupby("cluster_id")[col].nunique()
    
    # Clusters where at least 2 sources provide a value (overlapping values)
    value_counts = non_null.groupby("cluster_id")[col].count()
    overlapping_clusters = (value_counts >= 2).sum()
    conflicting_clusters = (nunique_per_cluster > 1).sum()
    non_conflicting_clusters = total_clusters - conflicting_clusters
    
    consistency_all = non_conflicting_clusters / total_clusters
    consistency_overlap = (
        (overlapping_clusters - conflicting_clusters) / overlapping_clusters
        if overlapping_clusters > 0 else 1.0
    )
    
    consistency_rows.append({
        "attribute": col,
        "total_matched_clusters": total_clusters,
        "overlapping_clusters": int(overlapping_clusters),
        "conflicting_clusters": int(conflicting_clusters),
        "consistency_overall": round(consistency_all, 4),
        "consistency_on_overlap": round(consistency_overlap, 4),
    })

df_consistency = pd.DataFrame(consistency_rows)
df_consistency.to_csv(out_dir / "pre_fusion_consistency.csv", index=False)

print("=== 1. Pre-Fusion Consistency by Attribute ===")
print(df_consistency.to_string(index=False))
print(f"\nMean Overall Consistency: {df_consistency['consistency_overall'].mean():.4f}")
print(f"Saved to: {out_dir / 'pre_fusion_consistency.csv'}")

# 5. Inspect PyDI.fusion method signatures for Step 4
print("\n=== 2. PyDI.fusion Signatures ===")
print("DataFusionStrategy.add_attribute_fuser:", inspect.signature(fusion.DataFusionStrategy.add_attribute_fuser))
print("DataFusionEngine.run:", inspect.signature(fusion.DataFusionEngine.run))
print("DataFusionEvaluator.evaluate:", inspect.signature(fusion.DataFusionEvaluator.evaluate))