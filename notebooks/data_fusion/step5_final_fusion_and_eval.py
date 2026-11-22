from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import networkx as nx
import PyDI.fusion as fusion
from custom_resolvers import iana_timezone, name_medoid
from eval_rules_relaxed import (
    eval_country_relaxed,
    eval_timezone_relaxed,
    eval_name_relaxed,
    make_num_eval,
)

# 1. Setup paths
ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

norm_dir = ROOT / "data" / "normalized_outputs"
ir_dir = ROOT / "data" / "identity_resolution" / "outputs"
gold_path = ROOT / "data" / "data_fusion" / "golden_standard" / "gold_standard_final.xml"
out_dir = ROOT / "data" / "data_fusion" / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

# 2. Load gold standard XML
def load_gold_xml(xml_path: Path) -> pd.DataFrame:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []
    for elem in root.findall("Airport"):
        row = {"id": elem.attrib.get("id")}
        for child in elem:
            val = child.text.strip() if child.text and child.text.strip() else None
            row[child.tag] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    for num_col in ["lat", "lon", "elevation"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
    return df

gold_df = load_gold_xml(gold_path)

# 3. Load normalized datasets and attach Provenance metadata
target_cols = [
    "name", "iata", "icao", "localCode", "lat", "lon",
    "elevation", "countryCode", "countryName", "city", "timezone"
]
all_cols = ["id"] + target_cols

oa = pd.read_csv(norm_dir / "ourairportsNormalized.csv", low_memory=False).reindex(columns=all_cols)
wd = pd.read_csv(norm_dir / "wikidataNormalized.csv", low_memory=False).reindex(columns=all_cols)
kg = pd.read_csv(norm_dir / "kaggleNormalized.csv", low_memory=False).reindex(columns=all_cols)

for df in [oa, wd, kg]:
    for num_col in ["lat", "lon", "elevation"]:
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

oa.attrs.update({"dataset_name": "OurAirports", "trust_score": 0.9, "date": "2026-09-30"})
wd.attrs.update({"dataset_name": "Wikidata",    "trust_score": 0.7, "date": "2026-09-30"})
kg.attrs.update({"dataset_name": "Kaggle",      "trust_score": 0.5, "date": "2025-06-23"})

datasets = [oa, wd, kg]

# 4. Load correspondences and build canonical OA ID lookup for evaluation
c_oa_kg = pd.read_csv(ir_dir / "correspondences_oa_kg.csv")[["id1", "id2", "score"]]
c_wd_oa = pd.read_csv(ir_dir / "wd_oa_correspondences.csv")[["id1", "id2", "score"]]
all_corr = pd.concat([c_oa_kg, c_wd_oa], ignore_index=True)

G = nx.Graph()
G.add_edges_from(zip(all_corr["id1"], all_corr["id2"]))
clusters = list(nx.connected_components(G))
id_to_oa_id = {}
for cluster in clusters:
    oa_members = [m for m in cluster if str(m).startswith("oa_")]
    if oa_members:
        for m in cluster:
            id_to_oa_id[m] = oa_members[0]

def map_to_oa_id(fused_id) -> str:
    first_token = str(fused_id).split("+")[0].strip()
    return id_to_oa_id.get(first_token, first_token)

# 5. Define Strict & Relaxed evaluation functions
def eval_str_ci(v1, v2) -> bool:
    if pd.isna(v1) or pd.isna(v2):
        return False
    return str(v1).strip().lower() == str(v2).strip().lower()

eval_rules_strict = {
    "name": eval_str_ci,
    "icao": eval_str_ci,
    "countryCode": eval_str_ci,
    "timezone": eval_str_ci,
    "lat": make_num_eval(0.01),
    "lon": make_num_eval(0.01),
    "elevation": make_num_eval(5.0),
}

eval_rules_relaxed = {
    "name": eval_name_relaxed,
    "icao": eval_str_ci,
    "countryCode": eval_country_relaxed,
    "timezone": eval_timezone_relaxed,
    "lat": make_num_eval(0.01),
    "lon": make_num_eval(0.01),
    "elevation": make_num_eval(10.0),
}

# 6. Build the Final Combined DataFusionStrategy using the best function per attribute
final_strategy = fusion.DataFusionStrategy(name="final_airport_fusion")

# Evaluated attributes (updated based on Step 4 strict & relaxed comparison)
final_strategy.add_attribute_fuser("name", name_medoid, evaluation_function=eval_rules_strict["name"])
final_strategy.add_attribute_fuser("icao", fusion.voting, evaluation_function=eval_rules_strict["icao"])
final_strategy.add_attribute_fuser("countryCode", fusion.voting, evaluation_function=eval_rules_strict["countryCode"])
final_strategy.add_attribute_fuser("timezone", iana_timezone, evaluation_function=eval_rules_strict["timezone"])
final_strategy.add_attribute_fuser("lat", fusion.prefer_higher_trust, evaluation_function=eval_rules_strict["lat"])
final_strategy.add_attribute_fuser("lon", fusion.prefer_higher_trust, evaluation_function=eval_rules_strict["lon"])
final_strategy.add_attribute_fuser("elevation", fusion.median, evaluation_function=eval_rules_strict["elevation"])

# Additional attributes for slot filling
final_strategy.add_attribute_fuser("iata", fusion.prefer_higher_trust)
final_strategy.add_attribute_fuser("localCode", fusion.prefer_higher_trust)
final_strategy.add_attribute_fuser("countryName", fusion.most_complete)
final_strategy.add_attribute_fuser("city", fusion.most_complete)

# 7. Run final data fusion (include_singletons=True)
engine = fusion.DataFusionEngine(strategy=final_strategy)
fused_df = engine.run(
    datasets=datasets,
    correspondences=all_corr,
    id_column="id",
    include_singletons=True,
)
fused_df["eval_id"] = fused_df["id"].apply(map_to_oa_id)
fused_df[["id"] + target_cols].to_csv(out_dir / "fused_airports_final.csv", index=False)

# 8. Evaluate Strict & Relaxed Accuracy (per attribute & overall)
fd = fused_df.drop_duplicates("eval_id")
acc_rows = []
overall_acc = {}

for rule_set, rules in {"strict": eval_rules_strict, "relaxed": eval_rules_relaxed}.items():
    total_correct, total_count = 0, 0
    for attr, fn in rules.items():
        m = gold_df[["id", attr]].merge(
            fd[["eval_id", attr]], left_on="id", right_on="eval_id",
            how="left", suffixes=("_gold", "_fused")
        )
        m = m[m[f"{attr}_gold"].notna()]
        ok = m.apply(lambda r: fn(r[f"{attr}_fused"], r[f"{attr}_gold"]), axis=1)
        c, n = int(ok.sum()), int(len(m))
        total_correct += c
        total_count += n
        acc_rows.append({
            "rule_set": rule_set, "attribute": attr,
            "accuracy": round(c / n, 4), "correct": c, "total": n
        })
    overall = round(total_correct / total_count, 4)
    overall_acc[rule_set] = overall
    acc_rows.append({
        "rule_set": rule_set, "attribute": "OVERALL",
        "accuracy": overall, "correct": total_correct, "total": total_count
    })

df_acc = pd.DataFrame(acc_rows)
df_acc.to_csv(out_dir / "final_accuracy_by_attribute.csv", index=False)

# 9. Compute Post-Fusion Density & Final Report Summary Metrics
oa_aligned = oa[target_cols]
fused_aligned = fused_df[target_cols]

oa_density = oa_aligned.notna().sum().sum() / (oa_aligned.shape[0] * oa_aligned.shape[1])
fused_density = fused_aligned.notna().sum().sum() / (fused_aligned.shape[0] * fused_aligned.shape[1])

col_density_cmp = pd.DataFrame({
    "attribute": target_cols,
    "OurAirports_density": oa_aligned.notna().mean().round(4).values,
    "Fused_density": fused_aligned.notna().mean().round(4).values,
})
col_density_cmp["density_change"] = (col_density_cmp["Fused_density"] - col_density_cmp["OurAirports_density"]).round(4)
col_density_cmp.to_csv(out_dir / "post_fusion_density_comparison.csv", index=False)

summary_metrics = {
    "largest_input_dataset": "OurAirports",
    "largest_input_entities": len(oa),
    "final_fused_entities": len(fused_df),
    "additional_entities_added": len(fused_df) - len(oa),
    "largest_input_density": round(oa_density, 4),
    "final_fused_density": round(fused_density, 4),
    "density_increase_absolute": round(fused_density - oa_density, 4),
    "overall_accuracy_strict": overall_acc["strict"],
    "overall_accuracy_relaxed": overall_acc["relaxed"],
}
pd.DataFrame([summary_metrics]).to_csv(out_dir / "final_summary_metrics.csv", index=False)

# 10. Compute Pre-Fusion Consistency across matched clusters
all_records = pd.concat([oa, wd, kg], ignore_index=True).set_index("id")
cons_rows = []
for col in target_cols:
    comparable, consistent = 0, 0
    for cluster in clusters:
        vals = all_records.reindex(list(cluster))[col].dropna()
        if len(vals) >= 2:
            comparable += 1
            if vals.astype(str).str.strip().str.lower().nunique() == 1:
                consistent += 1
    cons_rows.append({
        "attribute": col,
        "comparable_clusters": comparable,
        "consistent_clusters": consistent,
        "pre_fusion_consistency": round(consistent / comparable, 4) if comparable else None,
    })
df_cons = pd.DataFrame(cons_rows)
df_cons.to_csv(out_dir / "pre_fusion_consistency.csv", index=False)

# 11. Generate Error Analysis table (with strict & relaxed flags)
fused_lookup = fd.set_index("eval_id")
error_rows = []
for _, gold_row in gold_df.iterrows():
    gid = gold_row["id"]
    if gid not in fused_lookup.index:
        continue
    f_row = fused_lookup.loc[gid]
    for attr in eval_rules_strict:
        g_val = gold_row[attr]
        if pd.isna(g_val):
            continue
        f_val = f_row[attr]
        ok_strict = eval_rules_strict[attr](f_val, g_val)
        ok_relaxed = eval_rules_relaxed[attr](f_val, g_val)
        if not ok_strict:
            error_rows.append({
                "id": gid,
                "attribute": attr,
                "fused_value": f_val,
                "gold_value": g_val,
                "pass_relaxed": ok_relaxed,
            })

df_errors = pd.DataFrame(error_rows)
df_errors.to_csv(out_dir / "error_analysis.csv", index=False)

print("=== 1. Final Report Summary Metrics ===")
for k, v in summary_metrics.items():
    print(f"  {k:28s}: {v}")

print("\n=== 2. Final Accuracy by Attribute (Strict vs. Relaxed) ===")
print(df_acc.pivot(index="attribute", columns="rule_set", values="accuracy").to_string())

print("\n=== 3. Density Comparison (OurAirports vs. Final Fused) ===")
print(col_density_cmp.to_string(index=False))

print("\n=== 4. Pre-Fusion Consistency Across Matched Clusters ===")
print(df_cons.to_string(index=False))

print(f"\nAll final outputs saved to: {out_dir}")