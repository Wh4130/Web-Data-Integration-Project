from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import networkx as nx
import PyDI.fusion as fusion
from custom_resolvers import trusted_vote, iana_timezone, name_medoid

# 1. Setup paths
ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

norm_dir = ROOT / "data" / "normalized_outputs"
ir_dir = ROOT / "data" / "identity_resolution" / "outputs"
gold_path = ROOT / "data" / "data_fusion" / "golden_standard" / "gold_standard_final.xml"
out_dir = ROOT / "data" / "data_fusion" / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

# 2. Parse gold_standard_final.xml into a DataFrame
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
print(f"Loaded gold standard: {len(gold_df)} entities")

# 3. Load normalized datasets and attach Provenance metadata
target_cols = [
    "id", "name", "iata", "icao", "localCode", "lat", "lon",
    "elevation", "countryCode", "countryName", "city", "timezone"
]

oa = pd.read_csv(norm_dir / "ourairportsNormalized.csv", low_memory=False).reindex(columns=target_cols)
wd = pd.read_csv(norm_dir / "wikidataNormalized.csv", low_memory=False).reindex(columns=target_cols)
kg = pd.read_csv(norm_dir / "kaggleNormalized.csv", low_memory=False).reindex(columns=target_cols)

for df in [oa, wd, kg]:
    for num_col in ["lat", "lon", "elevation"]:
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

# Provenance metadata: trust_score from Proposal profiling, dates from Git log & Kaggle API
oa.attrs["dataset_name"] = "OurAirports"
oa.attrs["trust_score"] = 0.9
oa.attrs["date"] = "2026-09-30"

wd.attrs["dataset_name"] = "Wikidata"
wd.attrs["trust_score"] = 0.7
wd.attrs["date"] = "2026-09-30"

kg.attrs["dataset_name"] = "Kaggle"
kg.attrs["trust_score"] = 0.5
kg.attrs["date"] = "2025-06-23"

datasets = [oa, wd, kg]

# 4. Load correspondences and build a lookup from any cluster member ID to its OurAirports (oa_) ID
c_oa_kg = pd.read_csv(ir_dir / "correspondences_oa_kg.csv")[["id1", "id2", "score"]]
c_wd_oa = pd.read_csv(ir_dir / "wd_oa_correspondences.csv")[["id1", "id2", "score"]]
all_corr = pd.concat([c_oa_kg, c_wd_oa], ignore_index=True)

G = nx.Graph()
G.add_edges_from(zip(all_corr["id1"], all_corr["id2"]))
id_to_oa_id = {}
for cluster in nx.connected_components(G):
    oa_members = [m for m in cluster if str(m).startswith("oa_")]
    if oa_members:
        canonical_id = oa_members[0]
        for m in cluster:
            id_to_oa_id[m] = canonical_id

def map_to_oa_id(fused_id) -> str:
    first_token = str(fused_id).split("+")[0].strip()
    return id_to_oa_id.get(first_token, first_token)

# 5. Define evaluation functions
def eval_str_ci(v1, v2) -> bool:
    if pd.isna(v1) or pd.isna(v2):
        return False
    return str(v1).strip().lower() == str(v2).strip().lower()

def make_num_eval(tol: float):
    def _eval(v1, v2) -> bool:
        if pd.isna(v1) or pd.isna(v2):
            return False
        try:
            return abs(float(v1) - float(v2)) <= tol
        except (ValueError, TypeError):
            return False
    return _eval

eval_rules = {
    "name": eval_str_ci,
    "icao": eval_str_ci,
    "countryCode": eval_str_ci,
    "timezone": eval_str_ci,
    "lat": make_num_eval(0.01),
    "lon": make_num_eval(0.01),
    "elevation": make_num_eval(5.0),
}

# 6. Define candidate conflict resolution functions (Content-based vs. Provenance-based)
candidates = {
    "name": {
        "longest_string": fusion.longest_string,
        "shortest_string": fusion.shortest_string,
        "voting": fusion.voting,
        "weighted_voting": fusion.weighted_voting,
        "prefer_higher_trust": fusion.prefer_higher_trust,
    },
    "icao": {
        "voting": fusion.voting,
        "most_complete": fusion.most_complete,
        "weighted_voting": fusion.weighted_voting,
        "prefer_higher_trust": fusion.prefer_higher_trust,
    },
    "countryCode": {
        "voting": fusion.voting,
        "most_complete": fusion.most_complete,
        "weighted_voting": fusion.weighted_voting,
        "prefer_higher_trust": fusion.prefer_higher_trust,
    },
    "timezone": {
        "voting": fusion.voting,
        "most_complete": fusion.most_complete,
        "weighted_voting": fusion.weighted_voting,
        "prefer_higher_trust": fusion.prefer_higher_trust,
    },
    "lat": {
        "average": fusion.average,
        "median": fusion.median,
        "prefer_higher_trust": fusion.prefer_higher_trust,
    },
    "lon": {
        "average": fusion.average,
        "median": fusion.median,
        "prefer_higher_trust": fusion.prefer_higher_trust,
    },
    "elevation": {
        "average": fusion.average,
        "median": fusion.median,
        "minimum": fusion.minimum,
        "maximum": fusion.maximum,
        "prefer_higher_trust": fusion.prefer_higher_trust,
    },
}
candidates["name"]["name_medoid"] = name_medoid
candidates["name"]["trusted_vote"] = trusted_vote
candidates["icao"]["trusted_vote"] = trusted_vote
candidates["countryCode"]["trusted_vote"] = trusted_vote
candidates["timezone"]["iana_timezone"] = iana_timezone
candidates["timezone"]["trusted_vote"] = trusted_vote

# 7. Run fusion & evaluation experiments
results = []

for attr, funcs in candidates.items():
    for func_name, func in funcs.items():
        strategy = fusion.DataFusionStrategy(name=f"{attr}_{func_name}")
        strategy.add_attribute_fuser(attr, func, evaluation_function=eval_rules[attr])
        
        engine = fusion.DataFusionEngine(strategy=strategy)
        fused_df = engine.run(
            datasets=datasets,
            correspondences=all_corr,
            id_column="id",
            include_singletons=False,
        )
        
        # Align fused cluster ID to its OurAirports ID for gold standard evaluation
        fused_df["eval_id"] = fused_df["id"].apply(map_to_oa_id)
        fd = fused_df.drop_duplicates("eval_id")
        m = gold_df[["id", attr]].merge(
            fd[["eval_id", attr]], left_on="id", right_on="eval_id",
            how="left", suffixes=("_gold", "_fused"))
        m = m[m[f"{attr}_gold"].notna()]
        m["correct"] = m.apply(
            lambda r: eval_rules[attr](r[f"{attr}_fused"], r[f"{attr}_gold"]), axis=1)
        m[~m["correct"]].to_csv(out_dir / f"errors_{attr}_{func_name}.csv", index=False)
        
        evaluator = fusion.DataFusionEvaluator(strategy=strategy)
        metrics = evaluator.evaluate(
            fused_df=fused_df,
            fused_id_column="eval_id",
            gold_df=gold_df,
            gold_id_column="id",
        )
        
        acc = metrics.get(f"{attr}_accuracy", metrics.get("overall_accuracy", None))
        results.append({
            "attribute": attr,
            "function": func_name,
            "accuracy": round(acc, 4) if acc is not None else None,
        })
        print(f"[{attr:11s}] {func_name:20s} -> Accuracy: {acc:.4f}")

# 8. Save comparison table
df_res = pd.DataFrame(results)
df_res.to_csv(out_dir / "fusion_accuracy_comparison.csv", index=False)

print("\n=== Summary: Best Function per Attribute ===")
best_df = df_res.sort_values("accuracy", ascending=False).groupby("attribute", as_index=False).first()
print(best_df.to_string(index=False))
print(f"\nSaved full comparison table to: {out_dir / 'fusion_accuracy_comparison.csv'}")