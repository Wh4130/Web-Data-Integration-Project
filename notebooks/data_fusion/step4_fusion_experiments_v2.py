from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import networkx as nx
import PyDI.fusion as fusion
from custom_resolvers import trusted_vote, iana_timezone, name_medoid
from eval_rules_relaxed import (
    eval_country_relaxed,
    eval_timezone_relaxed,
    eval_name_relaxed,
    make_num_eval,
)

# 1. 路径设置
ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

norm_dir = ROOT / "data" / "normalized_outputs"
ir_dir = ROOT / "data" / "identity_resolution" / "outputs"
gold_path = ROOT / "data" / "data_fusion" / "golden_standard" / "gold_standard_final.xml"
out_dir = ROOT / "data" / "data_fusion" / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

# 2. 读取 Gold Standard
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

# 3. 读取标准化数据集
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

oa.attrs.update({"dataset_name": "OurAirports", "trust_score": 0.9, "date": "2026-09-30"})
wd.attrs.update({"dataset_name": "Wikidata", "trust_score": 0.7, "date": "2026-09-30"})
kg.attrs.update({"dataset_name": "Kaggle", "trust_score": 0.5, "date": "2025-06-23"})
datasets = [oa, wd, kg]

# 4. 读取匹配关系
c_oa_kg = pd.read_csv(ir_dir / "correspondences_oa_kg.csv")[["id1", "id2", "score"]]
c_wd_oa = pd.read_csv(ir_dir / "wd_oa_correspondences.csv")[["id1", "id2", "score"]]
all_corr = pd.concat([c_oa_kg, c_wd_oa], ignore_index=True)

G = nx.Graph()
G.add_edges_from(zip(all_corr["id1"], all_corr["id2"]))
id_to_oa_id = {}
for cluster in nx.connected_components(G):
    oa_members = [m for m in cluster if str(m).startswith("oa_")]
    if oa_members:
        for m in cluster:
            id_to_oa_id[m] = oa_members[0]

def map_to_oa_id(fused_id) -> str:
    first_token = str(fused_id).split("+")[0].strip()
    return id_to_oa_id.get(first_token, first_token)

# 5. Strict 与 Relaxed 两套评估规则
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

# 6. 候选融合函数
candidates = {
    "name": {
        "longest_string": fusion.longest_string,
        "shortest_string": fusion.shortest_string,
        "voting": fusion.voting,
        "weighted_voting": fusion.weighted_voting,
        "prefer_higher_trust": fusion.prefer_higher_trust,
        "name_medoid": name_medoid,
        "trusted_vote": trusted_vote,
    },
    "icao": {
        "voting": fusion.voting,
        "most_complete": fusion.most_complete,
        "weighted_voting": fusion.weighted_voting,
        "prefer_higher_trust": fusion.prefer_higher_trust,
        "trusted_vote": trusted_vote,
    },
    "countryCode": {
        "voting": fusion.voting,
        "most_complete": fusion.most_complete,
        "weighted_voting": fusion.weighted_voting,
        "prefer_higher_trust": fusion.prefer_higher_trust,
        "trusted_vote": trusted_vote,
    },
    "timezone": {
        "voting": fusion.voting,
        "most_complete": fusion.most_complete,
        "weighted_voting": fusion.weighted_voting,
        "prefer_higher_trust": fusion.prefer_higher_trust,
        "iana_timezone": iana_timezone,
        "trusted_vote": trusted_vote,
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

# 7. 运行 Strict + Relaxed 实验（每个函数只融合一次，同时用两套规则评估）
results = []
elev_sweep_rows = []

for attr, funcs in candidates.items():
    for func_name, func in funcs.items():
        strategy = fusion.DataFusionStrategy(name=f"{attr}_{func_name}")
        strategy.add_attribute_fuser(attr, func, evaluation_function=eval_rules_strict[attr])
        engine = fusion.DataFusionEngine(strategy=strategy)
        fused_df = engine.run(
            datasets=datasets,
            correspondences=all_corr,
            id_column="id",
            include_singletons=False,
        )
        fused_df["eval_id"] = fused_df["id"].apply(map_to_oa_id)
        fd = fused_df.drop_duplicates("eval_id")

        m_base = gold_df[["id", attr]].merge(
            fd[["eval_id", attr]], left_on="id", right_on="eval_id",
            how="left", suffixes=("_gold", "_fused")
        )
        m_base = m_base[m_base[f"{attr}_gold"].notna()]

        for rule_set, rules in {"strict": eval_rules_strict, "relaxed": eval_rules_relaxed}.items():
            m = m_base.copy()
            m["correct"] = m.apply(
                lambda r: rules[attr](r[f"{attr}_fused"], r[f"{attr}_gold"]), axis=1
            )
            acc = m["correct"].mean()
            m[~m["correct"]].to_csv(
                out_dir / f"errors_{rule_set}_{attr}_{func_name}.csv", index=False
            )
            results.append({
                "rule_set": rule_set,
                "attribute": attr,
                "function": func_name,
                "accuracy": round(acc, 4),
                "correct": int(m["correct"].sum()),
                "total": int(len(m)),
            })
            print(f"[{rule_set:7s}][{attr:11s}] {func_name:20s} -> {acc:.4f}")

        # 顺便跑 elevation 的多容差扫描
        if attr == "elevation":
            for tol in [5.0, 10.0, 25.0, 50.0]:
                ev = make_num_eval(tol)
                ok = m_base.apply(lambda r: ev(r["elevation_fused"], r["elevation_gold"]), axis=1)
                elev_sweep_rows.append({
                    "function": func_name,
                    "tolerance_m": tol,
                    "accuracy": round(ok.mean(), 4),
                    "correct": int(ok.sum()),
                    "total": int(len(m_base)),
                })

# 8. 保存汇总 CSV
df_res = pd.DataFrame(results)
df_res.to_csv(out_dir / "fusion_accuracy_strict_vs_relaxed.csv", index=False)

df_elev = pd.DataFrame(elev_sweep_rows)
df_elev.to_csv(out_dir / "elevation_tolerance_sweep.csv", index=False)

print(f"\n已保存双口径对比表: {out_dir / 'fusion_accuracy_strict_vs_relaxed.csv'}")
print(f"已保存高程容差扫描表: {out_dir / 'elevation_tolerance_sweep.csv'}")