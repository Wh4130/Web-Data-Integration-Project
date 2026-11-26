from pathlib import Path
import pandas as pd


ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

norm_dir = ROOT / "data" / "normalized_outputs"
out_dir = ROOT / "data" / "data_fusion" / "outputs"
ir_dir = ROOT / "data" / "identity_resolution" / "outputs"

# 自动寻找 record_to_group.csv（可能在 outputs 或 identity_resolution 下）
r2g_path = out_dir / "record_to_group.csv"
if not r2g_path.exists():
    r2g_path = ir_dir / "record_to_group.csv"

r2g = pd.read_csv(r2g_path)

# 2. 合并三个源表，方便直接看每条记录的机场名、国家、经纬度和时区
cols = ["id", "name", "countryCode", "lat", "lon", "timezone", "elevation"]
oa = pd.read_csv(norm_dir / "ourairportsNormalized.csv", low_memory=False).reindex(columns=cols)
wd = pd.read_csv(norm_dir / "wikidataNormalized.csv", low_memory=False).reindex(columns=cols)
kg = pd.read_csv(norm_dir / "kaggleNormalized.csv", low_memory=False).reindex(columns=cols)
all_records = pd.concat([oa, wd, kg], ignore_index=True)

# 3. 查这 10 个时区错误所在的完整分组
ids = [
    "oa_23331", "oa_39944", "oa_42905", "oa_46254", "oa_47016",
    "oa_64619", "oa_65548", "oa_67052", "oa_80602", "oa_82121"
]

gid = r2g.set_index("record_id")["group_id"]
target_groups = r2g[r2g["group_id"].isin(gid.loc[ids])].copy()

# 如果存在 split_odd 文件，顺便把拆分后的 group_id 也拼进来对比
split_path = r2g_path.parent / "record_to_group_split_odd.csv"
if split_path.exists():
    r2g_split = pd.read_csv(split_path).rename(columns={"group_id": "group_id_split"})
    target_groups = target_groups.merge(r2g_split, on="record_id", how="left")

# 拼上源数据属性
res = target_groups.merge(all_records, left_on="record_id", right_on="id", how="left")
res = res.drop(columns=["id"]).sort_values(["group_id", "record_id"])

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(res.to_string(index=False))
res.to_csv(out_dir / "tz_errors_diagnosis.csv", index=False, encoding="utf-8-sig")
print(f"已保存到: {out_dir / 'tz_errors_diagnosis.csv'}")