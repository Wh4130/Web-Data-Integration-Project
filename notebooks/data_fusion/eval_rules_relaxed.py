import re
from datetime import datetime
from difflib import SequenceMatcher
from zoneinfo import ZoneInfo
import pandas as pd

# 领地 ISO 码 -> 主权国家 ISO 码映射
TERRITORY_TO_SOVEREIGN = {
    # France
    "PF": "FR", "NC": "FR", "WF": "FR", "PM": "FR", "GP": "FR",
    "BL": "FR", "MF": "FR", "MQ": "FR", "RE": "FR", "YT": "FR", "GF": "FR",
    # United States
    "MP": "US", "GU": "US", "UM": "US", "VI": "US", "PR": "US", "AS": "US",
    # Netherlands
    "BQ": "NL", "AW": "NL", "CW": "NL", "SX": "NL",
    # Australia
    "NF": "AU", "CX": "AU", "CC": "AU",
    # United Kingdom
    "TC": "GB", "KY": "GB", "AI": "GB", "VG": "GB", "BM": "GB",
    "GI": "GB", "FK": "GB", "MS": "GB", "SH": "GB",
    # New Zealand & Denmark
    "CK": "NZ", "NU": "NZ", "TK": "NZ", "FO": "DK", "GL": "DK",
}

STOPWORDS = {
    "airport", "international", "airfield", "aerodrome", "regional",
    "municipal", "national", "field", "heliport", "air", "base", "ab", "afb"
}

def eval_country_relaxed(v1, v2) -> bool:
    if pd.isna(v1) or pd.isna(v2):
        return False
    c1 = str(v1).strip().upper()
    c2 = str(v2).strip().upper()
    if c1 == c2:
        return True
    return (
        TERRITORY_TO_SOVEREIGN.get(c1) == c2
        or TERRITORY_TO_SOVEREIGN.get(c2) == c1
    )

def eval_timezone_relaxed(v1, v2) -> bool:
    if pd.isna(v1) or pd.isna(v2):
        return False
    t1, t2 = str(v1).strip(), str(v2).strip()
    if t1.lower() == t2.lower():
        return True
    try:
        ref = datetime(2026, 1, 15, 12, 0, 0)
        off1 = ZoneInfo(t1).utcoffset(ref)
        off2 = ZoneInfo(t2).utcoffset(ref)
        return off1 is not None and off1 == off2
    except Exception:
        return False

def _clean_name(s: str) -> str:
    s = re.sub(r"[^a-z0-9\s]", " ", str(s).lower())
    tokens = [t for t in s.split() if t not in STOPWORDS]
    return " ".join(tokens)

def eval_name_relaxed(v1, v2) -> bool:
    if pd.isna(v1) or pd.isna(v2):
        return False
    s1, s2 = _clean_name(v1), _clean_name(v2)
    if not s1 or not s2:
        return str(v1).strip().lower() == str(v2).strip().lower()
    if s1 == s2:
        return True
    set1, set2 = set(s1.split()), set(s2.split())
    if set1.issubset(set2) or set2.issubset(set1):
        return True
    return SequenceMatcher(None, s1, s2).ratio() >= 0.85

def make_num_eval(tol: float):
    def _eval(v1, v2) -> bool:
        if pd.isna(v1) or pd.isna(v2):
            return False
        try:
            return abs(float(v1) - float(v2)) <= tol
        except (ValueError, TypeError):
            return False
    return _eval