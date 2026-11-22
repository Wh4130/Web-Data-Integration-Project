"""Custom conflict resolution functions for PyDI (airport fusion).

PyDI calls a resolver as:
    resolver(values, sources=[record_ids], source_datasets={record_id: dataset},
             trust_map={dataset: trust}, group_id=..., attribute=...)
and expects: (value, confidence, metadata)
"""
import re
from collections import defaultdict
from difflib import SequenceMatcher

import pandas as pd

IANA_RE = re.compile(r"^[A-Za-z]+(?:/[A-Za-z0-9_+\-]+)+$")
STOP_WORDS = re.compile(r"\b(international|intl|airport)\b")


def _rows(values, sources, source_datasets, trust_map):
    """Aligned (value, dataset, trust) rows without missing values."""
    rows = []
    for i, v in enumerate(values):
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            continue
        if isinstance(v, str) and not v.strip():
            continue
        rid = sources[i] if sources and i < len(sources) else None
        ds = (source_datasets or {}).get(rid, "unknown")
        rows.append((v, ds, float((trust_map or {}).get(ds, 1.0))))
    return rows


def trusted_vote(values, *, sources=None, source_datasets=None, trust_map=None, **kw):
    """Weighted voting where each vote counts as the trust of its source.
    (PyDI's built-in weighted_voting gets no weights from the engine and
    therefore behaves like plain voting.)"""
    rows = _rows(values, sources, source_datasets, trust_map)
    if not rows:
        return None, 0.0, {"reason": "no_valid_values"}
    score = defaultdict(float)
    for v, _, t in rows:
        score[v] += t
    best = max(score.items(), key=lambda kv: (kv[1], kv[0] == rows[0][0]))
    conf = best[1] / sum(score.values())
    return best[0], conf, {"rule": "trusted_vote", "scores": dict(score)}


def iana_timezone(values, *, sources=None, source_datasets=None, trust_map=None, **kw):
    """Only IANA ids (e.g. Europe/Berlin) are valid timezone values.
    Wikidata values like 'UTC+01:00' or 'Eastern Time Zone' are ignored.
    Among valid values, the most trusted source wins."""
    rows = [r for r in _rows(values, sources, source_datasets, trust_map)
            if isinstance(r[0], str) and IANA_RE.match(r[0].strip())]
    if not rows:
        return None, 0.0, {"reason": "no_iana_value"}
    v, ds, t = max(rows, key=lambda r: r[2])
    return v.strip(), 1.0 if len({r[0] for r in rows}) == 1 else 0.7, {
        "rule": "iana_timezone", "dataset": ds}


def _norm_name(s):
    s = STOP_WORDS.sub(" ", str(s).lower())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def name_medoid(values, *, sources=None, source_datasets=None, trust_map=None, **kw):
    """Pick the name most similar to all other names (the 'medoid'), so a
    short Kaggle name or a Wikidata typo loses against the consensus.
    Ties: higher trust, then longer string."""
    rows = _rows(values, sources, source_datasets, trust_map)
    if not rows:
        return None, 0.0, {"reason": "no_valid_values"}
    if len(rows) == 1:
        return rows[0][0], 0.5, {"rule": "name_medoid", "reason": "single_value"}

    def score(i):
        v = rows[i][0]
        sims = [SequenceMatcher(None, _norm_name(v), _norm_name(w[0])).ratio()
                for j, w in enumerate(rows) if j != i]
        return (sum(sims) / len(sims), rows[i][2], len(str(v)))

    best = max(range(len(rows)), key=score)
    return rows[best][0], score(best)[0], {"rule": "name_medoid"}
