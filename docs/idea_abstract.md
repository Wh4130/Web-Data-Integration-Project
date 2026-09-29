# Idea Abstract: Airport Data Integration

## Topic

Integration of airport data from three heterogeneous, independently maintained
web sources into a single, fused dataset that is more complete and more
accurate than any single source alone.

## Data Sources

| Source | Type / Access | Notes |
|---|---|---|
| **OurAirports** | Open, bulk CSV download (`airports.csv`, `runways.csv`, `countries.csv`, ...) | Community-maintained, global coverage; core reference attributes: ICAO/IATA/local codes, name, type, coordinates, elevation, country/region/municipality. |
| **Wikidata** | Linked Open Data, queried via SPARQL against a [QLever](https://qlever.dev/wikidata) mirror | Structured entity data (`instance of: airport`, Q1248784); profiling shows strong, low-conflict coverage for `country` (P17), `located in` (P131), and coordinates (P625), but sparse coverage for `operator`/`owner`. Also carries IATA/ICAO identifiers (P238/P239) — reserved as an evaluation signal, not as the matching mechanism (see Planned Approach, step 3). |
| **Kaggle Airports Dataset** | Static CSV download | Includes **timezone**, an attribute not present in OurAirports — the main added value of this source; schema and coverage still to be profiled. |

## Motivation

No single source is complete on its own:
- OurAirports has the broadest coverage of airports worldwide but lacks attributes like timezone and some semantic/linked metadata.
- Wikidata offers rich, linked semantic attributes but with inconsistent completeness across records.
- The Kaggle dataset contributes the timezone attribute, likely derived from a different pipeline (e.g. geocoding), which is not directly available from OurAirports.

Integrating the three sources should yield a dataset that is both broader in
coverage and richer in attributes than any individual source.

## Planned Approach

1. **Data profiling** — analyze each source's schema, attribute coverage,
   value formats, and data quality (completeness, consistency of identifiers
   such as IATA/ICAO codes) to confirm all three meet the standards required
   for integration.
2. **Schema mapping** — align attributes across sources (e.g. `iata_code` /
   `ident` in OurAirports vs. the corresponding Wikidata property, vs. the
   Kaggle column naming).
3. **Identity resolution / record linkage** — match airport records across
   sources using similarity over name, coordinates, and country/location —
   deliberately *not* using ICAO/IATA codes as the matching mechanism, since a
   shared near-complete identifier would trivialize the matching problem the
   project is meant to exercise. Instead, treat airports where ICAO agrees
   exactly across sources as a ground-truth set to evaluate the similarity
   matcher's precision/recall.
4. **Data fusion** — resolve conflicting attribute values across sources
   (e.g. differing coordinates or names) using a fusion strategy, and produce
   the final integrated airport dataset, including the added timezone
   attribute.
5. **Evaluation** — assess completeness and correctness gains of the fused
   dataset versus each individual source.

## Open Questions / Next Steps

- Define concrete standards/criteria for source data quality (schema
  analysis, to be discussed next).
- Confirm exact Kaggle dataset (URL/version) and its provenance for the
  timezone attribute.
- Decide on matching key priority and conflict-resolution rules for fusion.
