import requests
import pandas as pd

def run_sparql_query(query: str, 
                     endpoint: str = "https://qlever.dev/api/dbpedia") -> pd.DataFrame:
    resp = requests.get(
        endpoint,
        params={"query": query},
        headers={"Accept": "application/sparql-results+json"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    columns = data["head"]["vars"]
    rows = [
        {col: b.get(col, {}).get("value") for col in columns}
        for b in data["results"]["bindings"]
    ]
    return pd.DataFrame(rows)

if __name__ == "__main__":
    res = run_sparql_query(
        query = """PREFIX dbo:  <http://dbpedia.org/ontology/>
PREFIX geo:  <http://www.w3.org/2003/01/geo/wgs84_pos#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?n_distinct_lat (COUNT(?airport) AS ?num_airports)
WHERE {
  SELECT ?airport (COUNT(DISTINCT ?la) AS ?n_distinct_lat)
  WHERE {
    ?airport a dbo:Airport ;
             rdfs:label ?name .
    FILTER(LANG(?name) = "en")
    OPTIONAL { ?airport geo:lat ?la }
  }
  GROUP BY ?airport
}
GROUP BY ?n_distinct_lat
ORDER BY ?n_distinct_lat
""",
    endpoint = "https://qlever.dev/api/dbpedia"
    )
    print(res)