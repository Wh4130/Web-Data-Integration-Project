1. Elevation unit of measure -> meter (OurAirports need to translate it)
2. Integrated schema is extended to include almost all columns from all datasets that are not semantically identical. (Done by Wally already)
3. Coord -> separate to lat, long two columns, don't use nested structure.
4. After completion, under `schema` folder, one datasource per sub-folder, under which there should be two files:
   - `<datasource>.mfd` file: the mapping format that elaborates how source data is mapped into the target schema.
   - `<datasource>_mapped.xml`: Data of the resulting xml file.
5. `Airport/@id` format is `{source_prefix}_{n}`, assigned in the mapping (not taken from a source column, since not every source has an id):
   - Prefix: `wd` = Wikidata, `oa` = OurAirports, `kg` = Kaggle.
   - `n` = 0-based row index in that source's interim CSV (first data row = 0, header excluded), e.g. `wd_531`, `oa_39281`, `kg_0`.
   - In MapForce: `concat(<prefix constant>, auto-number)` with auto-number starting at 0.


## Division of Labor:
- OurAirports: Zheng Fu, Jing
- Kaggle: Niklas 
- Wikidata: Borya, Wally
