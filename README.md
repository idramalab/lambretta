# lambretta
Source code and labeled dataset for Lambretta

#Steps of Running the code and generating intermediate files.

- Run candidate_query_generator.py  (Running claimextractor.py would be needed if we started from raw tweets directly, without the claims extracted). But that's been already done, and training_claims.csv is a result of that. This should output candidate_queries.txt and dict_claim_query.json.
- Run fetch_results.py . Make sure to update the awk_source_path (path of the datasource) so that the awk_query function works properly. In case you are using ElasticSearch or other interfaces, you can update the function accordingly. Additionally, also specify `awk_output_export_path` which will be written as an intermediate output file and needed on further step. 
- Run generate_semantic_features.py. This needs file written in `awk_output_export_path` in `fetch_results.py` as input. The output of running this will be `results_scoring.json`.
- Run export_all_features.py. This needs the file `results_scoring.json` from previous step. This will write `export_ltr_train.txt` and `export_ltr_test.txt` , the files that shall go into the Learning To Rank Java programs.

# Dataset
The file _lambretta_dataset.json_ contains the dictionary of claims and list of tweets discussing the claim, alongside their moderation status (0 or 1). 
