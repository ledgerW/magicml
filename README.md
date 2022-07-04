# Add Newly Released Set  
### Get New Card Data  
1. [MTGJSON](https://mtgjson.com/downloads/all-files/)  
  - Get the **AllPrintingsCSVFiles.zip**  
  - Unzip and replace existing files in `data/mtgjson`  
2. [Scryfall](https://scryfall.com/docs/api/bulk-data)  
  - Get the **Default Cards**  
  - Rename to **cards.json** and move to `data/scryfall`  

### Upload to S3  
1. Update and Upload local `data/supported_sets.txt` to `s3://magicml-raw-data.dev/supported_sets`  
1. Upload local `data/mtgjson` to `s3://magicml-raw-data.dev/mtgjson`  
2. Upload local `data/scryfall` to `s3://magicml-raw-data.dev/scryfall`  
  - `set AWS_PROFILE=lw2134`  
  - `aws s3 sync data/supported_sets s3://magicml-raw-data.dev/supported_sets`
  - `aws s3 sync data/mtgjson s3://magicml-raw-data.dev/mtgjson`  
  - `aws s3 sync data/scryfall s3://magicml-raw-data.dev/scryfall`   

 
### Run Data Prep Service : Cards Prep 
1. Execute `data-prep:cards_prep` function  
  - with an HTTP call - OR  
  - has S3 trigger on s3/raw_bucket/scryfall/*.json  


### Run GPL Fine Tuning on Corpus with New Set  
- Currently running manually:  
  - TODO: update training to get corpus from `s3://magicml-clean-data.dev/cards/corpus.jsonl`
  - `cd services/similarity`  
  - `python src/sm_train.py [PARAMS]`


### Run Similarity Service : Get Embeddings  
1. Execute `similarity:similarity.get_embeddings` function  
  - with HTTP call - OR    
  - set up as Lambda Destination from Cards Prep  


### Run Similarity Service : Stage Embed Master  
1. Execute `similarity:similarity.stage_embed_master` function  
  - with HTTP call - OR    
  - has S3 trigger on s3/inference_bucket/embeddings_matrix.parquet  


### Promote to Prod  
1. If everything looks good on `dev`, then repeat for `prod`  


## Manual pip install steps for GPL  
1. activate magicml conda env: `activate magicml`  
2. `git clone https://github.com/beir-cellar/beir.git`  
3. `pip install -e ./beir`  
4. `pip install easy-elasticsearch>=0.0.7`  
5. `pip install gpl --no-deps`