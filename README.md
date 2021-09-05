# Add Newly Released Set  
### Get New Card Data  
1. [MTGJSON](https://mtgjson.com/downloads/all-files/)  
  - Get the **AllPrintingsCSVFiles.zip**  
  - Unzip and replace existing files in `data/mtgjson`  
2. [Scryfall](https://scryfall.com/docs/api/bulk-data)  
  - Get the **Default Cards**  
  - Rename to **cards.json** and move to `data/scryfall`  

### Upload to S3  
1. Upload local `data/mtgjson` to `s3://magicml-raw-data.dev/mtgjson`  
2. Upload local `data/scryfall` to `s3://magicml-raw-data.dev/scryfall`  
  - `set AWS_PROFILE=lw2134`
  - `aws s3 sync data/mtgjson s3://magicml-raw-data.dev/mtgjson` 

### Add New Set Name  
1. Add the new set name to list at `libs/magic_lib.supported_sets`  

### Run Data Prep Service : Cards Prep 
1. Execute `data-prep:cards_prep` function  
  - currently with an HTTP call  
  - set up as S3 trigger or StepFunction  
    - when new data hits S3  

### Run Similarity Service : Get Embeddings  
1. Execute `similarity:similarity.get_embeddings` function  
  - currently with HTTP call  
  - set up as Lambda Destination / S3 Trigger / or StepFunction  

### Run Similarity Service : Stage Embed Master  
1. Execute `similarity:similarity.stage_embed_master` function  
  - currently with HTTP call  
  - set up as Lambda Destination / S3 Trigger / or StepFunction  

### Promote to Prod  
1. If everything looks good on `dev`, then repeat for `prod`