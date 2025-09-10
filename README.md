## Vegetation Height Estimation Product for the Pacific Region

Processor is at src/utils.py/VegProcessor

`python src/run_task.py --model-zip-uri "https://dep-public-staging.s3.us-west-2.amazonaws.com/dep_s2_vegheight/models/dep-veg-models.zip" --tile-id 64,20 --version 0.0.0 --datetime 2024`


## NOTE:

location to upload new models in the future: 
https://us-west-2.console.aws.amazon.com/s3/upload/dep-public-staging?region=us-west-2&bucketType=general&prefix=dep_s2_vegheight/models/

current model download link: 

https://dep-public-staging.s3.us-west-2.amazonaws.com/dep_s2_vegheight/models/dep-veg-models.zip

https://us-west-2.console.aws.amazon.com/s3/object/dep-public-staging?region=us-west-2&bucketType=general&prefix=dep_s2_vegheight/models/dep-veg-models.zip

## Q&A

1. 

```
items = searcher.search(geobox)

print(f"Found {len(items)} items") # 6 items found

data = loader.load(items, geobox) # xr.Dataset 

```
6 found but when loading, only 1 dataset is returned. Why?

2. run_task.py

3. remember to edit the small details in print_task.py 

4. steps to set up to run on AWS? 

upload models to s3
Create Dockerfile, test locally
push to repo
edit yaml

