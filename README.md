## Vegetation Height Estimation Product for the Pacific Region

Processor is at src/utils.py/VegProcessor

`python src/run_task.py --tile-id 45,55 --version 0.0.0 --output-bucket dep-public-staging --collection-url-root https://stac.staging.digitalearthpacific.org/collections --datetime 2022`


## NOTE:

location to upload new models in the future: 
https://us-west-2.console.aws.amazon.com/s3/upload/dep-public-staging?region=us-west-2&bucketType=general&prefix=dep_s2_vegheight/models/

current model download link: 

https://dep-public-staging.s3.us-west-2.amazonaws.com/dep_s2_vegheight/models/dep-veg-models.zip
