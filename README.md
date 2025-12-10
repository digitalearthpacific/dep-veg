## Vegetation Height Estimation Product for the Pacific Region

Processor is at src/utils.py/VegProcessorKeepNonVegPixels

`python src/run_task.py --tile-id 45,55 --version 1.0.0 --output-bucket dep-public-staging --collection-url-root https://stac.staging.digitalearthpacific.io/collections --datetime 2022`

NOTE: datetime can be a year or a period between years (2023-2024) or a specific date (2024-10-01). Data is only available at quarterly start dates. If data isn't available, the code simply exits.

## workflow to run 
- In Production:

```
kind: Workflow
metadata:
  generateName: vegheight-
  namespace: argo
spec:
  entrypoint: workflow-entrypoint
  serviceAccountName: open-data-bucket-writer
  podGC:
    strategy: OnWorkflowSuccess
    deleteDelayDuration: 600s
  parallelism: 20
  podMetadata:
    labels:
      app: s2-vegheight
    annotations:
      karpenter.sh/do-not-disrupt: "true"
  arguments:
    parameters:
      - name: version
        value: "0.0.1" # The version of the data product being made
      - name: image-name
        value: "dep-veg" # The Docker image
      - name: base-product
        value: "s2"
      - name: image-tag
        value: "0.0.1-24-gf6a1c6d"
      - name: year
        value: "2017"
      - name: bucket
        value: "dep-public-data" 
        # The bucket to store the data
        # dep-public-staging
        # dep-public-data 
      - name: collection
        value: "https://stac.digitalearthpacific.org/collections"
      - name: overwrite
        value: "--no-overwrite" # Can be "--overwrite" or "--no-overwrite"
      - name: model
        value: "https://dep-public-staging.s3.us-west-2.amazonaws.com/dep_s2_vegheight/models/dep-veg-models.zip"
  templates:
    - name: workflow-entrypoint
      dag:
        tasks:
          - name: generate-ids
            template: generate
            arguments:
              parameters:
                - name: limit
                  value: "5"
                - name: year
                  value: "{{ workflow.parameters.year }}"
                - name: version
                  value: "{{ workflow.parameters.version }}"
                - name: bucket
                  value: "{{ workflow.parameters.bucket }}"
                - name: overwrite
                  value: "{{ workflow.parameters.overwrite }}"
          - name: process-id
            depends: generate-ids.Succeeded
            template: process
            arguments:
              parameters:
                - name: tile-id
                  value: "{{item}}"
                - name: year
                  value: "{{ workflow.parameters.year }}"
                - name: version
                  value: "{{ workflow.parameters.version }}"
                - name: bucket
                  value: "{{ workflow.parameters.bucket }}"
                - name: collection
                  value: "{{ workflow.parameters.collection }}"
                - name: overwrite
                  value: "{{ workflow.parameters.overwrite }}"
                - name: model
                  value: "{{ workflow.parameters.model }}"
            withParam: "{{ tasks.generate-ids.outputs.result }}"
    - name: generate
      inputs:
        parameters:
          - name: limit
          - name: year
          - name: version
          - name: bucket
          - name: overwrite
      container:
        image: "ghcr.io/digitalearthpacific/{{ workflow.parameters.image-name }}:{{ workflow.parameters.image-tag }}"
        imagePullPolicy: IfNotPresent
        resources:
          requests:
            memory: 100Mi
            cpu: 1.0
        command: [python]
        args:
          - src/print_tasks.py
          - --datetime
          - "{{ inputs.parameters.year }}"
          - --version
          - "{{ inputs.parameters.version }}"
          # - --limit
          # - "{{ inputs.parameters.limit }}"
          - --output-bucket
          - "{{ inputs.parameters.bucket }}"
          - "{{ inputs.parameters.overwrite }}"
    - name: process
      retryStrategy:
        retryPolicy: OnError
        limit: "2"
      inputs:
        parameters:
          - name: tile-id
          - name: year
          - name: version
          - name: bucket
          - name: collection
          - name: overwrite
          - name: model
      tolerations:
          - key: "nvidia.com/gpu"
            operator: "Exists"
            effect: "NoSchedule"
      container:
        image: "ghcr.io/digitalearthpacific/{{ workflow.parameters.image-name }}:{{ workflow.parameters.image-tag }}"
        imagePullPolicy: IfNotPresent
        resources:
          requests:
            cpu: 14
            memory: 48Gi
            nvidia.com/gpu: 1
          limits:
            cpu: 16
            memory: 60Gi
            nvidia.com/gpu: 1
        command: [python]
        args:
          - src/run_task.py
          - --tile-id
          - "{{ inputs.parameters.tile-id }}"
          - --datetime
          - "{{ inputs.parameters.year }}"
          - --version
          - "{{ inputs.parameters.version }}"
          - --output-bucket
          - "{{ inputs.parameters.bucket }}"
          - --collection-url-root
          - "{{ inputs.parameters.collection }}"
          - --model-zip-uri
          - "{{ inputs.parameters.model }}"
          - "{{ inputs.parameters.overwrite }}"
```


## NOTE:

location to upload new models in the future: 
https://us-west-2.console.aws.amazon.com/s3/upload/dep-public-staging?region=us-west-2&bucketType=general&prefix=dep_s2_vegheight/models/

current model download link: 

https://dep-public-staging.s3.us-west-2.amazonaws.com/dep_s2_vegheight/models/dep-veg-model-v1.zip
