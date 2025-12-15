## Training and Open-Loop Testing Prediction Models

### Data Download and Format Unification

For training and in-distribution validation we use the official [nuScenes](https://github.com/nutonomy/nuscenes-devkit), [Argoverse2](https://github.com/argoverse/av2-api), [Waymo-Open](https://github.com/waymo-research/waymo-open-dataset) and [Shifts](https://github.com/Shifts-Project/shifts/) dataset splits. For out-of-distribution testing we use the official [NuPlan](https://github.com/motional/nuplan-devkit) validation split. Please refer to the official documentations for downloading these datasets.

After downloading the datasets we use [scenarionet repository](https://github.com/metadriverse/scenarionet) to convert them into a single dataset format. Please refer to the documentation of scenarionet. Its dependency is not included in this repo. We recommend reserving another environment for the scenarionet install and data conversion.

### Configuration

We use Hydra for the configuration of our dataloader, predictor models and run logic. You can inspect and set the respective default configurations in the [configs/predictor](configs/predictor/) folders yaml files. The priority of configuration overrides is set in the parent [training.yaml](predictors/configs/training.yaml) file. You can also override these default configurations using the cmd interface. To check if a configuration is successfully applied, you can run
```
 python predictor/run.py \
    --cfg job
```

### Data Preprocessing and Caching

During the first training run the script will automatically cache the preprocessed model inputs for each split selected for training, validation or testing. You can define the respective root paths for the scenarionet data source and the cached data target in the [configs/paths](configs/paths) file. You can configure different options for the preprocessing, such as prediction target types or map ranges, in the [predictors/configs/data](predictors/configs/data) config files.

### Training

We execute training, testing and/or predictions by running the [predictors/run.py](predictors/run.py) file. Per default, we use a DDP strategy for multi-GPU runs. E.g., with wayformer you can start a training run:

```
 python predictor/run.py \
    train=True \
    tune=False
    test=True \
    predict=False \
    predictor/model=wayformer \
    predictor/trainer=wayformer \
    predictor/data=marginal_all \
```

Configurations for AutobotEgo, MTR, Wayformer and their ablations are given in the [configs/predictor/model](../configs/predictor/model) folder. 

### Logging

We use [aim](https://github.com/aimhubio/aim) as logger for our experiments. You can configure the logging directory and artifact root paths in [configs/paths](../configs/paths) files. You can configure further logger options in the [configs/predictor/trainer](../configs/predictor/trainer) files. Any training run gets a unique run_hash. You can inspect loss and metrics in the aimUI by navigating to your logging directory and executing
```
 aim up
```

### Open-Loop Prediction

A prediction callback writes per-sample metrics for all configured datasplits into the result directory, stored as dataframe inside a parquet file, given a suitable checkpoint. Metrics are configured in the [configs/predictor/metrics/](../configs/predictor/metrics/predict.yaml) files

```
 python predictor/run.py \
    train=False \
    tune=False
    test=False \
    predict=True \
    predictor/model=wayformer \
    predictor/trainer=wayformer \
    predictor/data=marginal_all \
    predictor.trainer.resume_ckpt_path=>insert_your_checkpoint< \
```

## Configurations

You can easily switch out the model of interest, e.g.,

```
 python predictor/run.py \
    train=False \
    tune=False
    test=False \
    predict=True \
    predictor/model=MTR \
    predictor/trainer=MTR \
    predictor/data=marginal_all \
    predictor.trainer.resume_ckpt_path=>insert_your_checkpoint< \
```