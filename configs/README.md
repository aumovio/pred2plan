## Configuration Setup

The configuration attempts to split respective configurations for the planner, predictor and simulator. 

```
configs/
├── hydra
├── override
├── paths
├── planner # contains the respective planner configs, including mpcc, bmpcc and rbmpcc
├── predictor # contains predictor sub configs
│   ├── callbacks 
│   ├── data.yaml # configuration for the dataloader, including datasplits, compression, cache name
│   ├── metrics.yaml # configuration for metrics during train, val, test, predict steps
│   ├── model.yaml # configuration for respective models, including #layers and dimensions
│   ├── trainer.yaml # configuration for lightning trainer object, including batch sizes and epochs
│   └── tuner.yaml # configuration for tuner
├── simulator
│   ├── nuplan # standard nuplan configuration, including callbacks and filters
│   └── scenario_filter # configuration for scenario splits
```
