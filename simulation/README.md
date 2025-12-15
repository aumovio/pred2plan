## Simulation of Prediction-Planner Pairs in NuPlan

### Starting a Simulation

You can start a simulation by selecting the simulator, planner, predictor with its respective checkpoint. E.g., for the wayformer-MPCC pair:

```
 python simulation/sim.py \
    simulator=nuplan_nonreactive \
    planner=mpcc \
    predictor/model=wayformer \
    predictor/data=marginal_all \
    predictor.trainer.resume_ckpt_path=>insert_your_checkpoint<
    simulator/scenario_filter=test14_hard
    simulator.scenario_filter.limit_total_scenarios=1
    planner.modular_planner.visualization=True
```

The simulations outputs open-loop metric results for the prediction model, closed-loop metric results for the planner and if visualization is set to True a video.

### Configurations

The global configuration script is [configs/simulation](../configs/simulation.yaml) file. For the general configuration setup logic refer to [configs/README.md](../configs/README.md). You can easily switch to reactive simulations on another scenario split with a different predictor-planner pair like this:

```
 python simulation/sim.py \
    simulator=nuplan_reactive \
    planner=bmpcc \
    predictor/model=MTR \
    predictor/data=marginal_all \
    predictor.trainer.resume_ckpt_path=>insert_your_checkpoint<
    simulator/scenario_filter=test14_random
```

For the Kinematic models the definition of a checkpoint is not necessary:

```
 python simulation/sim.py \
    simulator=nuplan_nonreactive \
    planner=mpcc \
    predictor/model=constvel \
    predictor/data=marginal_all \
    simulator/scenario_filter=test14_random
```

and respectively:

```
 python simulation/sim.py \
    simulator=nuplan_reactive \
    planner=mpcc \
    predictor/model=mapconstvel \
    predictor/data=joint_mapconstvel \
    simulator/scenario_filter=test14_random
```

