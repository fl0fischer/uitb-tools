# uitb-tools

This python package provides tools to analyze data from biomechanical movement simulations related to the [User-in-the-Box (uitb)](https://github.com/aikkala/user-in-the-box) framework.

The following tools are currently available:
- `uitb_evaluate`, a plotting and evaluation tool, including methods to plot end-effector trajectories, joint trajectories, and summary statistics such as movement duration (Fitts' Law), and methods to compute and visualize quantitative measurements such as RMSE between simulation data and user data *(still WIP)*.
- `uitb_reach_envelope`, a tool that enables interactive visualisation of the body-centred positions that a given simulated user can theoretically reach, along with the target positions that occur in a given VR interaction task. For details, see [SIM2VR: Towards Automated Biomechanical Testing in VR](todo:add-link-to-supp-material).
- `uitb_reward_scaling`, a tool that can be used to predict and visualise the effects of different reward function components, therefore informing the choice of appropriate reward/cost weights (e.g., effort cost weights, or the magnitude of a bonus term provided at early termination). For details, see [SIM2VR: Towards Automated Biomechanical Testing in VR](todo:add-link-to-supp-material).

### Installation:

```bash
pip install -e .
```
