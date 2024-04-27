# Probabilistic Navigation Framework
This repository contains the dynamic Bayesian actor model for our manuscript: "Human navigation strategies, their errors and variability result from dynamic interactions of spatial uncertainties". This allows simulating triangle completion data for 5 experiments across 3 different studies. 

## System Requirements

**Development Environment:**
- **Language:** Python 3.7.13
- **Operating System:** macOS 12.2.1, optimized for M1 Pro chips.

**Large-scale Simulation Environment:**
- **Language:** Python 3.7.13
- **Operating System:** Redhat 8.8, Intel® Xeon® Platinum Prozessor 9242. 

**Important Considerations:**
The 'MA57' solver used to perform belief-space planning, requires additional `libhsl` and `libcoin` files, which can be licencesd from [HSL](http://www.hsl.rl.ac.uk). These libraries are crucial for the correct execution of the code; lacking them results in the model freezing during each iteration. Detailed instructions for installing CasADI + IPOPT + HSL are available [here](https://github.com/ami-iit/ami-commons/blob/master/doc/casadi-ipopt-hsl.md).

## Installation Guide
To begin, execute `setup-env.sh` to create a virtual environment and install all necessary dependencies. For a detailed list of required packages and software dependencies, please consult the `requirements.txt` file in this repository.

## Demo
The model can be explored through the interactive notebook contained in "notebooks/Belief Space MPC.ipynb". This notebook offers a hands-on experience with the homing task across the three studies. Experiment with different scenarios and tweak model parameters. A detailed list of parameters and their explanations are accessible in `parameters.py`, with specific values used in our simulations provided in the manuscript's supplementary files.

## Instruction Use

**Model Components:**
- `model.py`: Implements the models dynamic, observation functions and noise models.
- `belief_space_planner.py`: Focuses on optimization within belief space planning, including cost-function setup, constraints specification, and optimizer selection.
- `model_predictive_control.py`: Manages state initialization and serves as a wrapper for the main model functionalities.
- `mpc.py`: Executes the update steps of model predictive control for the triangle completion task.
- `state_estimation.py`: Implements the state-estimations update and prediction steps.
- `parameters.py`: Contains all parameters relevant to the model, facilitating easy adjustments and optimizations.

## Simulation Scripts
For large-scale simulations if navigation behavior in entire experiments, access to a compute cluster is recommended. The provided simulation scripts ensure efficient scheduling and parallel execution of individual trials via RayTune in combination with a SLURM scheduling system. These scripts are located in a separate folder within the repository.



