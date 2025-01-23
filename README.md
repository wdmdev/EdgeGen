# EdgeGen
Generation of Neural Network Architectures for Resource Constrained Hardware guided by constraints e.g. on flash storage and memory usage.
The project is initial work and exploration. Multiple improvements can be done and bugs still exist.

**N.B.** The main design space of the project is the [Younger Dataset](https://github.com/Yangs-AI/Younger) though the code has been design for expansion to other design spaces as well. 
The project also implements a BO search over the OFAProxylessNAS search space from [mcunet](https://github.com/mit-han-lab/mcunet).
However, this is not maintained as it was only done to test the extendability of the code base to other design spaces than the Younger dataset.

## Project Dependency Management 
This project uses the [uv](https://docs.astral.sh/uv/) Python package and project manager.

## Setup

**Step 1**

Run the folowing command to set up the `uv` python `.venv` and install all dependencies.
```sh
uv init && uv sync
```
**Step 2**

Download the `Filter Series With Attributes` from `Yangs Cloud` which hosts the filtered Younger Dataset [here](https://datasets.yangs.cloud/public/assets/4dd11295-28c6-4d5e-a6bb-0ce86bfc7c57?download=)

**Step 3**

Unzip the file and rename it to `networks`

**Step 4**
Place the `networks` folder in `src/edgegen/design_space/architectures/younger`

Now all should be setup.

## Running the Code
To generate architectures use
```sh
uv run src/edgegen/younger_generate_random_walk.py --walk_length <walk_len> --num_walks <num_walks> (optional) --valid_ops <valid_ops> (optional) --valid_input_ops <input_ops> (optional) --valid_output_ops <output_ops>
```
The default valid operations can be found in `src/edgegen/design_space/architectures/younger/config`

At the moment constraints cannot be specified through the command line. Instead, modify `younger_generate_random_random_walk.py` directly.

To implement new constraints inherit from the `Constraint` class in `src/edgegen/evaluation/constraints/constraint.py`.


## Finally

Good luck, have fun!