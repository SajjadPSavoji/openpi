# openpi

## Requirements

## Installation

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## Fine-Tuning Base Models on Your Own Data

We will fine-tune the $\pi_0$ model on the data generated from maniskill. as a running example for how to fine-tune a base model on your own data. We will explain three steps:
1. Convert your data to a LeRobot dataset (Done in Maniskill Repo)
2. Defining training configs and running training (Has been Done)
3. Spinning up a policy server and running inference

### 1. Convert your data to a LeRobot dataset

Read Maniskill Readme for this step.

### 2. Defining training configs and running training

To fine-tune a base model on your own data, you need to define configs for data processing and training. We provide example configs with detailed comments for Noahbiarm in the config files.

Before we can run training, we need to compute the normalization statistics for the training data. Run the script below with the name of your training config:

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_noahbiarm
```

Now we can kick off training with the following command (the `--overwrite` flag is used to overwrite existing checkpoints if you rerun fine-tuning with the same config):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_noahbiarm --exp-name=my_experiment --overwrite
```

The command will log training progress to the console and save checkpoints to the `checkpoints` directory. 


### 3. Spinning up a policy server and running inference

Once training is complete, we can run inference by spinning up a policy server and then querying it from a Libero evaluation script. Launching a model server is easy (we use the checkpoint for iteration 20,000 for this example, modify as needed):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_noahbiarm --policy.dir=checkpoints/pi0_noahbiarm/my_experiment/30000
```

This will spin up a server that listens on port 8001 and waits for observations to be sent to it. We can then run the evaluation script on maniskill to query the server.