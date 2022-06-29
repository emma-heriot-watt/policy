# Submitting a job to Cirrus GPU Nodes

The following examples are specific to Cirrus and project ec202.

The official documentation for using the Cirrus GPU nodes can be found [here](https://cirrus.readthedocs.io/en/main/user-guide/gpu.html).

You can find an example submission script in `scripts/slurm/job_multi_node.slurm`.
You need to do at least the following modifications:

1. Modify the number of nodes (each node has 4 GPUs).
2. Activate your environment. Cirrus suggests creating a custom miniconda environment. For further instructions, see [here](https://cirrus.readthedocs.io/en/main/user-guide/python.html?highlight=pytorch#custom-miniconda3-environments).
3. Add you personal WANDB_API_KEY which you can find in the settings of your wandb profile.
