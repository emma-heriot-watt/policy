<div align='center'>

# EMMA SimBot: Policy

<a href="https://www.python.org/">
  <img alt="Python 3.9" src="https://img.shields.io/badge/-Python 3.9+-blue?logo=python&logoColor=white">
</a>
<a href="https://pytorch.org/">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white">
</a>
<a href="https://pytorchlightning.ai/">
  <img alt="Lightning"
  src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white">
</a>
<a href="https://python-poetry.org">
  <img alt="Poetry" src="https://img.shields.io/badge/Poetry-1E293B?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB2aWV3Qm94PSIwIDAgNDQ4LjE3IDU2MCI+PGRlZnM+PHN0eWxlPi5jbHMtMXtpc29sYXRpb246aXNvbGF0ZTt9LmNscy0ye2ZpbGw6dXJsKCNyYWRpYWwtZ3JhZGllbnQpO30uY2xzLTN7ZmlsbDp1cmwoI3JhZGlhbC1ncmFkaWVudC0yKTt9LmNscy00LC5jbHMtNSwuY2xzLTZ7bWl4LWJsZW5kLW1vZGU6bXVsdGlwbHk7fS5jbHMtNHtmaWxsOnVybCgjbGluZWFyLWdyYWRpZW50KTt9LmNscy01e2ZpbGw6dXJsKCNsaW5lYXItZ3JhZGllbnQtMik7fS5jbHMtNntmaWxsOnVybCgjbGluZWFyLWdyYWRpZW50LTMpO30uY2xzLTd7bWl4LWJsZW5kLW1vZGU6c2NyZWVuO2ZpbGw6dXJsKCNyYWRpYWwtZ3JhZGllbnQtMyk7fTwvc3R5bGU+PHJhZGlhbEdyYWRpZW50IGlkPSJyYWRpYWwtZ3JhZGllbnQiIGN4PSI0MzguMyIgY3k9IjYzOS4wMSIgcj0iNTY5Ljk0IiBncmFkaWVudFRyYW5zZm9ybT0idHJhbnNsYXRlKDAgMCkiIGdyYWRpZW50VW5pdHM9InVzZXJTcGFjZU9uVXNlIj48c3RvcCBvZmZzZXQ9IjAiIHN0b3AtY29sb3I9IiM2ODc3ZWMiLz48c3RvcCBvZmZzZXQ9IjAuNiIgc3RvcC1jb2xvcj0iIzUzNjJjZiIvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzQzNTJiOSIvPjwvcmFkaWFsR3JhZGllbnQ+PHJhZGlhbEdyYWRpZW50IGlkPSJyYWRpYWwtZ3JhZGllbnQtMiIgY3g9IjY1LjY0IiBjeT0iLTE2LjIxIiByPSI3NDYuNDYiIGdyYWRpZW50VHJhbnNmb3JtPSJ0cmFuc2xhdGUoMCAwKSIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiPjxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iIzAwZDVmZiIvPjxzdG9wIG9mZnNldD0iMC4zOCIgc3RvcC1jb2xvcj0iIzAwYjhlYiIvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzAwODBjNSIvPjwvcmFkaWFsR3JhZGllbnQ+PGxpbmVhckdyYWRpZW50IGlkPSJsaW5lYXItZ3JhZGllbnQiIHgxPSI3NC43NyIgeTE9IjY3LjMiIHgyPSIyNzcuMjMiIHkyPSI1MTIuNzIiIGdyYWRpZW50VW5pdHM9InVzZXJTcGFjZU9uVXNlIj48c3RvcCBvZmZzZXQ9IjAiIHN0b3AtY29sb3I9IiMyOTRjYTciLz48c3RvcCBvZmZzZXQ9IjAuNDgiIHN0b3AtY29sb3I9IiM5NmE3ZDQiLz48c3RvcCBvZmZzZXQ9IjAuODQiIHN0b3AtY29sb3I9IiNlMWU2ZjMiLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiNmZmYiLz48L2xpbmVhckdyYWRpZW50PjxsaW5lYXJHcmFkaWVudCBpZD0ibGluZWFyLWdyYWRpZW50LTIiIHgxPSItMjI4Ljc0IiB5MT0iLTE0NC4yOSIgeDI9IjQ1MSIgeTI9IjY1MS44OSIgZ3JhZGllbnRUcmFuc2Zvcm09InRyYW5zbGF0ZSgwIDApIiBncmFkaWVudFVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+PHN0b3Agb2Zmc2V0PSIwIiBzdG9wLWNvbG9yPSIjNjg3N2VjIi8+PHN0b3Agb2Zmc2V0PSIwLjI5IiBzdG9wLWNvbG9yPSIjOTdhMWYyIi8+PHN0b3Agb2Zmc2V0PSIwLjc3IiBzdG9wLWNvbG9yPSIjZTJlNGZiIi8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjZmZmIi8+PC9saW5lYXJHcmFkaWVudD48bGluZWFyR3JhZGllbnQgaWQ9ImxpbmVhci1ncmFkaWVudC0zIiB4MT0iLTE1MS4yMiIgeTE9Ii0yODUuOSIgeDI9IjQ1MC4wOCIgeTI9IjQzMC42MyIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiPjxzdG9wIG9mZnNldD0iMCIgc3RvcC1jb2xvcj0iIzgzOTdjYyIvPjxzdG9wIG9mZnNldD0iMC4xNSIgc3RvcC1jb2xvcj0iIzk3YThkNCIvPjxzdG9wIG9mZnNldD0iMC43MyIgc3RvcC1jb2xvcj0iI2UyZTZmMyIvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iI2ZmZiIvPjwvbGluZWFyR3JhZGllbnQ+PHJhZGlhbEdyYWRpZW50IGlkPSJyYWRpYWwtZ3JhZGllbnQtMyIgY3g9IjI1OS42OCIgY3k9Ii0zNC43MSIgcj0iNDMxLjM3IiBncmFkaWVudFVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+PHN0b3Agb2Zmc2V0PSIwIiBzdG9wLWNvbG9yPSIjZmZmIi8+PHN0b3Agb2Zmc2V0PSIxIi8+PC9yYWRpYWxHcmFkaWVudD48L2RlZnM+PHRpdGxlPmxvZ28tb3JpZ2FtaTwvdGl0bGU+PGcgY2xhc3M9ImNscy0xIj48ZyBpZD0iTGF5ZXJfMSIgZGF0YS1uYW1lPSJMYXllciAxIj48cGF0aCBjbGFzcz0iY2xzLTIiIGQ9Ik0xNjguMDgsNTYwQTU3MC41NCw1NzAuNTQsMCwwLDAsNDU5Ljg0LDQwMy41OUw1Ni4yNSwwVjQ0OC4xN1oiIHRyYW5zZm9ybT0idHJhbnNsYXRlKC01Ni4yNSkiLz48cGF0aCBjbGFzcz0iY2xzLTMiIGQ9Ik01Ni4yNSw0NDguMTdDMzAzLjc3LDQ0OC4xNyw1MDQuNDIsMjQ3LjUyLDUwNC40MiwwSDU2LjI1WiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTU2LjI1KSIvPjxwYXRoIGNsYXNzPSJjbHMtNCIgZD0iTTU2LjI1LDQ0OC4xN2gwTDczLjUsNDY1LjQyYzEyMS41Ny00LjQ1LDIzMS40LTU1LjY4LDMxMi0xMzYuMjNsLTEyLjI5LTEyLjI4QTQ0Ni44LDQ0Ni44LDAsMCwxLDU2LjI1LDQ0OC4xN1oiIHRyYW5zZm9ybT0idHJhbnNsYXRlKC01Ni4yNSkiLz48cGF0aCBjbGFzcz0iY2xzLTUiIGQ9Ik0xNjguMDgsNTYwQTU3MC41NCw1NzAuNTQsMCwwLDAsNDU5Ljg0LDQwMy41OUw1Ni4yNSwwVjQ0OC4xN1oiIHRyYW5zZm9ybT0idHJhbnNsYXRlKC01Ni4yNSkiLz48cGF0aCBjbGFzcz0iY2xzLTYiIGQ9Ik00NTkuODQsNDAzLjU5LDU2LjI1LDAsNDIzLjE0LDQzNy4xM0M0MzUuODMsNDI2LjQ2LDQ0OC4xMiw0MTUuMzEsNDU5Ljg0LDQwMy41OVoiIHRyYW5zZm9ybT0idHJhbnNsYXRlKC01Ni4yNSkiLz48cGF0aCBjbGFzcz0iY2xzLTciIGQ9Ik01Ni4yNSwwLDM3My4xNiwzMTYuOTFxNC4yMy00LjI1LDguMzUtOC42WiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTU2LjI1KSIvPjwvZz48L2c+PC9zdmc+">
</a>
<a href="https://hydra.cc/">
  <img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd">
</a>

  <br>

<a href="https://github.com/pre-commit/pre-commit">
  <img alt="pre-commit" src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white">
</a>
<a href="https://github.com/psf/black">
  <img alt="style: black" src="https://img.shields.io/badge/style-black-000000.svg">
</a>
<a href="https://wemake-python-stylegui.de/en/">
  <img alt="wemake-python-stylegude" src="https://img.shields.io/badge/style-wemake-000000.svg">
</a>

<br>

[![Continuous Integration](https://github.com/emma-simbot/policy/actions/workflows/continuous-integration.yml/badge.svg)](https://github.com/emma-simbot/policy/actions/workflows/continuous-integration.yml)
[![Tests](https://github.com/emma-simbot/policy/actions/workflows/tests.yml/badge.svg)](https://github.com/emma-simbot/policy/actions/workflows/tests.yml)

  </div>

---

## Quick start

Assuming you have [pyenv](https://github.com/pyenv/pyenv) and [Poetry](https://python-poetry.org/), clone the repository and run:

```bash
# Use Python 3.9.13 in the project
pyenv local 3.9.13

# Tell Poetry to use pyenv
poetry env use $(pyenv which python)

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell

# Install pre-commit hooks
pre-commit install
```

Check out the [CONTRIBUTING.md](https://github.com/emma-simbot/policy/blob/main/CONTRIBUTING.md) for more detailed information on getting started.

### Installing optional dependencies

We've separated specific groups of dependencies so that you only need to install what you need.

- For RL, run `poetry install -E rl`

## Writing code and running things

### Project structure

This is organised in very similarly to structure from the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template#project-structure) to facilitate reproducible research code.

- `scripts` — `sh` scripts to run experiments
- `configs` — configurations files using the [Hydra framework](https://hydra.cc/)
- `docker` — Dockerfiles to ease deployment
- `notebooks` — Jupyter notebook for analysis and exploration
- `storage` — data for training/inference _(and maybe use symlinks to point to other parts of the filesystem)_
- `tests` — [pytest](https://docs.pytest.org/en/) scripts to verify the code
- `src` — where the main code lives

### Running things

Train model with default configuration

```bash
# train on CPU
python run.py trainer.gpus=0

# train on 1 GPU
python run.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python run.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python run.py trainer.max_epochs=20 datamodule.train_batch_size=64
```

Especially when you're trying to specify extra parameters for the class `Trainer` from
[PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.html#module-pytorch_lightning.trainer.trainer),
you might run into trouble when running a command like the following:

```bash
python run.py trainer.precision=16 datamodule.train_batch_size=64
```

This is because Hydra allows you to modify only parameters specified in the configuration file. So
if you don't have `precision` among them, Hydra will complain. If you're sure that the parameters
is allowed, just change the previous command as follows:

```bash
python run.py +trainer.precision=16 datamodule.train_batch_size=64
```

In this way, Hydra will automatically _append_ the new parameter to the configuration dictionary of
the `Trainer` we're trying to instantiate.

<details>
<summary>It's annoying, why do I have to do that? </summary>

<br>

We're working on a possible fix and we're exploring different options. If you're interested in
this, please follow this [issue](https://github.com/emma-simbot/research-base/issues/26).

</details>
