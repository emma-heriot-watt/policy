# Contributing

- [Quick start](#quick-start)
- [Installing things](#installing-things)
  - [Poetry](#poetry)
  - [Managing Python versions...](#managing-python-versions)
    - [...with pyenv](#with-pyenv)
    - [...with Anaconda](#with-anaconda)
- [Developer tooling](#developer-tooling)
  - [Code style](#code-style)
    - [Task runner](#task-runner)
    - [Formatting](#formatting)
    - [Linting](#linting)
    - [Type checking](#type-checking)
    - [Working with branches](#working-with-branches)

## Quick start

Assuming you have [pyenv](https://github.com/pyenv/pyenv) and [Poetry](https://python-poetry.org/), clone the repository and run:

```bash
# Use Python 3.9.9 in the project
pyenv local 3.9.9

# Tell Poetry to use pyenv
poetry env use $(pyenv which python)

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell

# Install pre-commit hooks
pre-commit install
```

## Installing things

We've tried to keep necessary things as simplistic as possible. However, we need to install some things.

### Poetry

This project uses Poetry for **creating virtual environments** and **managing Python packages**. This should be installed globally and can be done by running:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

You can verify it's installed and accessible by running `poetry --version`.

Once you've got Poetry installed, we think it's best to install Python dependencies into a `.venv/` folder within the cloned repo. Tell Poetry to handle this for you:

```bash
poetry config virtualenvs.in-project true
```

For more on how to manage, add, remove, and update dependencies, see the [official Poetry documentation](https://python-poetry.org/docs/basic-usage/).

### Managing Python versions...

There are two ways of managing your Python environments. We recommend [pyenv](https://github.com/pyenv/pyenv), but we have also included instructions for [Anaconda](https://anaconda.com).

#### ...with pyenv

Install pyenv following the [instructions within the official repo](https://github.com/pyenv/pyenv#installation) for your system. **Remember to do step 2**!

You can verify it's installed with `pyenv --version`.

1. Install the Python version you want with `pyenv install 3.9.9`
2. Go to the cloned repo
3. Assign the specific Python version to the project by running `pyenv local 3.9.9`

If you want a different version of Python, just change the version in the steps.

#### ...with Anaconda

Install Anaconda using the [instructions on the official website](https://anaconda.com/).

Then create an environment for your project by running:

```bash
conda create -n PROJECT_NAME python=3.9
conda activate PROJECT_NAME
```

## Developer tooling

- Dependency management with [Poetry](https://python-poetry.org/)
- Easier task running with [Poe the Poet](https://github.com/nat-n/poethepoet)
- Code formatting with [Black](https://github.com/psf/black) and [Prettier](https://prettier.io/)
- Linting with [pre-commit](https://pre-commit.com/) and [Flake8](http://flake8.pycqa.org/), using the strict [wemake-python-styleguide](https://wemake-python-stylegui.de/en/latest/)
- Automated Python Docstring Formatting with [docformatter](https://github.com/myint/docformatter)
- Continuous integration with [GitHub Actions](https://github.com/features/actions)
- Testing with [pytest](https://docs.pytest.org/en/latest/)
- Code coverage with [coverage.oy](https://coverage.readthedocs.io/)
- Static type-checking with [mypy](http://mypy-lang.org/)
- Automated Python syntax updates with [pyupgrade](https://github.com/asottile/pyupgrade)
- Security audit with [Bandit](https://github.com/PyCQA/bandit)
- Automated release notes with [Release Drafter](https://github.com/release-drafter/release-drafter)
- Manage project labels with [GitHub Labeler](https://github.com/marketplace/actions/github-labeler)
- Automated dependency updates with [Dependabot](https://dependabot.com/)

### Code style

To ensure all code is standardized, we use [black](https://github.com/psf/black), along with other automatic formatters. To enforce a consistent coding style, we use [Flake8](https://flake8.pycqa.org/en/latest/) with the [wemake-python-styleguide](https://wemake-python-stylegui.de/en/latest/). To verify and enforce type annotations, we use [mypy](https://mypy.readthedocs.io/en/stable/). Common tasks can be called using [Poe the Poet](https://github.com/nat-n/poethepoet).

#### Task runner

[Poe the Poet](https://github.com/nat-n/poethepoet) is a task runner that works well with Poetry. To see what tasks exist, run `poe` in your terminal. If you want to add more tasks, check [the documentation](https://github.com/nat-n/poethepoet) and [what already exists](https://github.com/emma-simbot/research-base/blob/main/pyproject.toml).

If you have issues getting `poe` to work, make sure that you are already within the activated shell (by running `poetry shell`).

#### Formatting

If you want to automatically format on every commit, you can use [pre-commit](https://pre-commit.com/). As mentioned above, run `pre-commit install` and it will install the hooks.

To manually format all files, run

```bash
poe format
```

#### Linting

If you want to check the linting rules on the codebase, run

```bash
poe lint
```

#### Type checking

If you want to check types on the codebase, run

```bash
poe typecheck
```

#### Working with branches

We've settled on a middle ground when it comes to developing: **keep the `main` branch clean**.

Within branches, you can do whatever you want to do, but you should **never push anything directly to the `main` branch**.

For every PR, an automated set of linters and formatters will check your code to
see whether it follows the set rules. If it fails, **do not merge** with the
`main` branch.
