# linear-regressor-medical-insurance

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Implementation of linear regression and configurable logistic regression using batch and stochastic gradient descent. Dataset: [Kaggle Insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)

--------

## Description

This project started as a personal experiment to understand gradient descent and linear models from scratch. I wanted a cloud backup for my code, and figured it could be helpful or interesting to others learning how linear and logistic regression work. Includes an example dataset (Kaggle Insurance) to try out predictions.


_Features_
- Implements linear regression with batch and stochastic gradient descent.
- Can be configured for logistic regression for binary classification tasks.
- Simple Python implementation with no external ML libraries required.


_Why it’s useful_
- Educational: demonstrates gradient descent updates step by step.
- Lightweight and extendable: you can apply it to other datasets or regression problems easily.

_How to run it yourself_
- Project scaffolding was created using cookiecutter-data-science.
- This project uses Poetry to manage dependencies and virtual environments.

```
# Install dependencies and create virtual environment
poetry install

# Activate the virtual environment
poetry shell

# Run training
python lin_mod/modeling/train.py
```

--------

## Project Organization (from ccds)

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         lin-mod and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── lin-mod   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes lin-mod a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```
