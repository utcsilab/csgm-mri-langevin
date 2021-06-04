# mri-langevin

NOTE: Please run **all** commands from the root directory of the repository, i.e from ```mri-langevin/```

## Setup environment
---
1. ```python -m venv env```
1. ```source env/bin/activate```
1. ```pip install -U pip```
1. ```pip install -r requirements.txt```
1. ```git submodule update --init --recursive```
1. ```bash setup.sh```

## Instructions for running individual experiments
1. T2-Brains:
```python run_langevin.py batch_size=8```
1. T1-Brains:
```python run_langevin.py +configs=run_langevin_T1 batch_size=14```
1. FLAIR-Brains:
```python run_langevin.py +configs=run_langevin_FLAIR batch_size=16```
1. fastMRI-Knees:
```python run_langevin.py +configs=run_langevin_knees batch_size=16```
1. abdomens:
```python run_langevin.py +configs=run_langevin_abdomens batch_size=34```
1. Stanford knees:
```python run_langevin.py +configs=run_langevin_stanford_knees batch_size=24```

NOTE: all experiments and reconstructions can be viewed at this [link](https://www.comet.ml/anonymous-bobo-neurips21#projects)

To run different sampling patterns and acceleration, do
```python run_langevin.py pattern=random orientation=horizontal R=8```
