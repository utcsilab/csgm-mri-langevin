# csgm-mri-langevin

NOTE: Please run **all** commands from the root directory of the repository, i.e from ```mri-langevin/```

## Setup environment
1. ```python -m venv env```
1. ```source env/bin/activate```
1. ```pip install -U pip```
1. ```pip install -r requirements.txt```
1. ```git submodule update --init --recursive```

## Install BART for sensitivity map estimation

BART provides tools for processing MRI data. Our experiments require BART for estimating sensitivity maps, and BART can be installed using the following commands.

1. ```sudo apt-get install make gcc libfftw3-dev liblapacke-dev libpng-dev libopenblas-dev```
1. ```wget https://github.com/mrirecon/bart/archive/v0.6.00.tar.gz```
1. ```tar xzvf v0.6.00.tar.gz```
1. ```cd bart-0.6.00```
1. ```make```

## Download data and checkpoints

1. ```gdown https://drive.google.com/uc?id=1vAIXf8n67yEAPmH2I9qiDWzmq9fGKPYL```
1. ```tar -zxvf checkpoint.tar.gz```
1. ```gdown https://drive.google.com/uc?id=1mpnV1iXid1PG0RaJswM6t9yI76b2IPxc```
1. ```tar -zxvf datasets.tar.gz```

## Script for estimating sensitivity maps from data

The script ```estimate_maps.py``` will estimate sensitivity maps. An example usage is

```python estimate_maps.py --input-dir=datasets/brain_T2 --output-dir=datasets/brain_T2_maps```

## Example commands
We provide configuration files in ```configs/``` that contain hyper-parameters used in our experiments. Here are example commands for using the configuration files.

1. T2-Brains:
```python main.py +file=brain_T2```
1. T1-Brains:
```python main.py +file=brain_T1```
1. FLAIR-Brains:
```python main.py +file=brain_FLAIR```
1. fastMRI Knees:
```python main.py +file=knees```
1. Abdomens:
```python main.py +file=abdomen```
1. Stanford knees:
```python main.py +file=stanford_knees```
1. To run with horizontal measurements:
```python main.py +file=brain_T2 orientation=horizontal```
1. To run with random measurements:
```python main.py +file=brain_T2 pattern=random```
1. To change acceleration factor:
```python main.py +file=brain_T2 R=8```

## Plotting results
We use CometML to save results. Please see ```plot-demo.ipynb``` for example reconstructions.

## Citations

If you find this repository useful, please consider citing the following papers:
```
@article{jalal2021robust,
  title={Robust Compressed Sensing MRI with Deep Generative Priors},
  author={Jalal, Ajil and Arvinte, Marius and Daras, Giannis and Price, Eric and Dimakis, Alexandros G and Tamir, Jonathan I},
  journal={Advances in Neural Information Processing Systems},
  year={2021}
}

@article{jalal2021instance,
  title={Instance-Optimal Compressed Sensing via Posterior Sampling},
  author={Jalal, Ajil and Karmalkar, Sushrut and Dimakis, Alexandros G and Price, Eric},
  journal={International Conference on Machine Learning},
  year={2021}
}

```

Our code uses prior work from the following papers, which must
be cited:
```
@inproceedings{song2019generative,
  title={Generative modeling by estimating gradients of the data distribution},
  author={Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  pages={11918--11930},
  year={2019}
}

@article{song2020improved,
  title={Improved Techniques for Training Score-Based Generative Models},
  author={Song, Yang and Ermon, Stefano},
  journal={arXiv preprint arXiv:2006.09011},
  year={2020}
}
```

We use data from the NYU fastMRI dataset, which must also be cited:
```
@inproceedings{zbontar2018fastMRI,
    title={{fastMRI}: An Open Dataset and Benchmarks for Accelerated {MRI}},
    author={Jure Zbontar and Florian Knoll and Anuroop Sriram and Tullie Murrell and Zhengnan Huang and Matthew J. Muckley and Aaron Defazio and Ruben Stern and Patricia Johnson and Mary Bruno and Marc Parente and Krzysztof J. Geras and Joe Katsnelson and Hersh Chandarana and Zizhao Zhang and Michal Drozdzal and Adriana Romero and Michael Rabbat and Pascal Vincent and Nafissa Yakubova and James Pinkerton and Duo Wang and Erich Owens and C. Lawrence Zitnick and Michael P. Recht and Daniel K. Sodickson and Yvonne W. Lui},
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1811.08839},
    year={2018}
}

@article{knoll2020fastmri,
  title={fastMRI: A publicly available raw k-space and DICOM dataset of knee images for accelerated MR image reconstruction using machine learning},
  author={Knoll, Florian and Zbontar, Jure and Sriram, Anuroop and Muckley, Matthew J and Bruno, Mary and Defazio, Aaron and Parente, Marc and Geras, Krzysztof J and Katsnelson, Joe and Chandarana, Hersh and others},
  journal={Radiology: Artificial Intelligence},
  volume={2},
  number={1},
  pages={e190007},
  year={2020},
  publisher={Radiological Society of North America}
}
```
