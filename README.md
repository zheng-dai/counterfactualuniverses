# Counterfactual Universes

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20965172.svg)](http://dx.doi.org/10.5281/zenodo.20965172)

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [Instructions for use](#instructions-for-use)
- [License](#license)

# Overview

This repo contains the custom code and data to run and analyze the results of our paper "[Outputs of Generative Diffusion Models are Often Unattributable](https://zheng-dai.github.io/AblationBasedCounterfactuals/)".

In this paper, we demonstrate that we can create a diffusion ensemble out of models that are trained on different data, which allows us to analyze what happens if training data is omitted by omitting models from the ensemble. We call the universe of variations that can be created via omission of single samples (or units) the counterfactual universe. The geometric structure of the counterfactual universe can then be used to quantify the attributability of the original factual sample. Our analysis shows that attributability decays as models are trained on increasing amounts of data.

This repo provides scripts to train diffusion ensembles, generate counterfactual universes, and interface with them via the CounterfactualLoader. Precomputed counterfactual universes are also included along with processed data used in the study.

# Repo Contents

- [src](./src): Contains code for training and running diffusion ensembles and analyzing counterfactual universes.
- [data](./data): Counterfactual radiuses and false attribution rates we measured, along with raw counterfactual universes we generated that are under 100MB. Please reach out to the corresponding authors concerning access to the larger raw counterfactual universes.
- [demo](./demo): A set of python scripts that run through a full training to counterfactual universe analysis pipeline on an MNIST dataset of 1000 images.
- [analysis](./analysis): A set of notebooks that demonstrate how data is structured and can be accessed.


# System Requirements

## Hardware Requirements

Training and running the diffusion ensemble requries an NVIDIA GPU with Compute Capability 7.5 (sm_75) and around 10GiB of memory. Viewing counterfactual universes does not require specialized hardware.

## Software Requirements

The code is developed and tested on Ubuntu 22.04 LTS. The code is written in Python 3. The conda environment in which the code was developed and tested is provided in `environment.yml`, which also contains the package dependencies.

# Installation Guide

Please consult the [official documentation](https://www.anaconda.com/docs/getting-started) on how to set up conda on your machine. Once conda is installed, simply navigate to the root of this repo, create, and activate the environment.

```
conda env create --file environment.yml
conda activate counterfactualuniverses
```

This will install the necessary packages, which are listed in `environment.yml`. The process should take a few minutes on most machines.


# Demo

The [demo directory](./demo) contains a set of python scripts that provides the full pipeline from training to analysis. The first two scripts should be run in order, while the last two can be run in any order after the first two have been run. The order of the scripts are:

  - [script_train_models.py](./demo/script_train_models.py): Trains a diffusion ensemble with 22 members on a subset of the MNIST dataset with 1000 images. This script also downloads the MNIST dataset in the directory where it is run. Trained models are saved in a subdirectory `demo`. This can take a few hours depending on hardware.
  - [script_generate_universe.py](./demo/script_generate_universe.py): Generates 10 images and their counterfactual universes. The output is saved to `demo.ctf`. This can take a few hours depending on hardware.
  - [script_print_results.py](./demo/script_print_results.py): Reports the counterfactual radius of the counterfactual universes using the Euclidean distance (without scaling the images to 256x256xRGB). This takes at most a few minutes.
  - [script_make_visual.py](./demo/script_make_visual.py): Creates a visual of the counterfactual universe at `out.png`. The top is the factual image, the counterfactual universe is below, and the corresponding removed samples are at the bottom. This takes at most a few minutes.

This sequence can be accomplished with the following set of commands starting in the root directory of the repo:

```
cd demo
python script_train_models.py
python script_generate_universe.py
python script_print_results.py
python script_make_visual.py
```

Note that the last script requires some interaction to select which counterfactual universe to visualize.


# Instructions for use


The demo scripts are high level wrappers around the [DiffusionEnsemble](./src/DiffusionEnsemble.py) and [CounterfactualLoader](./src/DiffusionEnsemble.py), and can be used as a base for constructing new analysis pipelines. Many of the datasets we use in this study can be conveniently loaded from [torchvision](https://docs.pytorch.org/vision/main/datasets.html), which is included in the conda environment. A more detailed demo on how to interface with the CounterfactualLoader is provided in the [CounterfactualLoader notebook](./analysis/CounterfactualLoader_demo.ipynb).

In addition, processed data such as measured Counterfactual Radiuses and the frequency of attribution changes are provided as `.csv` files under `data`. Their organization is illustrated in the [Processed Data Demo notebook](./analysis/processed_data_demo.ipynb) which reproduces the primary result of our study.


# License


This code release is covered under the [**Apache 2.0 License**](./LICENSE).
