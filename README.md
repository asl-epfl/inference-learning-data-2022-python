# Inference and Learning from Data: Python Code

This repository should contain the code and generated figures relative to the simulation examples presented in the following books:

- *A. H. Sayed, Inference and Learning from Data, Vol. 1. Cambridge University Press, 2022.*
- *A. H. Sayed, Inference and Learning from Data, Vol. 2. Cambridge University Press, 2022.*
- *A. H. Sayed, Inference and Learning from Data, Vol. 3. Cambridge University Press, 2022.*

The folder structure for each chapter is similar to:

``` bash

chapter-03
│   chapter-03.ipynb
│   functions.py
│
└───data
│   │   data-1.mat
│   │   data-2.mat
│   │   ...
│   
└───figs
    │   fig-1.pdf
    │   fig-2.pdf
    │   ...

```

Where each chapter contains the main code in the form of a notebook, auxiliary function in the form of .py files, and dedicated folders `data` and `figs`, which collect the necessary data to run the code and the generated figures respectively.

This content should by no means be shared or published without the authorization of Prof. Ali H. Sayed.

## Tutorial to run the python notebooks

### 1. Install Python

First of all, we need to install python in our machine. To check if you already have python installed just run this command in your terminal:

`python --version`

If you get a message like this:

`Python 3.11.3` (these numbers are just the python version that is installed in your machine).

If you get an error message, it means that you don't have python installed in your machine. For installing it, just get it from https://www.python.org/downloads/.

### 2. Install Python Packages

In Python, we inevitably will use tons of packages as: numpy, matplotlib, pickle, pytorch, etc. We already listed all the packages that will be needed to run all the chapters notebooks. For installing them just run this command in your terminal:

`pip install -r requirements.txt`

This command line will install all packages listed in the file `requirements.txt`. In this file contains the version of each package too.

### 3. Jupyter Lab

After installing all the packages, we can run the codes using Jupyter Notebooks. For this, run this command in your terminal:

`jupyter lab`

After doing this, Jupyter Lab will open in your browser. 

### 4. Running Python Notebooks

After doing all these steps you will be able to run all the python notebooks from all the chapters. 