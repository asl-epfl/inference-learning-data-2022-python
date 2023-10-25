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

### 1. Download the repository

There are two ways for downloading the content of this repository. The first way is directly by the **GitHub**. Just click in the **<> Code** green button in the top of this page and then click in **Download zip** buttom. With this, you can unzip the downloaded folder in your desired installation path and access all the content of this repository.

The other way is directly with the command line. For this, you will need to have **Git** installed in your machine, for this you can check <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>. After that, you can run this command line in your terminal:

`git clone https://github.com/asl-epfl/E3-python-code-2023.git`

With this command all the content of this repository will be downloaded in your machine.

To access the folder in your terminal you can run:

`cd installation_path/E3-python-code-2023`

### 2. Install Python

First, you need to install python in your machine. To check if you already have python installed just run this command in your terminal:

`python --version`

If you get a message like this:

`Python 3.11.3` (the numbers indicate the python version that is installed in your machine).

If you get an error message, it means that you don't have python installed in your machine. In this case, download and install python from <https://www.python.org/downloads/>.

### 3. Install Python Packages

For the code in this repository to run, we will use some additional libraries such as: numpy, matplotlib, pickle, pytorch, etc. They are already listed in the file 'requirements.txt'. To instal them, run this command in your terminal:

`pip install -r requirements.txt`

This command line will install all packages listed in the file `requirements.txt`. In this file, we specify the version of each package too.

### 4. Jupyter Lab

After installing all the packages, we can run the codes using Jupyter Notebooks. For this, run this command in your terminal:

`jupyter lab`

After doing this, Jupyter Lab will open in your browser. 

### 5. Running Python Notebooks

After doing all these steps you will be able to run all the python notebooks from all the chapters. 