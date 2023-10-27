# Inference and Learning from Data: Python Code

This repository contains code relative to simulations and figures from the following reference:

- A. H. Sayed, *Inference and Learning from Data*, Vol. 1. Cambridge University Press, 2022.
- A. H. Sayed, *Inference and Learning from Data*, Vol. 2. Cambridge University Press, 2022.
- A. H. Sayed, *Inference and Learning from Data*, Vol. 3. Cambridge University Press, 2022.

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

where each chapter contains the main code in the form of a notebook, auxiliary functions in the form of .py files, and dedicated folders `data` and `figs`, which collect the necessary data to run the code and the generated figures respectively.

## Tutorial to run the python notebooks

### 1. Download the repository

There are two ways to download the content of this repository. The first one is directly by  **GitHub**. Just click in the **<> Code** green button in the top of this page and then click in **Download zip** buttom. You can unzip the downloaded folder in your desired installation path and access all the content of this repository.

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

## Disclaimer 
This statement is applicable to all code/software provided with the text listed above, including the code used to prepare data and generate figures and numerical results for all chapters.

``The computer codes are provided "as is" without any guarantees. Practitioners should use them at their own risk. While the codes in the text are useful for instructional purposes, they are not intended to serve as examples of full-blown or optimized designs. The author has made no attempt at optimizing the codes, perfecting them, or even checking them for absolute accuracy. In order to keep the codes at a level that is easy to follow by students, the author has often chosen to sacrifice performance or even programming elegance in lieu of simplicity. Students can use the computer codes to run variations of the examples shown in the text.``

MIT License

Copyright (c) 2022. A. H. Sayed, Inference and Learning from Data, Cambridge University Press, vols. I, II, and III, 2022.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Simulation Datasets
In several examples in the work by A. H. Sayed, Inference and Learning from Data, Cambridge University Press, 2022, the simulations rely on publicly available datasets. The sources for these datasets are acknowledged in the appropriate locations in the text. Here we provide an aggregate summary for ease of reference:
1. Iris dataset. This classical dataset contains information about the sepal length and width for three types of iris flowers: Virginica, Setosa, and Versicolor. It was originally used by Fisher (1936) and is available at the UCI Machine Learning Repository at https://archive.ics.uci.edu/ml/datasets/iris Actually, three of the datasets in our list are downloaded from this useful repository - see Dua and Graff (2019).
2. MNIST dataset. This is a second popular dataset, which is useful for classifying handwritten digits. It was used in the work by LeCun et al. (1998) on document recognition. It contains 60,000 labeled training examples and 10,000 labeled test examples for the digits 0 through 9. It can be downloaded from http://yann.lecun.com/exdb/mnist/.
3. CIFAR-10 dataset. The dataset consists of color images that can belong to one of 10 classes: airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks. It is described by Krizhevsky (2009) and can be downloaded from the website http://www.cs.toronto.edu/~kriz/cifar.html
4. FBI crime dataset. The dataset contains statistics showing the burglary rates per 100,000 inhabitants for the period 1997-2016. The source of the data is the US Criminal Justice Information Services Division at the link https://ucr.fbi.gov/crime-in-the-u.s/2016/crime-in-the-u.s.- 2016/tables/table-1.
5. Sea level and global temperature changes dataset. The sea level dataset measures the change in sea level relative to the start of year 1993. There are 952 data points consisting of fractional year values. The source of the data is the NASA Goddard Space Flight Center at https://climate.nasa.gov/vital-signs/sea-level/ For information on how the data was generated, the reader may consult Beckley et al. (2017) and the report GSFC (2017). The temperature dataset measures changes in the global surface temperature relative to the average over the period 1951- 1980. There are 139 measurements between the years 1880 and 2018. The source of the data is the NASA Goddard Institute for Space Studies (GISS) at https://climate.nasa.gov/vital-signs/global- temperature/.
6. Breast cancer Wisconsin dataset. This dataset consists of 569 samples, with each sample corresponding to a benign or malignant cancer classification. For information on how the data was generated, the reader may consult Mangasarian, Street, and Wolberg (1995). The data can be downloaded from the UCI Machine Learning Repository at the link https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
7. Heart-disease Cleveland dataset. The dataset consists of 297 samples that belong to patients with and without heart disease. It is available on the UCI Machine Learning Repository at https://archive.ics.uci.edu/ml/datasets/heart+Disease The investigators responsible for the collection of the data are the leading 4 co-authors of the article by Detrano et al. (1989).
          
### References
1. B. D. Beckley, P. S. Callahan, D. W. Hancock, G. T. Mitchum, and R. D. Ray (2017), “On the cal-mode correction to TOPEX satellite altimetry and its effect on the global mean sea level time series,” J. Geophysical Research: Oceans, vol. 122, no. 11, pp. 8371-8384.
2. R. Detrano, A. Janosi, W. Steinbrunn, M. Pfisterer, J. Schmid, S. Sandhu, K. Guppy, S. Lee, and V. Froelicher (1989), “International application of a new probability algorithm for the diagnosis of coronary artery disease,” American J. Cardiology, vol. 64, pp. 304- 310.
3. D. Dua and C. Graff (2019), UCI Machine Learning Repository, http://archive.ics.uci.edu/ml, School of Information and Computer Science, University of California, Irvine.
4. R. A. Fisher (1936), “The use of multiple measurements in taxonomic problems,” Annals of Eugenics, vol. 7, no. 2, pp. 179-188.
5. GSFC (2017), “Global mean sea level trend from integrated multi-mission ocean altimeters TOPEX/Poseidon, Jason-1, OSTM/Jason-2,” ver. 4.2PO.DAAC, CA, USA. Dataset accessed 2019-03-18 at http://dx.doi.org/10.5067/GMSLM-TJ42.
6. A. Krizhevsky, (2009), Learning Multiple Layers of Features from Tiny Images, MS dissertation, Computer Science Department, University of Toronto, Canada.
7. Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner (1998), “Gradient-based learning applied to document recognition,” Proc. IEEE, vol. 86, no. 11, pp. 2278-2324.
8. O. L. Mangasarian, W. N. Street, and W. H. Wolberg (1995), “Breast cancer diagnosis and prognosis via linear programming,” Operations Research, vol. 43, no. 4, pp. 570-577.
