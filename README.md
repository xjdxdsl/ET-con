# et-learn
Learning bounded treewidth Bayesian networks in the space of elimination orders
## Introduction
The computational complexity of inference is now one of the most relevant topics in the field of Bayesian networks. Although the literature contains approaches that learn Bayesian networks from high dimensional datasets, traditional methods do not bound the inference complexity of the learned models, often producing models where exact inference is intractable. This paper focuses on learning tractable Bayesian networks from data. To address this problem, we propose strategies for learning Bayesian networks in the space of elimination orders. In this manner, we can efficiently bound the inference complexity of the networks during the learning process. Searching in the combined space of directed acyclic graphs and elimination orders can be extremely computationally demanding. We demonstrate that one type of elimination trees, which we define as valid, can be used as an equivalence class of directed acyclic graphs and elimination orders, removing redundancy. We propose methods for incrementally compiling local changes made to directed acyclic graphs in elimination trees and for searching for elimination trees of low width. Using these methods, we can move through the space of valid elimination trees in polynomial time with respect to the number of network variables and in linear time with respect to treewidth. Experimental results show that our approach successfully bounds the inference complexity of the learned models, while it is competitive with other state-of-the-art methods in terms of fitting to data.

##Prerequirements and installing guide
This software has been developed as a Python 2.7.15 package and includes some functionalities in Cython and C++11 (version 5.4.0). Consequently, it is needed a Python environment and internet connectivity to download additional package dependencies. Python software can be downloaded from <https://www.python.org/downloads/>.

We provide the steps for a clean installation in Ubuntu 16.04. This software has not been tried under Windows.

The package also uses the following dependencies. 

|Library    |Version|License|
|-----------|-------|-------|
| pandas    |   0.23|  BSD 3|
|  numpy    | 1.14.3|    BSD|
| Cython    | 0.28.2| Apache|

They can be installed through the following sentence:
sudo pip install "Library" 
where "Library" must be replaced by the library to be installed.

Open the folder where you have saved TSEM project files (e.g., "~/Downloads/et-learn") and compile Cython files running the following command in the command console:
python compile.py

## Example.py
File "example.py" provides a demo that shows how to use the code to learn Bayesian networks with bounded treewidth

