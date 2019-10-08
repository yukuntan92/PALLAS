PAMFA
======

A Python package for inference of gene regulatory networks from noisy gene expression data
Version 0.0.1

Document content
----

* Overview
* Installation
* Dependencies
* Running instructions
* Input format
* Examples
* Reference

Overview
----


Installation
----

SSH

`git@github.tamu.edu:yukuntan/Gene-regulatory-network-inference.git`

HTTPS

`https://github.tamu.edu/yukuntan/Gene-regulatory-network-inference.git`

Dependencies (Python 3)
----

* numpy
* math
* sys
* random
* itertools
* collections
* tqdm
* scipy


Running instructions
----

The tool is executed from a command line.

Required:
./main.py data=file_name data_type=rnaseq (or microarray)

There are several parameters that can be adjusted by the user which can make the inference more accurate.

- noise: is the search space of process noise. The process noise is the noise when the gene transit from one state to another. 0-0.1 is default which means at most 10% probability that gene get opposite value.

- baseline: is the search space for down expressed gene level. Default range is based on minimum value and mean value of the dataset.

- delta: is the search space for the difference between down expressed gene level and up expressed gene level. Default range is based on minimum value, maximum value and mean value of the dataset.

- variance: is the environment noise for the data. Default range is based on minimum value, maximum value and mean value of the dataset.

- diff_baseline: is to decide whether each gene's down expressed level have a big difference. Default is 'False' which means we treat all the genes have roughly the same down expressed level.

- diff_delta: is to decide whether the difference between down expressed level and up expressed level of each gene has a big difference. Default is 'False' which means we treat all the genes have roughly the same difference between down expressed level and up expressed level.

- diff_variance: is to decide whether the environment noise of each gene is roughly the same or not. Default is 'False' which means all genes expression are obtained with the same environment noise.

- fish: is the number of fishes in the fish school. The more fish means the more accurate result will be found. The default value is set to three times the number of variables.

- iteration: is the number of iterations for the fishes updated in the fish school. The more iteration also means the obtained result will be more accurate. The default value is set to 5000.

- lambda: is the parameter of regularization term of LASSO. The larger of lambda, the sparse of the inferred network will be. Default value is 0.01.

- particle: is the number of particles in the auxiliary particle filter. The more particles involved, the more accurate of the probability will be estimated. Default value is two to the number of genes.

- depth: is the sequencing depth of gene when we use rnaseq data. Based on the paper from Imani, M., & Braga-Neto, U. M. (2016). "Maximum-likelihood adaptive filter for partially observed boolean dynamical systems. IEEE Transactions on Signal Processing, 65(2), 359-371.", depth=1.02 (1K-50K reads); depth=22.52 (500K-550K reads); depth=196.43 (5M-5M+50K reads). Default value is 22.52.

- input: is the input sequence to gene network, e.g. if we know Gene 1 is damaged, then input=1. Default is 'False'.


Input format
----

A	B	C	D 

49.45	30.03	58.82	45.05 

25.28	36.16	32.40	30.23

41.18	31.44	26.09	25.52

The first line is gene_id. Each column is split by tab.


Examples
----

./main.sh data=../example/nb_data.txt data_type=rnaseq iteration=1000 input=0

dict_items([('data', '/Users/yukuntan/Desktop/package/example/nb_data.txt'), ('data_type', 'NB'), ('noise', [0, 0.1]), ('baseline', [0, 6.491481972157369]), ('delta', [0.6931471805599454, 7.1846291527173145]), ('variance', [0.5, 6]), ('diff_baseline', False), ('diff_delta', False), ('diff_variance', False), ('fish', 60), ('iteration', 1000), ('lambda', 0.01), ('particle', 16), ('depth', 22.52), ('input', 0)])
100%|█████████████████████████████████████| 1000/1000 [1:17:13<00:00,  5.44s/it]

     Source	Target	Interaction

     1	1	activation

     3	1	inhibition

     4	1	activation

     1	2	activation

     3	2	inhibition

     4	2	inhibition

     2	3	activation

     3	4	activation

     process noise	=	0.067

     baseline	=	1.226

     delta	=	1.686

     environmental noise	=	0.969

Reference
----

Tan, Yukun, Fernando B. Lima Neto, and Ulisses Braga Neto. "Construction of Gene Regulatory Networks
Using Partially Observable Boolean Dynamical System and Swarm Intelligence." in process for IEEE Transactions on Signal Processing.

Tan, Yukun, Fernando B. Lima Neto, and Ulisses Braga Neto. "INFERENCE OF GENE REGULATORY NETWORKS BY MAXIMUM-LIKELIHOOD ADAPTIVE FILTERING AND DISCRETE FISH SCHOOL SEARCH." 2018 IEEE 28th International Workshop on Machine Learning for Signal Processing (MLSP). IEEE, 2018.




