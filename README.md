## PALLAS

A Python package for inference of gene regulatory networks from noisy gene expression data

## Document content

- Installation
- Dependencies
- Running instructions
- Input format
- Examples
- Reference

## Installation

**SSH**

`git@github.com:yukuntan92/PALLAS.git`

**HTTPS**

`https://github.com/yukuntan92/PALLAS.git`

## Dependencies (Python 3)

- numpy
- math
- sys
- itertools
- tqdm
- scipy
- operator

## Running instructions (the tool is executed from a command line)

**Two parameters are necessary. (Required)**

- **input**: the data path need to be given

- **data_type**: the tool accept two kinds of data - RNA-Seq data and microarray data. "rnaseq" or "microarray" need to be given.


**There are several parameters that can be adjusted by the user which can make the inference more accurate. (Optional)**

- **noise**: the search space of process noise need to be given with format "x-x" ("lower-upper"). The process noise is the noise when the gene transit from one state to another. 0.05 is default which means 5% probability that gene get opposite value.

- **baseline**: the search space for down expressed gene level is needed with format "x-x". Default range is based on minimum value and mean value of the dataset.

- **delta**: the search space for the difference between down expressed gene level and up expressed gene level is needed with format "x-x". Default range is based on minimum value, maximum value and mean value of the dataset.

- **variance**: the search space for the environment noise is needed with format "x-x". Default range is based on minimum value, maximum value and mean value of the dataset.

- **diff_baseline**: decide whether each gene's down expressed level have a big difference - "True" or "False". Default is "False" which means we treat all the genes have roughly the same down expressed level.

- **diff_delta**: decide whether the difference between down expressed level and up expressed level of each gene has a big difference - "True" or "False". Default is 'False' which means we treat all the genes have roughly the same difference between down expressed level and up expressed level.

- **diff_variance**: decide whether the environment noise of each gene has a big difference - "True" or "False". Default is 'False' which means all genes expression are obtained with the same environment noise.

- **fish**: the number of fishes in the fish school. The more fish means the more accurate result will be found. The default value is set to three times the number of dimensions of the problem.

- **iteration**: the number of iterations for the fishes updated in the fish school. The more iteration also means the obtained result will be more accurate. The default value is set to 5000.

- **lambda**: the parameter of regularization term of LASSO. The larger of lambda, the sparser of the inferred network will be. Default value is 0.01.

- **particle**: the number of particles in the auxiliary particle filter. The more particles involved, the more accurate of the probability will be estimated. Default value is two to the number of genes.

- **depth**: the sequencing depth of gene (needed by RNA-Seq data only). Based on [3]: depth=1.02 (1K-50K reads); depth=22.52 (500K-550K reads); depth=196.43 (5M-5M+50K reads). Default value is 1.02.

- **bias**: the regulation biases can take two values: 1/2 (positive bias) or -1/2 (negative bias). A positve bias means that an equal number of activation and inhibition inputs will produce activation. The default value is False, meaning that there is no prior knownledge of the bias and all the biases for genes should be searched. The user can also input a file which contains the prior knowledge of bias, e.g. known_bias.txt file in the example folder (the first row is title and each column is split by tab).

- **net**: the gene regulatory network which is the goal of this tool. The default value is False, meaning that there is no prior knownledge for the network and all the connections between genes need to be estimated. The user can also input a file which contains the prior knowledge in the case that they already known some interactions between genes, e.g. known_net.txt file in the example folder (the first row is title and each column is split by tab).

- **sample**: the number of samples in the input data. Default is 1.

- **running_time**: the number of times running by the tool. The tool will give ranks for the output results based on log-likelihood score and if the log-likelihood is equal, the sparser network will be ranked ahead.

- **full_info**: the default value if False, which only shows the inferred network. If the value of full_info is True, the result will also show the parameters estimated by the tool, e.g. baseline, delta, environmental noise and log-likelihood.

## Input format

```
A	B	C	D 

49.13	26.11	59.06	53.41

21.66	41.09	29.64	34.08

49.92	43.09	41.98	26.36

52.93	42.17	37.13	44.14

54.06	50.29	53.06	35.29

28.90	35.13	48.87	43.95
```

The 1st line is gene_id. From the 2rd line to the 7th line are the microarray time-series data from time one to time six. Each column is split by tab. If we have two samples and each sample has data length of 3, then the input format should the same with the example above (from 2rd to 4th is the first sample from time one to three and from 5th to 7th is the second sample from time one to time three). Currently, each sample should have the same data length and missing value is not accepted.

## Examples

Use p53-MDM2 negative-feedback gene regulatory network as an example [1]. The microarray synthetic data is generated under DNA damage condition (positive bias for the first gene) with data length equal to forty. In this example, we showed how to use the prior knownledge, and because of the that, the interactions which need to be estimated becomes less, so we set iteration equal to 200 to save time while ensuring accuracy. However, without the prior knownledge or you perfer more stable result, the default value 5000 will be a good selection.

`./PALLAS.py input=example/micro_data.txt data_type=microarray bias=example/known_bias.txt net=example/known_net.txt iteration=200 full_info=True`

```
dict_items([('input', 'example/micro_data.txt'), ('data_type', 'Gaussian'), ('noise', [0.05, 0.05]), ('baseline', [5.2089995, 38.9329854040625]), ('delta', [8.009887228645832, 57.75364759]), ('variance', [0.01, 126.36746947304513]), ('diff_baseline', False), ('diff_delta', False), ('diff_variance', False), ('fish', 24), ('iteration', 200), ('lambda', 0.01), ('particle', 16), ('bias', 'example/known_bias.txt'), ('net', 'example/known_net.txt'), ('sample', 1), ('running_time', 1), ('full_info', True)])
100%|█████████████████████████████████████████████████████████████| 200/200 [01:01<00:00,  3.32it/s]
Ranking 1
Source	Target	Interaction

3	1	inhibition

1	2	activation

3	2	inhibition

4	2	inhibition

2	3	activation

1	4	inhibition

2	4	activation

3	4	activation

bias = [ 0.5 -0.5 -0.5 -0.5]

process noise = 0.05

baseline = 28.84

delta = 21.55

environmental noise = 110.87

log-likelihood = -17.526
```


## Reference

1. Tan, Yukun, Fernando B. Lima Neto, and Ulisses Braga Neto. "PALLAS: Penalized mAximum LikeLihood via pArticle Swarm optimization for Inference of Gene Regulatory Networks from Time Series Data." in process

2. Tan, Yukun, Fernando B. Lima Neto, and Ulisses Braga Neto. "Inference of Gene Regulatory Networks by Maximum-likelihood Adaptive Filtering and Discrete Fish School Search." 2018 IEEE 28th International Workshop on Machine Learning for Signal Processing (MLSP). IEEE, 2018.

3. Imani, Mahdi, and Ulisses M. Braga-Neto. "Maximum-likelihood adaptive filter for partially observed boolean dynamical systems." IEEE Transactions on Signal Processing 65.2 (2016): 359-371.

