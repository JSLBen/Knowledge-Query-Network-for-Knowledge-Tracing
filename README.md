# PyTorch Implementation of [Knowledge Query Network for Knowledge Tracing`[LAK19]`](https://dl.acm.org/citation.cfm?id=3303772.3303786)
(*Dropout* not used in the implementation.)
# Requirements
1. Python 3.6
2. PyTorch 1.0
3. tensorboardX

# Datasets
*Datasets are borrowed from [DKVMN](https://github.com/jennyzhang0215/DKVMN) for fair comparisons.*
* ASSISTments 2009-2010
* ASSISTments 2015
* Statics 2011
* Synthetic-5

# Running examples
* python3 main.py --gpu 0 --dataset assist0910
* python3 main.py --gpu 0 --dataset assist15
* python3 main.py --gpu 0 --dataset statics11
* python3 main.py --gpu 0 --dataset synthetic-5 --version 0
* python3 main.py --gpu 0 --dataset synthetic-5 --version 19
