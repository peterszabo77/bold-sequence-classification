# bold-sequence-classification

An LSTM network based DNA sequence classifier

## About

The Bold Dataset (http://v3.boldsystems.org/) contains DNA barcode samples from various species.
The present DNA sequence classifier works with a small portion of the Bold dataset,containing barcode sequence data
from specimen of 10 different spider genera. (A sample of the training dataset can be found in the 'bolddata' directory.)

The present version of the classifier is a simple one with
- unidirectional LSTM
- no 'word' (nucleotide) embedding
- no attention mechanism

## Software / libraries

Python; pyTorch, matplotlib

## Output


## The training process

![training](https://github.com/peterszabo77/bold-sequence-classification/blob/master/images/training.png)
