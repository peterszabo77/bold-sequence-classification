# bold-sequence-classification

An LSTM network based DNA sequence classifier

## About

The Bold Dataset (http://v3.boldsystems.org/) contains DNA barcode samples from various species.
The present DNA sequence classifier works with a small portion of the Bold dataset, containing sequences of
10 different spider genera. (A sample of the training dataset can be found in the 'bolddata' directory.)

The present version of the classifier is a simple one with
- unidirectional LSTM
- no 'word' (nucleotide) embedding
- no attention mechanism

## Software / libraries

Python; pyTorch, matplotlib

## The training process

![training](https://github.com/peterszabo77/bold-sequence-classification/blob/master/images/training.png)

## Output

``
filtering records of the 10 most frequent genera:
Pardosa        4121
Tetragnatha    1049
Xysticus        854
Alopecosa       853
Clubiona        819
Dictyna         706
Neoscona        627
Philodromus     493
Araneus         492
Larinioides     485
Name: genus, dtype: int64

Whole dataset: 10499 records
Train dataset: 8395 records
Evaluation dataset: 2104 records

Using CUDA: False

training...
epoch [1/20] - loss: 2.2547, eloss: 2.2426, acc: 0.1388, eacc: 0.1658
epoch [2/20] - loss: 2.1836, eloss: 2.1924, acc: 0.1961, eacc: 0.2100
epoch [3/20] - loss: 2.0563, eloss: 2.0543, acc: 0.2427, eacc: 0.2392
epoch [4/20] - loss: 1.8978, eloss: 1.9647, acc: 0.2952, eacc: 0.2413
epoch [5/20] - loss: 1.8502, eloss: 1.9410, acc: 0.3106, eacc: 0.2517
epoch [6/20] - loss: 1.8098, eloss: 1.9866, acc: 0.3265, eacc: 0.2714
epoch [7/20] - loss: 1.7743, eloss: 1.8468, acc: 0.3413, eacc: 0.2711
epoch [8/20] - loss: 1.7513, eloss: 1.8029, acc: 0.3498, eacc: 0.3217
epoch [9/20] - loss: 1.6837, eloss: 1.7968, acc: 0.3751, eacc: 0.3270
epoch [10/20] - loss: 1.5781, eloss: 1.6741, acc: 0.3970, eacc: 0.3472
epoch [11/20] - loss: 1.4822, eloss: 1.4912, acc: 0.4149, eacc: 0.3973
epoch [12/20] - loss: 1.5126, eloss: 1.7731, acc: 0.4178, eacc: 0.2902
epoch [13/20] - loss: 1.4801, eloss: 1.4609, acc: 0.4202, eacc: 0.3967
epoch [14/20] - loss: 1.3404, eloss: 1.3969, acc: 0.4776, eacc: 0.4548
epoch [15/20] - loss: 1.2991, eloss: 1.3641, acc: 0.4792, eacc: 0.5007
epoch [16/20] - loss: 1.2865, eloss: 1.3001, acc: 0.4914, eacc: 0.5470
epoch [17/20] - loss: 1.5978, eloss: 2.1256, acc: 0.4116, eacc: 0.2112
epoch [18/20] - loss: 2.0344, eloss: 2.0928, acc: 0.2590, eacc: 0.2360
epoch [19/20] - loss: 1.9616, eloss: 2.0431, acc: 0.2925, eacc: 0.2102

epoch [20/20] - loss: 1.9492, eloss: 2.0252, acc: 0.2811, eacc: 0.2800
``
