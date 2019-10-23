# bold-sequence-classification

An LSTM network based DNA sequence classifier

## About

The Bold Dataset (http://v3.boldsystems.org/) contains DNA barcode samples from various species.
The present DNA sequence classifier works with a small portion of the Bold dataset, containing sequences of
10 different spider genera. (A sample of the training dataset can be found in the 'bolddata' directory.)

The present version of the classifier is a bidirectional LSTM with 'word' (nucleotide) embedding

## Software / libraries

Python; pyTorch, matplotlib

## The training process

![training](https://github.com/peterszabo77/bold-sequence-classification/blob/master/images/training.png)

## Per-class accuracy

![training](https://github.com/peterszabo77/bold-sequence-classification/blob/master/images/evaluation.png)

## Output

```
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
epoch [1/20] - loss: 1.7968, eloss: 0.9919, acc: 0.3614, eacc: 0.6780
epoch [2/20] - loss: 0.5932, eloss: 0.4154, acc: 0.8077, eacc: 0.8511
epoch [3/20] - loss: 0.2978, eloss: 0.2620, acc: 0.9117, eacc: 0.9310
epoch [4/20] - loss: 0.2032, eloss: 0.2271, acc: 0.9447, eacc: 0.9276
epoch [5/20] - loss: 0.1667, eloss: 0.1456, acc: 0.9505, eacc: 0.9677
epoch [6/20] - loss: 0.1395, eloss: 0.1386, acc: 0.9614, eacc: 0.9517
epoch [7/20] - loss: 0.1199, eloss: 0.1666, acc: 0.9667, eacc: 0.9463
epoch [8/20] - loss: 0.1030, eloss: 0.1357, acc: 0.9705, eacc: 0.9647
epoch [9/20] - loss: 0.0955, eloss: 0.1096, acc: 0.9729, eacc: 0.9722
epoch [10/20] - loss: 0.0860, eloss: 0.1255, acc: 0.9755, eacc: 0.9686
epoch [11/20] - loss: 0.1023, eloss: 0.0760, acc: 0.9706, eacc: 0.9749
epoch [12/20] - loss: 0.0599, eloss: 0.0870, acc: 0.9839, eacc: 0.9781
epoch [13/20] - loss: 0.0517, eloss: 0.1128, acc: 0.9840, eacc: 0.9658
epoch [14/20] - loss: 0.0923, eloss: 0.1794, acc: 0.9724, eacc: 0.9360
epoch [15/20] - loss: 0.0635, eloss: 0.0685, acc: 0.9794, eacc: 0.9790
epoch [16/20] - loss: 0.0634, eloss: 0.0641, acc: 0.9799, eacc: 0.9775
epoch [17/20] - loss: 0.0469, eloss: 0.0828, acc: 0.9837, eacc: 0.9746
epoch [18/20] - loss: 0.0344, eloss: 0.0778, acc: 0.9880, eacc: 0.9792
epoch [19/20] - loss: 0.0516, eloss: 0.0671, acc: 0.9831, eacc: 0.9793
epoch [20/20] - loss: 0.0926, eloss: 0.1285, acc: 0.9737, eacc: 0.9631

final evaluation...
```
