# CharRNN
Pytorch implementation of [Andrei Kapathy's character level recurrent language model](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

Use the data script to download sample shakespeare training data:
```
bash get_data.sh
```

Call the train.py script to train a model:
```
python train.py .
```

The script trains an LSTM character level model on all text files in the directory passed as argument. The model will preiodically be saved in the current dir. Options:

```
usage: train.py [-h] [-clip CLIP] [-dropout DROPOUT] [-seq_length SEQ_LENGTH]
                [-nhid NHID] [-nepoch NEPOCH] [-nlayer NLAYER]
                data_path

positional arguments:
  data_path             data dir

optional arguments:
  -h, --help            show this help message and exit
  -clip CLIP            gradient clip
  -dropout DROPOUT      dropout
  -seq_length SEQ_LENGTH
                        sequence length
  -nhid NHID            hidden units
  -nepoch NEPOCH        epochs
  -nlayer NLAYER        layers
```

To sample text from a use the predict.py script:
```
python predict.py charrrn-checkpoint.pth.tar
```

The model can optionally be provided with seed text. Change the temperature to increase/decrese randomness:

```
usage: predict.py [-h] [-seed SEED] [-steps STEPS] [-temp TEMP] model

positional arguments:
  model         model file

optional arguments:
  -h, --help    show this help message and exit
  -seed SEED    model seed text
  -steps STEPS  steps to sample
  -temp TEMP    temperature
```
