# implementation of paper 

## env

> python=3.6.5, pytorch=1.6.0, numpy=1.19.2, cudatoolkit=10.1.243, torchvision=0.7.0


## run
> cd synthetic

for M-MLP
> python run.py --model-type "MNN" --sample-size sample-size --seed seed --number-datasets 10000

for MLP
> python run.py --model-type "NN" --sample-size sample-size --seed seed --number-datasets 10000

for LSTM
> python run.py --model-type "LSTM" --sample-size sample-size --seed seed --number-datasets 10000