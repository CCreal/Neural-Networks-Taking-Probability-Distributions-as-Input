import argparse
import os
import time
import torch

from data import SynData
from model import ClassifyMMLP
from model import ClassifyMLP
from model import ClassifyLSTM

from torch import optim
# from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils import data
import random
import numpy as np
import pickle


# command line args
parser = argparse.ArgumentParser(description='MNN Experiment')

# optional
parser.add_argument('--number-datasets', type=int, default=10000, metavar='N',
                    help='number of synthetic datasets in collection (default: 10000)')
parser.add_argument('--batch-size', type=int, default=16,
                    help='batch size (of datasets) for training (default: 16)')
parser.add_argument('--sample-size', type=int, default=200,
                    help='number of samples per dataset (default: 200)')
parser.add_argument('--features-dim', type=int, default=1,
                    help='number of features per sample (default: 1)')
parser.add_argument('--distributions', type=str, default='easy',
                    help='which distributions to use for synthetic data '
                         '(easy: (Gaussian, Uniform, Laplacian, Exponential), '
                         'hard: (Bimodal mixture of Gaussians, Laplacian, '
                         'Exponential, Reverse Exponential) '
                         '(default: easy)')
parser.add_argument('--out-dim', type=int, default=64,
                    help='dimension of out (default: 64)')
parser.add_argument('--hidden-dim', type=int, default=128,
                    help='dimension of hidden layers in modules outside statistic network '
                         '(default: 128)')
parser.add_argument('--class-size', type=int, default=4,
                    help='number of distributions (default: 4)')                         
parser.add_argument('--print-vars', type=bool, default=False,
                    help='whether to print all learnable parameters for sanity check '
                         '(default: False)')
parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='learning rate for Adam optimizer (default: 1e-3).')
parser.add_argument('--epochs', type=int, default=50,      #####  default=50
                    help='number of epochs for training (default: 50)')
parser.add_argument('--save_interval', type=int, default=-1,
                    help='number of epochs between saving model '
                         '(default: -1 (save on last epoch))')
parser.add_argument('--cuda', type=bool, default=True,
                    help='whether to use cuda '
                         '(default: True)')
parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--model-type', type=str, default='MNN',
                    help='which model to run: MNN, NN, LSTM')                           
args = parser.parse_args()
# assert args.output_dir is not None
# os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
# os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)

if args.cuda:
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed
    torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)

# experiment start time
time_stamp = time.strftime("%d-%m-%Y-%H-%M-%S")

device="cuda" if args.cuda else "cpu"
def run(model, optimizer, loaders, datasets):
    result_save={}
    epoch_save=[]
    acc_train_save=[]
    acc_val_save=[]

    train_loader, test_loader = loaders
    train_dataset, test_dataset = datasets

    save_interval = args.epochs if args.save_interval == -1 else args.save_interval

    # main training loop
    for epoch in range(args.epochs):
        # train step
        model.train()
        acc = 0
        for i, (batch, label) in enumerate(train_loader):
            batch = batch.to(device)
            label = label.to(device)

            loss, output, label = model.step(batch, label, optimizer)

            prediction = torch.max(output, 1)[1]

            pred_y = prediction.data.cpu().numpy()
            acc += sum((pred_y==label.data.cpu().numpy()))

        acc /= (len(train_dataset))
        # print('epoch:', epoch, 'acc: ', acc)
        acc_train_save.append(acc)
        epoch_save.append(epoch)

        with torch.no_grad():
            correct = 0
            total = 0
            for (batch, label) in test_loader:
                batch = batch.to(device)
                label = label.to(device)
                out = model(batch)
                _, predicted = torch.max(out.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        acc_val_save.append(correct / total) 
        print('epoch:', epoch, 'train acc:', acc, 'val acc:', correct / total)   

        # if (epoch + 1) % save_interval == 0:
        #     save_path = '/models/' + args.model_type + str(args.sample_size)+ str(args.seed)\
        #                 + '-{}.m'.format(epoch + 1)
        #     model.save(optimizer, save_path)
    
    save_path = './models/' + args.model_type + '-sample-size'+ str(args.sample_size)+ '-seed-'+ str(args.seed)\
                + '-{}.file'.format(epoch + 1)
    torch.save(model, save_path)
        
    with torch.no_grad():
        correct = 0
        total = 0
        for (batch, label) in test_loader:
            batch = batch.to(device)
            label = label.to(device)
            out = model(batch)
            _, predicted = torch.max(out.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        print('Test Accuracy of the model on the test: {} %'.format(100 * correct / total))
    result_save['epoch']=epoch_save
    result_save['acc_train']=acc_train_save
    result_save['acc_val']=acc_val_save

    return result_save



def main():
    if args.distributions == 'easy':
        distributions = [
            'gaussian',
            'uniform',
            'laplacian',
            'exponential'
        ]
    else:
        distributions = [
            'mixture of gaussians',
            'laplacian',
            'exponential',
            'reverse exponential'
        ]
    train_dataset = SynData(number_datasets=args.number_datasets,
                                         sample_size=args.sample_size,
                                         number_features=args.features_dim,
                                         distributions=distributions)

    n_test_datasets = args.number_datasets // 10
    test_dataset = SynData(number_datasets=n_test_datasets,
                                        sample_size=args.sample_size,
                                        number_features=args.features_dim,
                                        distributions=distributions)
    datasets = (train_dataset, test_dataset)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=0, drop_last=True)

    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0, drop_last=True)
    loaders = (train_loader, test_loader)

    if args.model_type=='MNN':
        model_kwargs = {
            'batch_size': args.batch_size, 
            'sample_size': args.sample_size,
            'features_dim': args.features_dim, 
            'hidden_dim':args.hidden_dim,
            'out_dim':args.out_dim,
            'class_size':args.class_size,
            'cuda':args.cuda
        }
        model = ClassifyMMLP(**model_kwargs).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        result=run(model, optimizer, loaders, datasets)
    
    elif args.model_type=='NN':
        model_kwargs = {
            'batch_size': args.batch_size, 
            'sample_size': args.sample_size,
            'features_dim': args.features_dim, 
            'hidden_dim':args.hidden_dim,
            'out_dim':args.out_dim,
            'class_size':args.class_size,
            'cuda':args.cuda
        }        
        model = ClassifyMLP(**model_kwargs).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        result=run(model, optimizer, loaders, datasets)
    
    elif args.model_type=='LSTM':
        model_kwargs = {
            'batch_size': args.batch_size, 
            'sample_size': args.sample_size,
            'features_dim': args.features_dim, 
            'hidden_dim':args.hidden_dim,
            'out_dim':args.out_dim,
            'class_size':args.class_size,
            'cuda':args.cuda
        }        
        model = ClassifyLSTM(**model_kwargs).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        result=run(model, optimizer, loaders, datasets)
    
    #with open(f'./results/{args.model_type}-sample-size-{args.sample_size}-seed-{args.seed}.pkl', 'wb') as f:
    #    pickle.dump(result, f)        

if __name__ == '__main__':
    main()
