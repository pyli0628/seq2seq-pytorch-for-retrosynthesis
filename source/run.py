import argparse
import os
import numpy as np
import random
import torch
import time
from data import Data
from experiment import Experiment
from models import seq2seq,seq2seq_luong
import pickle
class Option(object):
    def __init__(self,d):
        self.__dict__ = d
    def save(self):
        with open(os.path.join(self.this_expsdir, "option.txt"), "w") as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                f.write("%s, %s\n" % (key, str(value)))

parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=33, type=int)
parser.add_argument('--gpu', default="", type=str)
parser.add_argument('--no_cuda', default=False, action='store_true')
#path
parser.add_argument('--exps_dir', default='test', type=str)
parser.add_argument('--exp_name', default=None, type=str)
parser.add_argument('--datadir', default=None, type=str)

#model
parser.add_argument('--model',default=0,type=int)
parser.add_argument('--batch_size',default=32,type=int)
parser.add_argument('--hidden_size',default=512,type=int)
parser.add_argument('--dropout',default=0.2,type=float)
parser.add_argument('--vocab_size',default=65,type=int)
parser.add_argument('--learning_rate',default=0.0001,type=float)
parser.add_argument('--weight_decay',default=0.0,type=float)
parser.add_argument('--clip_norm', default=5.0, type=float)
parser.add_argument('--max_epoch',default=1000,type=int)
parser.add_argument('--layer',default=2,type=int)
parser.add_argument('--decoder_layer',default=4,type=int)

#eval
parser.add_argument('--eval',default=False,action='store_true')
parser.add_argument('--load', default=None, type=str)

d = vars(parser.parse_args())
option = Option(d)
os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu
random.seed(option.seed)
np.random.seed(option.seed)
torch.manual_seed(option.seed)

if option.exp_name is None:
    option.tag = time.strftime("%y-%m-%d-%H-%M")
else:
    option.tag = option.exp_name

option.this_expsdir = os.path.join(option.exps_dir, option.tag)
if not os.path.exists(option.this_expsdir):
    os.makedirs(option.this_expsdir)

option.ckpt_dir = os.path.join(option.this_expsdir, "ckpt")
if not os.path.exists(option.ckpt_dir):
    os.makedirs(option.ckpt_dir)
option.model_path = os.path.join(option.ckpt_dir, "model")

option.save()
print("Option saved.")

data = Data(option)
print('Data prepared')
if option.model==0:
    learner = seq2seq(option)
elif option.model==1:
    learner = seq2seq_luong(option)
print('Model prepared')
experiment = Experiment(option,learner,data)
print('Experiment is ready')

if not option.no_cuda:
    learner = learner.cuda()
if option.load is not None:
    with open(option.load, 'rb') as f:
        learner.load_state_dict(torch.load(f))

if not option.eval:
    print("Start training...")
    experiment.train()
else:
    learner.eval()
    print('Start evaluate....')
    predicts = experiment.evaluate()
    with open(os.path.join(option.this_expsdir, 'predict.pkl'), 'wb') as f:
        pickle.dump(predicts, f)
    print('Output saved')
