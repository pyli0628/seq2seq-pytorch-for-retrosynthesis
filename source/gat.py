import pandas as pd
import numpy as np
import re
from rdkit import Chem
from rdkit.Chem import MolFromSmiles,MolToSmiles
import copy
import os
import argparse
import time,random
import torch
import torch.nn as nn
import numpy as np
from data import Data, Second_data
from experiment import Experiment,Experiment_second
from models import SplitGraph,graph2seq, Graph2seq_lxg
import pickle

class Option(object):
    def __init__(self, d):
        self.__dict__ = d
    def save(self):
        with open(os.path.join(self.this_expsdir, "option.txt"), "w") as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                f.write("%s, %s\n" % (key, str(value)))

def main():
    parser = argparse.ArgumentParser(description="Experiment setup")
    #new
    parser.add_argument('--vocab_size',default=59,type=int)
    # misc
    parser.add_argument('--seed', default=33, type=int)
    parser.add_argument('--gpu', default="", type=str)
    parser.add_argument('--no_train', default=False, action="store_true")
    parser.add_argument('--from_model_ckpt', default=None, type=str)
    parser.add_argument('--no_rules', default=False, action="store_true")
    parser.add_argument('--rule_thr', default=1e-2, type=float)    
    parser.add_argument('--no_preds', default=False, action="store_true")
    parser.add_argument('--get_vocab_embed', default=False, action="store_true")
    parser.add_argument('--exps_dir', default='test', type=str)
    parser.add_argument('--exp_name', default='test', type=str)
    parser.add_argument('--load', default=None, type=str)
    # data property
    parser.add_argument('--datadir', default=None, type=str)
    parser.add_argument('--resplit', default=False, action="store_true")
    parser.add_argument('--no_link_percent', default=0., type=float)
    parser.add_argument('--type_check', default=False, action="store_true")
    parser.add_argument('--domain_size', default=128, type=int)
    parser.add_argument('--no_extra_facts', default=False, action="store_true")
    parser.add_argument('--query_is_language', default=False, action="store_true")
    parser.add_argument('--vocab_embed_size', default=128, type=int)
    # model architecture
    parser.add_argument('--num_layer', default=2, type=int)
    parser.add_argument('--heads', default=4, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--query_embed_size', default=128, type=int)
    # optimization
    parser.add_argument('--batch_size', default=30, type=int)
    parser.add_argument('--print_per_batch', default=3, type=int)
    parser.add_argument('--max_epoch', default=20000, type=int)
    parser.add_argument('--min_epoch', default=20000, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.00, type=float)
    parser.add_argument('--clip_norm', default=0.00, type=float)
    parser.add_argument('--no_norm', default=False, action="store_true")
    parser.add_argument('--no_cuda', default=False, action="store_true")
    parser.add_argument('--local', default=False, action="store_true")
    parser.add_argument('--thr', default=1e-20, type=float)
    parser.add_argument('--dropout', default=0., type=float)
    # evaluation
    parser.add_argument('--get_phead', default=False, action="store_true")
    parser.add_argument('--get_attentions', default=False, action="store_true")
    parser.add_argument('--adv_rank', default=False, action="store_true")
    parser.add_argument('--rand_break', default=False, action="store_true")
    parser.add_argument('--accuracy', default=False, action="store_true")
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--model', default=0, type=int)
    parser.add_argument('--maxlinks', default=1500, type=int)
    parser.add_argument('--accumulate_step', default=1, type=int)
    
    d = vars(parser.parse_args())
    option = Option(d)

    random.seed(option.seed)
    np.random.seed(option.seed)
    torch.manual_seed(option.seed)

    if option.exp_name is None:
      option.tag = time.strftime("%y-%m-%d-%H-%M")
    else:
      option.tag = option.exp_name  
    if option.resplit:
      assert not option.no_extra_facts
    if option.accuracy:
      assert option.top_k == 1
    
    os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu
    if option.model == 0:
        data = Data(option.datadir, option.seed)
    elif option.model == 1:
        data = Second_data(option.datadir,option.seed)
    print("Data prepared.")


    option.this_expsdir = os.path.join(option.exps_dir, option.tag)
    if not os.path.exists(option.this_expsdir):
        os.makedirs(option.this_expsdir)
    option.ckpt_dir = os.path.join(option.this_expsdir, "ckpt")
    if not os.path.exists(option.ckpt_dir):
        os.makedirs(option.ckpt_dir)
    option.model_path = os.path.join(option.ckpt_dir, "model")
    
    option.save()
    print("Option saved.")

    device = torch.device("cuda" if torch.cuda.is_available() and not option.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    option.device = device

    if option.model == 0:
        learner = SplitGraph(option)
        experiment = Experiment(option, learner=learner, data=data)
        print("Experiment created.")
    elif option.model ==1:
        learner = Graph2seq_lxg(option)      
        experiment = Experiment_second(option, learner=learner, data=data)
        print("Experiment created.")
    elif option.model ==2:
        learner = EdgeLearning_gated(option) # best
    elif option.model ==7:
        learner = EdgeLearn2_sp(option) # best

    if torch.cuda.is_available():
        learner.cuda()
        #gpus =  [int(x) for x in option.gpu.split(',')]
        #learner = nn.DataParallel(learner,gpus)
    if option.load is  not None: 
        with open(option.load, 'rb') as f:
            learner.load_state_dict(torch.load(f))
        if option.no_train:
            learner.eval()  

    if not option.no_train:
        print("Start training...")
        experiment.train()
    else:
        print('Start evaluate....')
        predicts = experiment.evaluate()
       
        print('Save output....')
        with open(os.path.join(option.this_expsdir,'predict.pkl'),'wb') as f:
            pickle.dump(predicts,f)
        print('Output saved')
        
    print("="*36 + "Finish" + "="*36)


if __name__ == "__main__":
    main()

