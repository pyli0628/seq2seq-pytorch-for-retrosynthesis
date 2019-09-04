import os
import torch

def Gen_vocab(path):
    with open(path,'r') as f:
        vocab = f.read()
    vocab = vocab.split('\n')
    vocab.pop()
    char = ['<PAD>','<SOS>','<EOS>','<UNK>']
    word_idx = {word: idx for idx, word in enumerate(char+vocab)}
    idx_word = {idx: word for idx, word in enumerate(char+vocab)}
    return word_idx,idx_word
def Load_smi(path):
    with open(path,'r') as f:
        data = f.read()
    data = data.split('\n')
    data.pop()
    data = list(map(lambda x:x.split(' '), data))
    return data
def Smi2vec(data, word_id, data_type):

    unk_idx = word_id.get("<UNK>")
    eos_idx = word_id.get('<EOS>')
    if data_type=='source':
        vec = list(list(map(lambda x:word_id.get(x,unk_idx),smi)) for smi in data)
    else:
        vec = list(list(map(lambda x:word_id.get(x,unk_idx),smi))+[eos_idx] for smi in data)
    return vec

def Pad(vec, max_length):
    vec = vec + [0 for i in range(max_length - len(vec))]
    return vec
def Pad_batch(data,word_id,data_type):
    vec = Smi2vec(data,word_id,data_type)
    length = [len(s) for s in vec]
    max_len = max(length)
    pad_vec = [Pad(s,max_len) for s in vec]

    pad_vec = torch.LongTensor(pad_vec).transpose(1,0) #(L,B)
    length = torch.LongTensor(length)
    if torch.cuda.is_available():
        pad_vec = pad_vec.cuda()
        length = length.cuda()
    return pad_vec,length

class Data():
    def __init__(self,option):
        self.option = option
        self.batch_sz = option.batch_size
        self.word_id,self.id_word = Gen_vocab(os.path.join(self.option.datadir+'vocab'))
        if not option.eval:
            self.train_source = Load_smi(os.path.join(self.option.datadir,'train_sources'))
            self.train_target = Load_smi(os.path.join(self.option.datadir,'train_targets'))
            self.valid_source = Load_smi(os.path.join(self.option.datadir,'valid_sources'))
            self.valid_target = Load_smi(os.path.join(self.option.datadir,'valid_targets'))
            self.test_source = Load_smi(os.path.join(self.option.datadir,'test_sources'))
            self.test_target = Load_smi(os.path.join(self.option.datadir,'test_targets'))
            self.train_num_batch = len(self.train_source) // self.batch_sz + 1
            self.valid_num_batch = len(self.valid_source) // self.batch_sz + 1
            self.test_num_batch = len(self.test_source) // self.batch_sz + 1
            if not len(self.train_source)%self.batch_sz:
                self.train_num_batch = self.train_num_batch-1
            if not len(self.valid_source)%self.batch_sz:
                self.valid_num_batch = self.valid_num_batch-1
            if not len(self.test_source)%self.batch_sz:
                self.test_num_batch = self.test_num_batch-1
            self.train_start = 0
            self.valid_start = 0
            self.test_start = 0
        else:
            self.test_source = Load_smi(os.path.join(self.option.datadir,'test_sources'))
            self.test_target = Load_smi(os.path.join(self.option.datadir,'test_targets'))
            self.test_num_batch = len(self.test_source) // self.batch_sz + 1
            self.test_start = 0

    def neat_batch(self,source,target,start,num_batch):
        source = source[start*self.batch_sz:(start+1)*self.batch_sz]
        target = target[start * self.batch_sz:(start + 1) * self.batch_sz]
        start = (start+1)%num_batch
        source_batch,source_len = Pad_batch(source,self.word_id,data_type='source')
        target_batch, target_len = Pad_batch(target,self.word_id,data_type='target')
        return source_batch,target_batch,source_len,start
    def next_train(self):
        source_batch,target_batch,source_len,self.train_start = self.neat_batch(self.train_source,self.train_target,
                                                    self.train_start,self.train_num_batch)
        return source_batch,target_batch,source_len
    def next_valid(self):
        source_batch,target_batch,source_len,self.valid_start  = self.neat_batch(self.valid_source,self.valid_target,
                                                    self.valid_start,self.valid_num_batch)
        return source_batch,target_batch,source_len
    def next_test(self):
        source_batch,target_batch,source_len,self.test_start  = self.neat_batch(self.test_source,self.test_target,
                                                    self.test_start,self.test_num_batch)
        return source_batch,target_batch,source_len
