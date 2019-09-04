import torch
import torch.nn as nn

USE_CUDA = torch.cuda.is_available()

class Encoder(nn.Module):
    def __init__(self,vocab_sz,hidden_sz,dropout):
        super(Encoder,self).__init__()
        self.embed = nn.Embedding(vocab_sz,hidden_sz)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_sz,hidden_sz)

    def forward(self, input_seq,input_len,hidden=None):
        # input_seq: (max_len,bz); input_len: (bz,)
        input_embed = self.embed(input_seq)#(max_len,bz,vocab_sz)
        input_embed = self.drop(input_embed)
        # sort by length and pack it
        len_sort,indices = torch.sort(input_len,descending=True)
        input_embed = input_embed[:,indices,:] 
        packed = nn.utils.rnn.pack_padded_sequence(input_embed,len_sort.cpu().numpy())
        output,hidden = self.gru(packed,hidden)
        output,_ = nn.utils.rnn.pad_packed_sequence(output)
        #restore the original order
        _,idx = torch.sort(indices)
        output = output[:,idx]
        hidden = hidden[:,idx]
        return  output,hidden #output:( max_len,bz,hz) hidden:(1,bz,hz)

class EncoderLstm(nn.Module):
    def __init__(self,vocab_sz,hidden_sz,layer,dropout):
        super(EncoderLstm,self).__init__()
        
        self.embed=nn.Embedding(vocab_sz,hidden_sz)
        self.drop =nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_sz,hidden_sz,layer,dropout=dropout,bidirectional=True)
    
    def forward(self,input_seq):
        input_embed = self.drop(self.embed(input_seq))
        out,(hidden,cell) = self.lstm(input_embed)

        return out,(hidden,cell)# (L,B,2H) (layer*2,B,H) (layer*2, B, H)

class Attention(nn.Module): 
    def __init__(self,hidden_sz):
        super(Attention,self).__init__()
        self.mlp = nn.Linear(hidden_sz,hidden_sz)
    def forward(self, encoder_output,decoder_hidden):
        #encoder_output: (L,B,H); decoder_hidden: (B,H)
        hidden = nn.ReLU()(self.mlp(decoder_hidden))#(B,H)
        e = torch.bmm(encoder_output.permute(1,0,2),hidden.unsqueeze(2))#(B,L,H) * (B,H,1) >> (B,L,1)
        att = nn.Softmax(dim=1)(e)
        return att #(B,L,1)
class AttentionLuong(nn.Module):
    def __init__(self,hidden_sz):
        self.mlp=nn.Linear(hidden_sz,2*hidden_sz)
    def forward(self,encoder_out,decoder_out):
        #encoder_out:(L,B,2H) decoder_out:(L',B,H)
        decoder_out =nn.ReLU()(self.mlp(decoder_out))#(L',B,2H)
        e= torch.bmm(encoder_out.permute(1,0,2),decoder_out.permute(1,2,0))#''(B,L,2H)*(B,2H,L') ''
        att = nn.Softmax(dim=1)(e)#(B,L,L')
        return att
class seq2seq_luong(nn.Module):
    def __init__(self,option):
        super(seq2seq_luong,self).__init__()
        self.vocab_sz = option.vocab_size
        self.hidden_sz = option.hidden_size
        self.att = Attention(self.hidden_sz)
        self.layer = option.decoder_layer
        self.encoder = EncoderLstm(self.vocab_sz,self.hidden_sz,self.layer,option.dropout)
        self.dec_embed=nn.Embedding(self.vocab_sz,self.hidden_sz)
        self.drop = nn.Dropout(option.dropout)
        self.lstm1 = nn.LSTMCell(self.hidden_sz,self.hidden_sz)
        self.lstm2 = nn.LSTMCell(self.hidden_sz,self.hidden_sz)
        self.lstm3 = nn.LSTMCell(self.hidden_sz,self.hidden_sz)
        self.lstm4 = nn.LSTMCell(self.hidden_sz,self.hidden_sz)
        self.mlp1 = nn.Linear(self.hidden_sz,self.hidden_sz)
        self.mlp2 = nn.Linear(2*self.hidden_sz,self.hidden_sz)
        self.out =nn.Linear(self.hidden_sz,self.vocab_sz)
    def forward(self,input_seq,target_seq):
        batch_sz = input_seq.size(1)
        encoder_output,hidden = self.encoder(input_seq)#(L,B,2H)
        #print(encoder_output.shape)
        try:
            maxlen = target_seq.size(0)
        except:
            maxlen = 140
        decoder_hidden = hidden[0].view(-1,batch_sz,2*self.hidden_sz)[-1].squeeze() #(b,2h) 
        decoder_cell = hidden[1].view(-1,batch_sz,2*self.hidden_sz)[-1].squeeze() #(b,2h)
        
        encoder_output = self.mlp2(encoder_output)#(l,b,h)
        decoder_hidden = self.mlp2(decoder_hidden)#(b,h)
        decoder_cell = self.mlp2(decoder_cell)#(b,h)
        h2 = torch.zeros(batch_sz,self.hidden_sz)
        h3 = torch.zeros(batch_sz,self.hidden_sz)
        h4 = torch.zeros(batch_sz,self.hidden_sz)
        c2 = torch.zeros(batch_sz,self.hidden_sz)
        c3 = torch.zeros(batch_sz,self.hidden_sz)
        c4 = torch.zeros(batch_sz,self.hidden_sz)
        decoder_input = torch.LongTensor([[1]*batch_sz])#(1,b)
        all_decoder_outputs = torch.zeros(maxlen,batch_sz,self.vocab_sz)
        inference = torch.zeros(maxlen,batch_sz)
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            all_decoder_outputs = all_decoder_outputs.cuda()
            inference = inference.cuda()
            h2=h2.cuda()
            h3=h3.cuda()
            h4=h4.cuda()
            c2=c2.cuda()
            c3=c3.cuda()
            c4=c4.cuda()
        if self.training:
            for t in range(maxlen):
                decoder_input = self.dec_embed(decoder_input).squeeze(0)#(b,h)
                decoder_input = self.drop(decoder_input)
                attention = self.att(encoder_output,decoder_hidden) #(B,L,1)
                context = torch.bmm(encoder_output.permute(1,2,0),attention).squeeze(2)#(B,H)
                decoder_input = torch.cat((decoder_input, context), dim=1)  # (b,2*h)
               # print(decoder_input.size())
                decoder_input = nn.Tanh()(self.mlp2(decoder_input)) #(b,h)
                    #print(decoder_input.shape,decoder_hidden.shape,decoder_cell.shape)
                decoder_hidden,decoder_cell = self.lstm1(decoder_input,(decoder_hidden,decoder_cell)) #(b,h)
                h2,c2 = self.lstm2(decoder_hidden,(h2,c2))
                h3,c3 = self.lstm3(h2,(h3,c3))
                h4,c4 = self.lstm4(h3,(h4,c4))
                decoder_output = nn.Softmax(1)(self.out(h4)) #(b,vocab_sz)

                all_decoder_outputs[t] = decoder_output
                topv,topi = torch.max(decoder_output,1)
                inference[t]=topi

                decoder_input = target_seq[t].view(1, batch_sz)
        else:
            for t in range(maxlen):
                decoder_input = self.dec_embed(decoder_input).squeeze(0)
                attention = self.att(encoder_output, decoder_hidden)  # (B,L,1)
                context = torch.bmm(encoder_output.permute(1, 2, 0), attention).squeeze(2)  # (B,H)
                decoder_input = torch.cat((decoder_input, context), dim=1)  # (b,2*h)
                decoder_input = nn.Tanh()(self.mlp2(decoder_input))  # (b,h)
                decoder_hidden,decoder_cell = self.lstm1(decoder_input,(decoder_hidden,decoder_cell)) #(b,h)
                h2,c2 = self.lstm2(decoder_hidden,(h2,c2))
                h3,c3 = self.lstm3(h2,(h3,c3))
                h4,c4 = self.lstm4(h3,(h4,c4))
                decoder_output = nn.Softmax(1)(self.out(h4)) #(b,vocab_sz)
                
                all_decoder_outputs[t] = decoder_output
                topv, topi = decoder_output.data.topk(1) #(b,1)
                inference[t] = topi.squeeze()
                decoder_input= topi.permute(1,0)
        loss = cal_loss(all_decoder_outputs.view(-1, self.vocab_sz), target_seq.contiguous().view(-1, 1))
        loss = torch.mean(loss)
        # acc
        temacc = torch.eq(inference.long(), target_seq).float()  # K,bs
        all_acc = torch.mean(temacc)
        acc = torch.eq(torch.sum(1 - temacc, 0), 0)
        acc = torch.mean(acc.float())
        return inference, loss, acc, all_acc

class seq2seq(nn.Module):
    def __init__(self,option):
        super(seq2seq,self).__init__()
        self.vocab_sz = option.vocab_size
        self.hidden_sz = option.hidden_size

        self.att = Attention(self.hidden_sz)
        self.encoder = Encoder(self.vocab_sz,self.hidden_sz,option.dropout)
        self.dec_embed = nn.Embedding(self.vocab_sz,self.hidden_sz)
        self.drop = nn.Dropout(option.dropout)
        self.decoder = nn.GRUCell(self.hidden_sz,self.hidden_sz)
        self.mlp = nn.Linear(2*self.hidden_sz,self.hidden_sz)
        self.out = nn.Linear(self.hidden_sz,self.vocab_sz)

    def forward(self,input_seq,input_len,target_seq):
        #input_seq:(L,b), target_seq: (L,B)
        batch_sz = input_seq.size(1)
        encoder_output,encoder_hidden = self.encoder(input_seq,input_len)
        #output: (L,B,H) ; hidden: (1,B,H)
        decoder_hidden = encoder_hidden.squeeze() #b,h
        try:
            maxlen = target_seq.size(0)
        except:
            maxlen = 150

        decoder_input = torch.LongTensor([[1]*batch_sz])
        all_decoder_outputs = torch.zeros(maxlen,batch_sz,self.vocab_sz)
        inference = torch.zeros(maxlen,batch_sz)
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            all_decoder_outputs = all_decoder_outputs.cuda()
            inference = inference.cuda()
        if self.training:
            for t in range(maxlen):
                decoder_input = self.dec_embed(decoder_input).squeeze(0)
                decoder_input = self.drop(decoder_input)
                attention = self.att(encoder_output,decoder_hidden) #(B,L,1)
                context = torch.bmm(encoder_output.permute(1,2,0),attention).squeeze(2)#(B,H)
                decoder_input = torch.cat((decoder_input, context), dim=1)  # (b,2*h)
                decoder_input = nn.Tanh()(self.mlp(decoder_input)) #(b,h)

                decoder_hidden = self.decoder(decoder_input,decoder_hidden) #(b,h)
                decoder_output = nn.Softmax(1)(self.out(decoder_hidden)) #(b,vocab_sz)

                all_decoder_outputs[t] = decoder_output
                topv,topi = torch.max(decoder_output,1)
                inference[t]=topi

                decoder_input = target_seq[t].view(1, batch_sz)
        else:
            for t in range(maxlen):
                decoder_input = self.dec_embed(decoder_input).squeeze(0)
                attention = self.att(encoder_output, decoder_hidden)  # (B,L,1)
                context = torch.bmm(encoder_output.permute(1, 2, 0), attention).squeeze(2)  # (B,H)
                decoder_input = torch.cat((decoder_input, context), dim=1)  # (b,2*h)
                decoder_input = nn.Tanh()(self.mlp(decoder_input))  # (b,h)

                decoder_hidden = self.decoder(decoder_input, decoder_hidden)  # (b,h)
                decoder_output = nn.Softmax(1)(self.out(decoder_hidden))  # (b,vocab_sz)

                all_decoder_outputs[t] = decoder_output
                topv, topi = decoder_output.data.topk(1) #(b,1)
                inference[t] = topi.squeeze()
                decoder_input= topi.permute(1,0)


        loss = cal_loss(all_decoder_outputs.view(-1, self.vocab_sz), target_seq.contiguous().view(-1, 1))
        loss = torch.mean(loss)
        # acc
        temacc = torch.eq(inference.long(), target_seq).float()  # K,bs
        all_acc = torch.mean(temacc)
        acc = torch.eq(torch.sum(1 - temacc, 0), 0)
        acc = torch.mean(acc.float())
        return inference, loss, acc, all_acc

def cal_loss(dist, target):
    #N,C; N,1
    min_num = torch.tensor([1e-10])
    if USE_CUDA:
        min_num = min_num.cuda()
    distribution = torch.max(dist,min_num)
    yonehot = torch.zeros_like(distribution)
    ones = torch.ones_like(target).float()
    yonehot.scatter_(1,target, ones)
    ylogp = torch.log(distribution)*yonehot
    loss = torch.sum(-ylogp,1,keepdim=True)
    return loss
