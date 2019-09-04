import os
import time
import pickle
import numpy as np
import torch


class Experiment():
    def __init__(self,option,learner,data):
        self.option = option
        self.learner = learner
        self.data = data

        self.msg_with_time = lambda msg:\
                            '%s Time elapsed %0.2f hrs (%0.1f mins)'\
                            %(msg,  (time.time()-self.start)/3600,  (time.time()-self.start)/60)
        self.start = time.time()
        self.epoch = 0
        self.best_valid_loss = np.inf
        self.best_valid_in_top = 0.
        self.train_stats = []
        self.valid_stats = []
        self.test_stats = []
        self.early_stopped = False


        self.optimizer = torch.optim.Adam(self.learner.parameters(),self.option.learning_rate,
                                          weight_decay=self.option.weight_decay)

    def one_epoch(self,mode,num_batch,next_fn):
        epoch_loss = []
        epoch_acc = []
        epoch_all_acc = []
        outputs = []
        self.optimizer.zero_grad()
        for batch in range(num_batch):
            input_seq, target, input_len = next_fn()
           # print(torch.sum(input_seq))
            if not self.option.no_cuda:
                input_seq = input_seq.cuda()
                input_len = input_len.cuda()
                target = target.cuda()
            if self.option.model==0: 
                if mode == 'train':
                    output, loss, acc, all_acc = self.learner(input_seq,input_len,target)
                    loss.backward()
                    if self.option.clip_norm>0:
                        torch.nn.utils.clip_grad_norm_(self.learner.parameters(),self.option.clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else:
                    output, loss, acc, all_acc = self.learner(input_seq, input_len, target)
                if batch%500 == 0 and mode=='train':
                    batch_msg = self.msg_with_time('Epoch %d  batch %d, mode %s, Loss %0.4f Acc %0.4f Overall_acc %0.4f'
                            %(self.epoch+1,batch,mode,loss.item(),acc.item(),all_acc.item()))
                    print(batch_msg)
            elif self.option.model==1:

                if mode == 'train':
                    output, loss, acc, all_acc = self.learner(input_seq,target)
                    loss.backward()
                    if self.option.clip_norm>0:
                        torch.nn.utils.clip_grad_norm_(self.learner.parameters(),self.option.clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else:
                    output, loss, acc, all_acc = self.learner(input_seq, target)
                if batch%500 == 0 and mode=='train':
                    batch_msg = self.msg_with_time('Epoch %d  batch %d, mode %s, Loss %0.4f Acc %0.4f Overall_acc %0.4f'
                            %(self.epoch+1,batch,mode,loss.item(),acc.item(),all_acc.item()))
                    print(batch_msg)
            epoch_loss += [loss.item()]
            epoch_acc += [acc.item()]
            epoch_all_acc += [all_acc.item()]
            outputs.append(output.cpu().numpy())
        msg = self.msg_with_time(
            "\nEpoch %d mode %s Loss %0.4f Acc %0.4f Overall_acc %0.4f."
            % (self.epoch + 1, mode, np.mean(epoch_loss), np.mean(epoch_acc), np.mean(epoch_all_acc)))
        print(msg)
        self.write_log_file(msg)
        return outputs, epoch_loss, epoch_acc

    def one_epoch_train(self):
        self.learner.train()
        outputs, loss, in_top = self.one_epoch("train",
                                               self.data.train_num_batch,
                                               self.data.next_train)

        self.train_stats.append([loss, in_top])

    def one_epoch_valid(self):
        self.learner.eval()
        outputs, loss, in_top = self.one_epoch("valid",
                                               self.data.valid_num_batch,
                                               self.data.next_valid)
        self.valid_stats.append([loss, in_top])
        self.best_valid_loss = min(self.best_valid_loss, np.mean(loss))
        self.best_valid_in_top = max(self.best_valid_in_top, np.mean(in_top))

    def one_epoch_test(self):
        self.learner.eval()
        outputs, loss, in_top = self.one_epoch("test",
                                               self.data.test_num_batch,
                                               self.data.next_test)
        self.test_stats.append([loss, in_top])

    def save_model(self):
        loss_improve = self.best_valid_loss == np.mean(self.valid_stats[-1][0])
        in_top_improve = self.best_valid_in_top == np.mean(self.valid_stats[-1][1])
        if in_top_improve:
            with open(self.option.model_path + '-best.pkl', 'wb') as f:
                torch.save(self.learner.state_dict(), f)

    def train(self):
        while (self.epoch < self.option.max_epoch ):
            self.one_epoch_train()
            if self.epoch%2 and self.epoch>20:
                self.one_epoch_valid()
                 #self.one_epoch_test()
                self.save_model()
            self.epoch += 1
        #all_test_in_top = [np.mean(x[1]) for x in self.test_stats]
        #best_test_epoch = np.argmax(all_test_in_top)
       # best_test = all_test_in_top[best_test_epoch]
#
        #msg = "Best test in top: %0.4f at epoch %d." % (best_test, best_test_epoch + 1)
        #print(msg)
        self.write_log_file(msg + "\n")
        #pickle.dump([self.train_stats, self.valid_stats, self.test_stats],
                    #open(os.path.join(self.option.this_expsdir, "results.pckl"), "w"))

    def evaluate(self):

        OUT = []
        for i in range(self.data.v_num_files):
            self.learner.eval()
            outputs, loss, in_top = self.one_epoch("eval",
                                                   self.data.test_num_batch,
                                                   self.data.next_test)
            OUT.extend(outputs)
        return OUT

    def close_log_file(self):
        self.log_file.close()

    def write_log_file(self, string):
        self.log_file = open(os.path.join(self.option.this_expsdir, "log.txt"), "a+")
        self.log_file.write(string + "\n")
        self.log_file.close()
