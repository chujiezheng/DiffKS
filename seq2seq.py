# coding:utf-8
import logging

import torch
from torch import nn, optim
import numpy as np
import tqdm
import os

from utils import Storage, cuda, BaseModel, CheckpointManager
from network import Network

class Seq2seq(BaseModel):
    def __init__(self, param):
        args = param.args
        net = Network(param)
        self.optimizer = optim.Adam(net.get_parameters_by_name(), lr=args.lr)
        checkpoint_manager = CheckpointManager(args.name, args.model_dir, 
                        args.checkpoint_steps, args.checkpoint_max_to_keep, "max")
        super().__init__(param, net, checkpoint_manager)

    def _preprocess_batch(self, data):
        incoming = Storage()
        incoming.data = data = Storage(data)
        incoming.data.batch_size = data.post.shape[0]
        #incoming.data.post = cuda(torch.LongTensor(data.post)) # length * batch_size
        incoming.data.resp = cuda(torch.LongTensor(data.resp)) # length * batch_size
        incoming.data.atten = cuda(torch.LongTensor(data.atten))
        incoming.data.wiki = cuda(torch.LongTensor(data.wiki))
        return incoming
    
    def get_next_batch(self, dm, key, restart=True):
        data = dm.get_next_batch(key)
        if data is None:
            if restart:
                dm.restart(key)
                return self.get_next_batch(dm, key, False)
            else:
                return None
        return self._preprocess_batch(data)

    def get_batches(self, dm, key):
        batches = list(dm.get_batches(key, batch_size=self.args.batch_size, shuffle=False))
        return len(batches), (self._preprocess_batch(data) for data in batches)

    def get_select_batch(self, dm, key, i):
        data = dm.get_batch(key, i)
        if data is None:
            return None
        return self._preprocess_batch(data)

    def train(self):
        args = self.param.args
        dm = self.param.volatile.dm
        datakey = 'train'
        
        i = 0
        while True:
            incoming = self.get_next_batch(dm, datakey, restart=False)
            
            if incoming is None:
                break
            incoming.args = Storage()
            
            if (i + 1) % args.batch_num_per_gradient == 0:
                self.zero_grad()
            self.net.forward(incoming)
            
            loss = incoming.result.loss
            if (i + 1) % 100 == 0:
                logging.info("epoch %d batch %d : gen loss=%f", self.now_epoch, i + 1, loss.detach().cpu().numpy())
            
            loss.backward()
            
            if (i + 1) % args.batch_num_per_gradient == 0:
                nn.utils.clip_grad_norm_(self.net.parameters(), args.grad_clip)
                self.optimizer.step()
                
            i += 1
    
    
    def train_process(self):
        args = self.param.args
        dm = self.param.volatile.dm

        while self.now_epoch < args.epochs:
            self.now_epoch += 1

            dm.restart('train', args.batch_size)
            self.net.train()
            self.train()

            self.net.eval()
            bleu = self.test('dev', False)['bleu-4']
            logging.info("epoch %d, evaluate dev, bleu-4 %f", self.now_epoch, bleu)
            logging.info("    best bleu-4 %f", self.checkpoint_manager.best_value)

            self.save_checkpoint(value=bleu)

    def test(self, key, write=True):
        args = self.param.args
        dm = self.param.volatile.dm
        
        from myCoTK.metric import MyRougeMetric
        rouge_metric = MyRougeMetric(dm, multi_turn_gen_key='multi_turn_gen',
                                                    multi_turn_reference_allvocabs_key="sent_allvocabs",
                                                    turn_len_key="turn_length")
        metric2 = dm.get_inference_metric()
        batch_num, batches = self.get_batches(dm, key)
        logging.info("eval free-run")
        label = []
        pred = []
        #prob = []
        for incoming in tqdm.tqdm(batches, total=batch_num):
            incoming.args = Storage()
            with torch.no_grad():
                self.net.detail_forward(incoming)
            data = Storage()
            data.resp = incoming.data.resp.detach().cpu().numpy()
            data.turn_length = incoming.data.sen_num
            data.multi_turn_gen = incoming.state.w_o_all.detach().cpu().numpy().transpose(2, 0, 1)
            data.sent_allvocabs = incoming.data.resp_allvocabs
            metric2.forward(data)
            rouge_metric.forward(data)
            
            label += np.array(incoming.acc.label, dtype=int).transpose().tolist()
            pred += np.array(incoming.acc.pred, dtype=int).transpose().tolist()
            
        
        res = metric2.close()
        res.update(rouge_metric.close())
        assert len(label) == len(pred)

        if write:
    
            if not os.path.exists(args.out_dir):
                os.makedirs(args.out_dir)
            filename = args.out_dir + "/%s_%s.txt" % (args.name, key)
    
            used_know = []
            valid_label = []
            valid_pred = []
            for i in range(len(res['gen'])):
                used_know.append([])
                for j in range(len(res['gen'][i])):
                    valid_label.append(label[i][j])
                    valid_pred.append(pred[i][j])
                    used_know[-1].append(' '.join(dm.convert_ids_to_tokens(dm.data[key]['wiki'][i][j][pred[i][j]])))

            res['acc'] = 100 * np.mean(np.array(valid_label) == np.array(valid_pred))
    
            import json
            with open(args.out_dir + '/acc_%s_%s.json' % (args.name, key), 'w') as f:
                json.dump({'pred': valid_pred,
                           'label': valid_label,
                           }, f, sort_keys=True)
            
    
            with open(filename, 'w') as f:
                logging.info("%s Test Result:", key)
                for each in sorted(list(res.keys())):
                    value = res[each]
                    if isinstance(value, float):
                        logging.info("\t%s:\t%f", each, value)
                        f.write("%s:\t%f\n" % (each, value))
                f.write('\n')
                for i in range(len(res['reference'])):
                    for j in range(len(res['reference'][i])):
                        f.write('post:\t%s\n' % ' '.join(dm.convert_ids_to_tokens(dm.data[key]['post'][i][j][1:])))
                        f.write("resp:\t%s\n" % " ".join(res['reference'][i][j]))
                        f.write('label:\t%d\tpred:\t%d\n' % (label[i][j], pred[i][j]))
                        f.write('used know:\t%s\n' % used_know[i][j])
                        f.write("gen:\t%s\n\n" % " ".join(res['gen'][i][j]))
                    f.write("------\n\n")
                f.flush()
            logging.info("result output to %s.", filename)
            
        return {key: val for key, val in res.items() if isinstance(val, (str, int, float))}

    def test_process(self):
        logging.info("Test Start.")
        self.net.eval()
        
        if self.args.dataset == 'WizardOfWiki':
            self.test("test_seen")
            self.test("test_unseen")
        elif self.args.dataset == 'HollE':
            self.test("test")
        else:
            raise ValueError
        logging.info("Test Finish.")
    
    def test_dev(self):
        logging.info("Dev Start.")
        self.net.train()
        print('param num', sum(param.numel() for param in self.net.parameters()))
        
        self.net.eval()
        self.test('dev')
        
        logging.info("Dev Finish.")