import csv
from collections import Counter
from itertools import chain
import json

import numpy as np
import codecs

from cotk.dataloader import MultiTurnDialog
from cotk.metric import MetricChain, MultiTurnDialogRecorder
from .metric import MyBleuMetric, MyDistinctMetric


class WizardOfWiki(MultiTurnDialog):
    def __init__(self, file_id, vocab_size=20000,
                 #min_vocab_times=10,
                 max_sent_length=40, invalid_vocab_times=0, max_context_length=100):
        self._file_path = file_id
        #self._min_vocab_times = min_vocab_times
        self._vocab_size = vocab_size
        self._max_sent_length = max_sent_length
        self._invalid_vocab_times = invalid_vocab_times
        self._max_context_length = max_context_length
        super(WizardOfWiki, self).__init__(key_name=['train', 'dev', 'test_seen', 'test_unseen'])

    def _load_data(self):
        r'''Loading dataset, invoked by SingleTurnDialog.__init__
        '''
        origin_data = {}
        for key in self.key_name:
            origin_data[key] = {}
            origin_data[key]['post'] = []
            origin_data[key]['resp'] = []
            origin_data[key]['atten'] = []
            origin_data[key]['chosen_topic'] = []
            origin_data[key]['wiki'] = []
            
            with open("%s/%s.json" % (self._file_path, key)) as f:
                origin = json.load(f)
            for dialog in origin:
                origin_data[key]['post'].append([each.split()[:] for each in dialog['posts']])
                origin_data[key]['resp'].append([each.split()[:] for each in dialog['responses']])
                origin_data[key]['chosen_topic'].append(dialog['chosen_topics'])
                origin_data[key]['wiki'].append([[each.split()[:] for each in know] for know in dialog['knowledge']])
                origin_data[key]['atten'].append([e if e != -1 else 0 for e in dialog['labels']])

        def flat(tree):
            res = []
            for i in tree:
                if isinstance(i, list):
                    res.extend(flat(i))
                else:
                    res.append(i)
            return res

        raw_vocab_list = flat(10 * origin_data['train']['post'] + 10 * origin_data['train']['resp'] + origin_data['train']['wiki'])
        # Important: Sort the words preventing the index changes between different runs
        vocab = sorted(Counter(raw_vocab_list).most_common(), key=lambda pair: (-pair[1], pair[0]))
                
        left_vocab = vocab[:self._vocab_size]#list(filter(lambda x: x[1] >= self._min_vocab_times, vocab))
        left_vocab = list(map(lambda x: x[0], left_vocab))
        vocab_list = self.ext_vocab + left_vocab
        valid_vocab_len = len(vocab_list)
        valid_vocab_set = set(vocab_list)

        for key in self.key_name:
            if key == 'train':
                continue
            raw_vocab_list.extend(flat(10 * origin_data[key]['post'] + 10 * origin_data[key]['resp'] + origin_data[key]['wiki']))
        vocab = sorted(Counter(raw_vocab_list).most_common(), key=lambda pair: (-pair[1], pair[0]))
        left_vocab = list(filter(lambda x: x[1] >= self._invalid_vocab_times and x[0] not in valid_vocab_set, vocab))
        left_vocab = list(map(lambda x: x[0], left_vocab))
        vocab_list.extend(left_vocab)

        print("valid vocab list length = %d" % valid_vocab_len)
        print("vocab list length = %d" % len(vocab_list))

        word2id = {w: i for i, w in enumerate(vocab_list)}
        line2id = lambda line: ([self.go_id] + list(map(lambda word: word2id.get(word, self.unk_id), line)) + [self.eos_id])[:self._max_sent_length]
        know2id = lambda line: list(map(lambda word: word2id.get(word, self.unk_id), line))[:self._max_sent_length]

        data = {}
        data_size = {}
        for key in self.key_name:
            data[key] = {}
            data[key]['chosen_topic'] = origin_data[key]['chosen_topic']
            data[key]['post'] = []
            data[key]['resp'] = []
            for p, r in zip(origin_data[key]['post'], origin_data[key]['resp']):
                data[key]['post'].append(list(map(line2id, p)))
                data[key]['resp'].append(list(map(line2id, r)))
            data[key]['wiki'] = []
            for i, sess in enumerate(origin_data[key]['wiki']):
                data[key]['wiki'].append([])
                for turn in sess:
                    data[key]['wiki'][-1].append(list(map(know2id, turn)))
            data[key]['atten'] = origin_data[key]['atten'][:]
            data_size[key] = len(data[key]['post'])

            vocab = flat(origin_data[key]['post'] + origin_data[key]['resp'] + origin_data[key]['wiki'])
            vocab_num = len(vocab)
            oov_num = len(list(filter(lambda word: word not in word2id, vocab)))
            length = []
            for p in origin_data[key]['post']:
                length.extend(list(map(len, p)))
            for r in origin_data[key]['resp']:
                length.extend(list(map(len, r)))
            cut_num = np.sum(np.maximum(np.array(length) - self._max_sent_length + 1, 0))
            print("%s set. OOV rate: %f, max length before cut: %d, cut word rate: %f" % \
                    (key, oov_num / vocab_num, max(length), cut_num / vocab_num))

        return vocab_list, valid_vocab_len, data, data_size

    def get_batch(self, key, index):
        if key not in self.key_name:
            raise ValueError("No set named %s." % key)
        
        res = {}
        batch_size = len(index)

        if key == 'train':
            posts = []
            for i in index:
                post = self.data[key]['post'][i].copy()
                resp = self.data[key]['resp'][i].copy()
                for j in range(len(post)):
                    if j == 0:
                        continue
                    post[j] = (post[j - 1] + resp[j - 1] + post[j])[-self._max_context_length:]
                posts.append(post)
        else:
            posts = [self.data[key]['post'][i].copy() for i in index]

        res["sen_num"] = np.array(list(map(lambda i: len(posts[i]), range(batch_size))))
        res["post_length"] = np.zeros((batch_size, np.max(res["sen_num"])), dtype=int)
        res["resp_length"] = np.zeros((batch_size, np.max(res["sen_num"])), dtype=int)

        for i, j in enumerate(index):
            post_len = list(map(len, posts[i]))
            res["post_length"][i, :len(post_len)] = post_len
            resp_len = list(map(len, self.data[key]['resp'][j]))
            res["resp_length"][i, :len(resp_len)] = resp_len

        res["wiki_num"] = np.zeros((batch_size, np.max(res["sen_num"])), dtype=int)
        for i, j in enumerate(index):
            wiki_num = list(map(len, self.data[key]['wiki'][j]))
            res["wiki_num"][i, :len(wiki_num)] = wiki_num
        res["wiki_length"] = np.zeros((batch_size, np.max(res["sen_num"]), np.max(res["wiki_num"])), dtype=int)
        for i1, j1 in enumerate(index):
            for i2 in range(res["sen_num"][i1]):
                wiki_length = list(map(len, self.data[key]['wiki'][j1][i2]))
                res["wiki_length"][i1, i2, :len(wiki_length)] = wiki_length

        res["post"] = np.zeros((batch_size, np.max(res["sen_num"]), np.max(res["post_length"])), dtype=int)
        res["resp"] = np.zeros((batch_size, np.max(res["sen_num"]), np.max(res["resp_length"])), dtype=int)
        res["atten"] = np.zeros((batch_size, np.max(res["sen_num"])), dtype=int)
        for i1, j1 in enumerate(index):
            for i2 in range(res["sen_num"][i1]):
                post = posts[i1][i2]
                resp = self.data[key]['resp'][j1][i2]
                res["post"][i1, i2, :len(post)] = post
                res["resp"][i1, i2, :len(resp)] = resp
            atten = self.data[key]['atten'][j1]
            res["atten"][i1, :len(atten)] = atten
        res["wiki"] = np.zeros((batch_size, np.max(res["sen_num"]), np.max(res["wiki_num"]), np.max(res["wiki_length"])), dtype=int)
        for i1, j1 in enumerate(index):
            for i2 in range(res["sen_num"][i1]):
                for i3 in range(res["wiki_num"][i1][i2]):
                    wiki = self.data[key]['wiki'][j1][i2][i3]
                    res["wiki"][i1, i2, i3, :len(wiki)] = wiki
        
        res['post_allvocabs'] = res['post'].copy()
        res['resp_allvocabs'] = res['resp'].copy()
        res["post"][res["post"] >= self.valid_vocab_len] = self.unk_id
        res["resp"][res["resp"] >= self.valid_vocab_len] = self.unk_id
        res["wiki"][res["wiki"] >= self.valid_vocab_len] = self.unk_id
        return res
    

    def get_inference_metric(self, multi_turn_gen_key="multi_turn_gen"):
        metric = MetricChain()
        metric.add_metric(MyBleuMetric(self, multi_turn_gen_key=multi_turn_gen_key,
                                                    multi_turn_reference_allvocabs_key="sent_allvocabs",
                                                    turn_len_key="turn_length"))
        metric.add_metric(MultiTurnDialogRecorder(self, multi_turn_gen_key=multi_turn_gen_key,
                                                  multi_turn_reference_allvocabs_key="sent_allvocabs",
                                                  turn_len_key="turn_length"))
        metric.add_metric(MyDistinctMetric(self, multi_turn_gen_key=multi_turn_gen_key, turn_len_key="turn_length"))
        return metric


class HollE(WizardOfWiki):
    def __init__(self, file_id, vocab_size=16000,
                 #min_vocab_times=10,
                 max_sent_length=40, invalid_vocab_times=0, max_context_length=100):
        self._file_path = file_id
        #self._min_vocab_times = min_vocab_times
        self._vocab_size = vocab_size
        self._max_sent_length = max_sent_length
        self._invalid_vocab_times = invalid_vocab_times
        self._max_context_length = max_context_length
        super(WizardOfWiki, self).__init__(key_name=['train', 'dev', 'test'])
        
    def _load_data(self):
        origin_data = {}
        for key in self.key_name:
            origin_data[key] = {}
            origin_data[key]['post'] = []
            origin_data[key]['resp'] = []
            origin_data[key]['wiki'] = []
            origin_data[key]['atten'] = []
            with open("%s/%s.json" % (self._file_path, key)) as f:
                origin = json.load(f)
            for dialog in origin:
                origin_data[key]['post'].append([each.split()[:] for each in dialog['post']])
                origin_data[key]['resp'].append([each.split()[:] if isinstance(each, str) else each[0].split()[:] for each in dialog['resp']])
                origin_data[key]['wiki'].append([[each.split()[:] for each in know] for know in dialog['know']])
                origin_data[key]['atten'].append([each + 1 for each in dialog['atten']])
        
        def flat(tree):
            res = []
            for i in tree:
                if isinstance(i, list):
                    res.extend(flat(i))
                else:
                    res.append(i)
            return res

        raw_vocab_list = flat(10 * origin_data['train']['post'] + 10 * origin_data['train']['resp'] + origin_data['train']['wiki'])
        vocab = sorted(Counter(raw_vocab_list).most_common(), key=lambda pair: (-pair[1], pair[0]))

        #left_vocab = list(filter(lambda x: x[1] >= self._min_vocab_times, vocab))
        left_vocab = vocab[:self._vocab_size]
        left_vocab = list(map(lambda x: x[0], left_vocab))
        vocab_list = self.ext_vocab + left_vocab
        valid_vocab_len = len(vocab_list)
        valid_vocab_set = set(vocab_list)

        for key in self.key_name:
            if key == 'train':
                continue
            raw_vocab_list.extend(flat(10 * origin_data[key]['post'] + 10 * origin_data[key]['resp'] + origin_data[key]['wiki']))
        vocab = sorted(Counter(raw_vocab_list).most_common(), key=lambda pair: (-pair[1], pair[0]))
        left_vocab = list(filter(lambda x: x[1] >= self._invalid_vocab_times and x[0] not in valid_vocab_set, vocab))
        left_vocab = list(map(lambda x: x[0], left_vocab))
        vocab_list.extend(left_vocab)

        print("valid vocab list length = %d" % valid_vocab_len)
        print("vocab list length = %d" % len(vocab_list))

        word2id = {w: i for i, w in enumerate(vocab_list)}
        line2id = lambda line: ([self.go_id] + list(map(lambda word: word2id.get(word, self.unk_id), line)) + [self.eos_id])[:self._max_sent_length]
        know2id = lambda line: (list(map(lambda word: word2id.get(word, self.unk_id), line)))[:self._max_sent_length]

        data = {}
        data_size = {}
        for key in self.key_name:
            data[key] = {}
            data[key]['post'] = []
            data[key]['resp'] = []
            for p, r in zip(origin_data[key]['post'], origin_data[key]['resp']):#, origin_data[key]['topic_wiki']):
                data[key]['post'].append(list(map(line2id, p)))
                data[key]['resp'].append(list(map(line2id, r)))
            data[key]['wiki'] = []
            for i, sess in enumerate(origin_data[key]['wiki']):
                data[key]['wiki'].append([])
                for turn in sess:
                    data[key]['wiki'][-1].append([[]] + list(map(know2id, turn)))
            data[key]['atten'] = origin_data[key]['atten'][:]
            data_size[key] = len(data[key]['post'])

            vocab = flat(origin_data[key]['post'] + origin_data[key]['resp'] + origin_data[key]['wiki'])
            vocab_num = len(vocab)
            oov_num = len(list(filter(lambda word: word not in word2id, vocab)))
            length = []
            for p in origin_data[key]['post']:
                length.extend(list(map(len, p)))
            for r in origin_data[key]['resp']:
                length.extend(list(map(len, r)))
            cut_num = np.sum(np.maximum(np.array(length) - self._max_sent_length + 1, 0))
            print("%s set. OOV rate: %f, max length before cut: %d, cut word rate: %f" % (key, oov_num / vocab_num, max(length), cut_num / vocab_num))

        return vocab_list, valid_vocab_len, data, data_size