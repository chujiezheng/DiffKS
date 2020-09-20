import random
import os
import multiprocessing
from multiprocessing import Pool
import numpy as np
import tqdm

from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from cotk.metric import MetricBase
from cotk.metric.bleu import _replace_unk
from cotk._utils import hooks

class MyBleuMetric(MetricBase): # for multi-turn
    _name = 'MyBleuMetric'
    _version = 1
    
    @hooks.hook_metric
    def __init__(self, dataloader, ignore_smoothing_error=False,
                    multi_turn_reference_allvocabs_key="reference_allvocabs",
                    multi_turn_gen_key="multi_turn_gen",
                    turn_len_key="turn_length"
                 ):
        super().__init__(self._name, self._version)
        self.dataloader = dataloader
        self.ignore_smoothing_error = ignore_smoothing_error
        self.multi_turn_reference_allvocabs_key = multi_turn_reference_allvocabs_key
        self.turn_len_key = turn_len_key
        self.multi_turn_gen_key = multi_turn_gen_key
        self.refs = []
        self.hyps = []

    def forward(self, data):
        super().forward(data)
        reference_allvocabs = data[self.multi_turn_reference_allvocabs_key]
        length = data[self.turn_len_key]
        gen = data[self.multi_turn_gen_key]

        if not isinstance(reference_allvocabs, (np.ndarray, list)):
            raise TypeError("Unknown type for reference_allvocabs.")
        if not isinstance(length, (np.ndarray, list)):
            raise TypeError("Unknown type for length")
        if not isinstance(gen, (np.ndarray, list)):
            raise TypeError("Unknown type for gen")

        if len(length) != len(reference_allvocabs) or len(length) != len(gen):
            raise ValueError("Batch num is not matched.")

        for i, turn_length in enumerate(length):
            gen_session = gen[i]
            ref_session = reference_allvocabs[i]
            for j in range(turn_length):
                self.hyps.append(list(self.dataloader.trim(gen_session[j])))
                self.refs.append([list(self.dataloader.trim(ref_session[j])[1:])])

    @hooks.hook_metric_close
    def close(self):
        result = super().close()
        if (not self.hyps) or (not self.refs):
            raise RuntimeError("The metric has not been forwarded data correctly.")
        self.hyps = _replace_unk(self.hyps, self.dataloader.unk_id)

        self._hash_relevant_data(self.refs)

        for i in range(1, 5):
            try:
                weights = [1. / i] * i + [0.] * (4 - i)
                result.update(
                    {"bleu-%d" % i: 100 * corpus_bleu(self.refs, self.hyps, weights,
                                                      smoothing_function=SmoothingFunction().method3)})

            except ZeroDivisionError as _:
                if not self.ignore_smoothing_error:
                    raise ZeroDivisionError("Bleu smoothing divided by zero. This is a known bug of corpus_bleu, \
                    usually caused when there is only one sample and the sample length is 1.")
                result.update({"bleu-%d" % i: 0, "bleu hashvalue": self._hashvalue()})

        return result
    

class MyRougeMetric(MetricBase): # for multi-turn
    _name = 'MyRougeMetric'
    _version = 1
    
    @hooks.hook_metric
    def __init__(self, dataloader, ignore_smoothing_error=False,
                    multi_turn_reference_allvocabs_key="reference_allvocabs",
                    multi_turn_gen_key="multi_turn_gen",
                    turn_len_key="turn_length"
                 ):
        super().__init__(self._name, self._version)
        self.dataloader = dataloader
        self.ignore_smoothing_error = ignore_smoothing_error
        self.multi_turn_reference_allvocabs_key = multi_turn_reference_allvocabs_key
        self.turn_len_key = turn_len_key
        self.multi_turn_gen_key = multi_turn_gen_key
        self.refs = []
        self.hyps = []

    def forward(self, data):
        super().forward(data)
        reference_allvocabs = data[self.multi_turn_reference_allvocabs_key]
        length = data[self.turn_len_key]
        gen = data[self.multi_turn_gen_key]

        if not isinstance(reference_allvocabs, (np.ndarray, list)):
            raise TypeError("Unknown type for reference_allvocabs.")
        if not isinstance(length, (np.ndarray, list)):
            raise TypeError("Unknown type for length")
        if not isinstance(gen, (np.ndarray, list)):
            raise TypeError("Unknown type for gen")

        if len(length) != len(reference_allvocabs) or len(length) != len(gen):
            raise ValueError("Batch num is not matched.")

        for i, turn_length in enumerate(length):
            gen_session = gen[i]
            ref_session = reference_allvocabs[i]
            for j in range(turn_length):
                self.hyps.append(' '.join(list(map(str, self.dataloader.trim(gen_session[j])))))
                self.refs.append(' '.join(list(map(str, self.dataloader.trim(ref_session[j])[1:]))))
                if len(self.hyps[-1]) == 0:
                    self.hyps[-1] = '<unk>'
                if len(self.refs[-1]) == 0:
                    self.refs[-1] = '<unk>'

    @hooks.hook_metric_close
    def close(self):
        result = super().close()
        if (not self.hyps) or (not self.refs):
            raise RuntimeError("The metric has not been forwarded data correctly.")

        self._hash_relevant_data(self.refs)

        rouge = Rouge()
        scores = rouge.get_scores(self.hyps, self.refs, avg=True)
        keys = list(scores.keys())
        res = {key: [] for key in keys}
        for key in keys:
            #for k in ['f', 'p', 'r']:
                #res[f'{key}-{k}'] = scores[key][k] * 100
            res[f'{key}'] = scores[key]['r'] * 100
        result.update(res)
        return result
    
    
class MyDistinctMetric(MetricBase): # for multi-turn
    _name = 'MyDistinctMetric'
    _version = 1

    def __init__(self, dataloader, multi_turn_gen_key="multi_turn_gen", turn_len_key="turn_length"):
        super().__init__(self._name, self._version)
        self.dataloader = dataloader
        self.turn_len_key = turn_len_key
        self.multi_turn_gen_key = multi_turn_gen_key
        self.hyps = []

    def calc_distinct_k(self, k, src=None):
        def h(vs, l):
            ret = 0
            for v in l:
                ret = ret * vs + v
            return ret

        if src is None:
            src = self.hyps
        d = {}
        vs = self.dataloader.vocab_size
        tot = 0
        for sen in src:
            for i in range(0, len(sen)-k):
                key = h(vs, sen[i:i+k])
                if key not in d:
                    d[key] = 0
                d[key] += 1
                tot += 1
        return len(d) / tot

    def forward(self, data):
        length = data[self.turn_len_key]
        gen = data[self.multi_turn_gen_key]
        if len(length) != len(gen):
            raise ValueError("Batch num is not matched.")

        for i, turn_length in enumerate(length):
            gen_session = gen[i]
            for j in range(turn_length):
                self.hyps.append(self.dataloader.trim(gen_session[j]))

    def close(self):
        ret = {}
        for k in range(1, 5):
            ret["dist-%d" % k] = 100 * self.calc_distinct_k(k)
        return ret
