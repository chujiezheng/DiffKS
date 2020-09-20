# coding:utf-8
import logging
import json
import os

from cotk.wordvector import WordVector, Glove
from myCoTK.dataloader import WizardOfWiki, HollE
from utils import debug, try_cache, cuda_init, Storage
from seq2seq import Seq2seq

def main(args):
    logging.basicConfig(filename=0,
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s',
        datefmt='%H:%M:%S')
    
    if args.debug:
        debug()
    logging.info(json.dumps(args, indent=2))
    
    cuda_init(args.cuda_num, args.cuda)
    
    volatile = Storage()
    volatile.load_exclude_set = args.load_exclude_set
    volatile.restoreCallback = args.restoreCallback
    
    if args.dataset == 'WizardOfWiki':
        data_class = WizardOfWiki
    elif args.dataset == 'HollE':
        data_class = HollE
    else:
        raise ValueError
    wordvec_class = WordVector.load_class(args.wvclass)
    if wordvec_class is None:
        wordvec_class = Glove
    
    if not os.path.exists(args.cache_dir):
        os.mkdir(args.cache_dir)
    args.cache_dir = os.path.join(args.cache_dir, args.dataset)
    
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    args.out_dir = os.path.join(args.out_dir, args.dataset)
    
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    if args.dataset not in args.model_dir:
        args.model_dir = os.path.join(args.model_dir, args.dataset)
    
    if args.cache:
        dm = try_cache(data_class, (args.datapath,), args.cache_dir)
        volatile.wordvec = try_cache(
            lambda wv, ez, vl: wordvec_class(wv).load_matrix(ez, vl),
            (args.wvpath, args.embedding_size, dm.vocab_list),
            args.cache_dir, wordvec_class.__name__)
    else:
        dm = data_class(args.datapath)
        wv = wordvec_class(args.wvpath)
        volatile.wordvec = wv.load_matrix(args.embedding_size, dm.vocab_list)
    
    volatile.dm = dm
    
    param = Storage()
    param.args = args
    param.volatile = volatile
    
    model = Seq2seq(param)
    if args.mode == "train":
        model.train_process()
    elif args.mode == "test":
        model.test_process()
    elif args.mode == 'dev':
        model.test_dev()
    else:
        raise ValueError("Unknown mode")