import logging
import time
import json

import torch
import numpy as np

import cotk.downloader
from .cuda_helper import cuda, Tensor
from .anneal_helper import AnnealHelper, AnnealParameter
from .storage import Storage

class BaseModel():
    def __init__(self, param, net, checkpoint_manager):
        self.param = param
        self.args = args = param.args
        self.net = net

        _ = list(self.net.get_parameters_by_name())
        self.now_batch = 0
        self.now_epoch = 0
        self.checkpoint_manager = checkpoint_manager

        if args.cuda:
            logging.info("initializing cuda")
            Tensor(1)
            logging.info("cuda initialized")

        if args.restore is not None:
            if args.restore.startswith("http"):
                restore = cotk.downloader.load_file_from_url(args.restore)
            else:
                restore = args.restore
            checkpoint = self.checkpoint_manager.restore(restore)
            self.net.load_state_dict(checkpoint, param.volatile.load_exclude_set)

        if args.restore is not None and param.volatile.restoreCallback:
            param.volatile.restoreCallback(self)

        cuda(self.net)

    def zero_grad(self):
        for p in self.net.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def save_checkpoint(self, value=None, filename=None):
        if filename is None:
            filename = "%s_%s" % (self.param.args.name,
                    time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        state = self.net.state_dict()
        self.checkpoint_manager.save(state, filename, value)

    def checkgrad(self):
        logging.info("checkgrad:")
        for name, p in self.net.named_parameters():
            if p.grad is not None and p.grad.abs().sum().tolist() > 0:
                logging.info("\t%s", name)

def get_mean(loss_arr, key):
    return np.mean(list(map(lambda x: x[key].detach().cpu().numpy(), loss_arr)))

def storage_to_list(incoming):
    for i, j in incoming.listitems():
        if "tolist" in dir(j):
            incoming[i] = j.tolist()
        elif isinstance(j, (float, int)):
            incoming[i] = j
    return incoming


