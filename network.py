# coding:utf-8
import logging

import torch
from torch import nn
import numpy as np

from utils import zeros, ones, LongTensor, maskedSoftmax,\
            BaseNetwork, MyGRU, Storage, gumbel_max, flattenSequence

# pylint: disable=W0221
class Network(BaseNetwork):
    def __init__(self, param):
        super().__init__(param)

        self.embLayer = EmbeddingLayer(param)
        self.postEncoder = PostEncoder(param)
        self.wikiEncoder = WikiEncoder(param)
        self.connectLayer = ConnectLayer(param)
        self.genNetwork = GenNetwork(param)

    def forward(self, incoming):
        incoming.result = Storage()
        incoming.result.word_loss = None
        incoming.state = state = Storage()
        incoming.statistic = statistic = Storage()
        statistic.batch_num = incoming.data.post.shape[0]
        statistic.sen_num = 0
        statistic.sen_loss = []
        
        state.last = incoming.data.post.shape[1]
    
        for i in range(state.last):
            state.num = i
            self.embLayer.forward(incoming)
            self.postEncoder.forward(incoming)
            self.wikiEncoder.forward(incoming)
            if not self.args.disentangle:
                self.connectLayer.forward(incoming)
            else:
                self.connectLayer.forward_disentangle(incoming)
            self.genNetwork.forward(incoming)
        
        incoming.result.loss = incoming.result.word_loss + incoming.result.atten_loss

        if torch.isnan(incoming.result.loss).detach().cpu().numpy() > 0:
            logging.info("Nan detected")
            logging.info(incoming.result)
            raise FloatingPointError("Nan detected")

    def detail_forward(self, incoming):
        incoming.acc = Storage()
        incoming.acc.pred = []
        incoming.acc.label = []
        incoming.acc.prob = []
        
        incoming.result = Storage()
        incoming.state = state = Storage()
        incoming.statistic = statistic = Storage()
        statistic.batch_num = incoming.data.post.shape[0]
        statistic.sen_loss = []
        statistic.sen_num = 0
        
        state.last = incoming.data.post.shape[1]
        # post, resp: [batch_size, turn_length, sent_length]
        
        def pad_post(posts):
            '''
            :param posts: list, [batch, turn_length, sent_length]
            '''
            post_length = np.array(list(map(len, posts)), dtype=int)
            post = np.zeros((len(post_length), np.max(post_length)), dtype=int)
            for j in range(len(post_length)):
                post[j, :len(posts[j])] = posts[j]
            return post, post_length

        dm = self.param.volatile.dm
        ori_post = incoming.data.post.tolist() # [batch_size, turn_length, sent_length]
        ori_post = [[(dm.trim(post) + [dm.eos_id])[:dm._max_sent_length] for post in posts] for posts in ori_post]
        new_post = [each[0] for each in ori_post]
        for i in range(state.last):
            state.num = i
            incoming.data.post, incoming.data.post_length = pad_post(new_post)

            self.embLayer.detail_forward(incoming)
            self.postEncoder.detail_forward(incoming)
            self.wikiEncoder.forward(incoming)
            if not self.args.disentangle:
                self.connectLayer.detail_forward(incoming)
            else:
                self.connectLayer.detail_forward_disentangle(incoming)
            self.genNetwork.detail_forward(incoming)

            if i < state.last - 1:
                gen_resp = incoming.state.w_o_all[-1].transpose(0, 1).cpu().tolist() # [batch, sent_length]
                new_post = []
                for j, gr in enumerate(gen_resp):
                    new_post.append((ori_post[j][i] + ([dm.go_id] + dm.trim(gr) + [dm.eos_id])[:dm._max_sent_length] + ori_post[j][i + 1])[-dm._max_context_length:])


class EmbeddingLayer(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.args = args = param.args
        self.param = param
        volatile = param.volatile

        self.embLayer = nn.Embedding(volatile.dm.vocab_size, args.embedding_size)
        self.embLayer.weight = nn.Parameter(torch.Tensor(volatile.wordvec))
        self.drop = nn.Dropout(args.droprate)

    def forward(self, incoming):
        '''
        inp: data
        output: post
        '''
        i = incoming.state.num
        incoming.post = Storage()
        incoming.post.embedding = self.drop(self.embLayer(LongTensor(incoming.data.post[:, i])))
        incoming.resp = Storage()
        incoming.resp.embedding = self.drop(self.embLayer(incoming.data.resp[:, i]))
        incoming.wiki = Storage()
        incoming.wiki.embedding = self.drop(self.embLayer(incoming.data.wiki[:, i]))
        incoming.resp.embLayer = self.embLayer
        
    def detail_forward(self, incoming):
        i = incoming.state.num
        incoming.post = Storage()
        incoming.post.embedding = self.embLayer(LongTensor(incoming.data.post))
        incoming.resp = Storage()
        incoming.wiki = Storage()
        incoming.wiki.embedding = self.embLayer(incoming.data.wiki[:, i])
        incoming.resp.embLayer = self.embLayer
    

class PostEncoder(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.args = args = param.args
        self.param = param
        
        self.postGRU = MyGRU(args.embedding_size, args.eh_size, bidirectional=True)
    
    def forward(self, incoming):
        incoming.hidden = hidden = Storage()
        i = incoming.state.num
        # incoming.post.embedding : batch * sen_num * length * vec_dim
        # post_length : batch * sen_num
        raw_post = incoming.post.embedding
        raw_post_length = LongTensor(incoming.data.post_length[:, i])
        incoming.state.valid_sen = torch.sum(torch.nonzero(raw_post_length), 1)
        raw_reverse = torch.cumsum(torch.gt(raw_post_length, 0), 0) - 1
        incoming.state.reverse_valid_sen = raw_reverse * torch.ge(raw_reverse, 0).to(torch.long)
        valid_sen = incoming.state.valid_sen
        incoming.state.valid_num = valid_sen.shape[0]
        
        post = torch.index_select(raw_post, 0, valid_sen).transpose(0, 1)  # [length, valid_num, vec_dim]
        post_length = torch.index_select(raw_post_length, 0, valid_sen).cpu().numpy()
        
        hidden.h, hidden.h_n = self.postGRU.forward(post, post_length, need_h=True)
        hidden.length = post_length

    def detail_forward(self, incoming):
        incoming.hidden = hidden = Storage()
        # incoming.post.embedding : batch * sen_num * length * vec_dim
        # post_length : batch * sen_num
        raw_post = incoming.post.embedding
        raw_post_length = LongTensor(incoming.data.post_length)
        incoming.state.valid_sen = torch.sum(torch.nonzero(raw_post_length), 1)
        raw_reverse = torch.cumsum(torch.gt(raw_post_length, 0), 0) - 1
        incoming.state.reverse_valid_sen = raw_reverse * torch.ge(raw_reverse, 0).to(torch.long)
        valid_sen = incoming.state.valid_sen
        incoming.state.valid_num = valid_sen.shape[0]
    
        post = torch.index_select(raw_post, 0, valid_sen).transpose(0, 1)  # [length, valid_num, vec_dim]
        post_length = torch.index_select(raw_post_length, 0, valid_sen).cpu().numpy() # [valid_num]
    
        hidden.h, hidden.h_n = self.postGRU.forward(post, post_length, need_h=True)
        hidden.length = post_length
        

class WikiEncoder(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.args = args = param.args
        self.param = param
        
        self.sentenceGRU = MyGRU(args.embedding_size, args.eh_size, bidirectional=True)
        
    def forward(self, incoming):
        i = incoming.state.num
        batch = incoming.wiki.embedding.shape[0]
        
        incoming.wiki_hidden = wiki_hidden = Storage()
        incoming.wiki_sen = incoming.data.wiki[:, i]  # [batch, wiki_sen_num, wiki_sen_len]
        wiki_length = incoming.data.wiki_length[:, i].reshape(-1)  # (batch * wiki_sen_num)
        embed = incoming.wiki.embedding.reshape((-1, incoming.wiki.embedding.shape[2], self.args.embedding_size))
        # (batch * wiki_sen_num) * wiki_sen_len * embedding_size
        embed = embed.transpose(0, 1)  # wiki_sen_len * (batch * wiki_sen_num) * embedding_size
        
        wiki_hidden.h1, wiki_hidden.h_n1 = self.sentenceGRU.forward(embed, wiki_length, need_h=True)
        # [wiki_sen_len, batch * wiki_sen_num, 2 * eh_size], [batch * wiki_sen_num, 2 * eh_size]
        wiki_hidden.h1 = wiki_hidden.h1.reshape((wiki_hidden.h1.shape[0], batch, -1, wiki_hidden.h1.shape[-1]))
        # [wiki_sen_len, batch,  wiki_sen_num, 2 * eh_size]
        wiki_hidden.h_n1 = wiki_hidden.h_n1.reshape((batch, -1, 2 * self.args.eh_size)).transpose(0, 1)
        # [wiki_sen_num, batch, 2 * eh_size]
    

class ConnectLayer(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.args = args = param.args
        self.param = param
        self.initLinearLayer = nn.Linear(args.eh_size * 4, args.dh_size)

        self.wiki_atten = nn.Softmax(dim=0)
        self.atten_lossCE = nn.CrossEntropyLoss(ignore_index=0)

        self.last_wiki = None
        self.hist_len = args.hist_len
        self.hist_weights = args.hist_weights
        
        self.compareGRU = MyGRU(2 * args.eh_size, args.eh_size, bidirectional=True)
        
        self.tilde_linear = nn.Linear(4 * args.eh_size, 2 * args.eh_size)
        self.attn_query = nn.Linear(2 * args.eh_size, 2 * args.eh_size, bias=False)
        self.attn_key = nn.Linear(4 * args.eh_size, 2 * args.eh_size, bias=False)
        self.attn_v = nn.Linear(2 * args.eh_size, 1, bias=False)

    def forward(self, incoming):
        incoming.conn = conn = Storage()
        index = incoming.state.num
        valid_sen = incoming.state.valid_sen
        valid_wiki_h_n1 = torch.index_select(incoming.wiki_hidden.h_n1, 1, valid_sen) # [wiki_sen_num, valid_num, 2 * eh_size]
        valid_wiki_sen = torch.index_select(incoming.wiki_sen, 0, valid_sen) # [valid_num, wiki_sen_num, wiki_sen_len]
        valid_wiki_h1 = torch.index_select(incoming.wiki_hidden.h1, 1, valid_sen) # [wiki_sen_len, valid_num, wiki_sen_num, 2 * eh_size]
        atten_label = torch.index_select(incoming.data.atten[:, index], 0, valid_sen)  # valid_num
        valid_wiki_num = torch.index_select(LongTensor(incoming.data.wiki_num[:, index]), 0, valid_sen)  # valid_num
        
        if index == 0:
            tilde_wiki = zeros(1, 1, 2 * self.args.eh_size) * ones(valid_wiki_h_n1.shape[0], valid_wiki_h_n1.shape[1], 1)
        else:
            wiki_hidden = incoming.wiki_hidden
            wiki_num = incoming.data.wiki_num[:, index] # [batch], numpy array
            wiki_hidden.h2, wiki_hidden.h_n2 = self.compareGRU.forward(wiki_hidden.h_n1, wiki_num, need_h=True)
            valid_wiki_h2 = torch.index_select(wiki_hidden.h2, 1, valid_sen) # wiki_len * valid_num * (2 * eh_size)
            
            tilde_wiki_list = []
            for i in range(self.last_wiki.size(-1)):
                last_wiki = torch.index_select(self.last_wiki[:, :, i], 0, valid_sen).unsqueeze(0)  # 1, valid_num, (2 * eh)
                tilde_wiki = torch.tanh(self.tilde_linear(torch.cat([last_wiki - valid_wiki_h2, last_wiki * valid_wiki_h2], dim=-1)))
                tilde_wiki_list.append(tilde_wiki.unsqueeze(-1) * self.hist_weights[i])
            tilde_wiki = torch.cat(tilde_wiki_list, dim=-1).sum(dim=-1)
        
        query = self.attn_query(incoming.hidden.h_n)  # [valid_num, hidden]
        key = self.attn_key(torch.cat([valid_wiki_h_n1[:tilde_wiki.shape[0]], tilde_wiki], dim=-1))  # [wiki_sen_num, valid_num, hidden]
        atten_sum = self.attn_v(torch.tanh(query + key)).squeeze(-1)  # [wiki_sen_num, valid_num]
        beta = atten_sum.t() # [valid_num, wiki_len]
        
        mask = torch.arange(beta.shape[1], device=beta.device).long().expand(beta.shape[0], beta.shape[1]).transpose(0, 1)  # [wiki_sen_num, valid_num]
        expand_wiki_num = valid_wiki_num.unsqueeze(0).expand_as(mask)  # [wiki_sen_num, valid_num]
        reverse_mask = (expand_wiki_num <= mask).float()  # [wiki_sen_num, valid_num]
        
        if index == 0:
            incoming.result.atten_loss = self.atten_lossCE(beta, atten_label)
        else:
            incoming.result.atten_loss += self.atten_lossCE(beta, atten_label)
        
        golden_alpha = zeros(beta.shape).scatter_(1, atten_label.unsqueeze(1), 1)
        golden_alpha = torch.t(golden_alpha).unsqueeze(2)
        wiki_cv = torch.sum(valid_wiki_h_n1[:golden_alpha.shape[0]] * golden_alpha, dim=0)  # valid_num * (2 * eh_size)
        conn.wiki_cv = wiki_cv
        conn.init_h = self.initLinearLayer(torch.cat([incoming.hidden.h_n, wiki_cv], 1))
        
        reverse_valid_sen = incoming.state.reverse_valid_sen
        if index == 0:
            self.last_wiki = torch.index_select(wiki_cv, 0, reverse_valid_sen).unsqueeze(-1)  # [batch, 2 * eh_size]
        else:
            self.last_wiki = torch.cat([torch.index_select(wiki_cv, 0, reverse_valid_sen).unsqueeze(-1),
                                        self.last_wiki[:, :, :self.hist_len-1]], dim=-1)

        atten_indices = atten_label.unsqueeze(1) # valid_num * 1
        atten_indices = torch.cat([torch.arange(atten_indices.shape[0]).unsqueeze(1), atten_indices.cpu()], 1) # valid_num * 2
        valid_wiki_h1 = torch.transpose(valid_wiki_h1, 0, 1) # valid_num * wiki_sen_len * wiki_len * (2 * eh_size)
        valid_wiki_h1 = torch.transpose(valid_wiki_h1, 1, 2) # valid_num * wiki_len * wiki_sen_len * (2 * eh_size)
        conn.selected_wiki_h = valid_wiki_h1[atten_indices.chunk(2, 1)].squeeze(1)
        conn.selected_wiki_sen = valid_wiki_sen[atten_indices.chunk(2, 1)].squeeze(1)


    def detail_forward(self, incoming):
        incoming.conn = conn = Storage()
        index = incoming.state.num
        valid_sen = incoming.state.valid_sen
        valid_wiki_h_n1 = torch.index_select(incoming.wiki_hidden.h_n1, 1, valid_sen) # [wiki_sen_num, valid_num, 2 * eh_size]
        valid_wiki_sen = torch.index_select(incoming.wiki_sen, 0, valid_sen) # [valid_num, wiki_sen_num, wiki_sen_len]
        valid_wiki_h1 = torch.index_select(incoming.wiki_hidden.h1, 1, valid_sen) # [wiki_sen_len, valid_num, wiki_sen_num, 2 * eh_size]
        atten_label = torch.index_select(incoming.data.atten[:, index], 0, valid_sen)  # valid_num
        valid_wiki_num = torch.index_select(LongTensor(incoming.data.wiki_num[:, index]), 0, valid_sen)  # valid_num
        
        if index == 0:
            tilde_wiki = zeros(1, 1, 2 * self.args.eh_size) * ones(valid_wiki_h_n1.shape[0], valid_wiki_h_n1.shape[1], 1)
        else:
            wiki_hidden = incoming.wiki_hidden
            wiki_num = incoming.data.wiki_num[:, index] # [batch], numpy array
            wiki_hidden.h2, wiki_hidden.h_n2 = self.compareGRU.forward(wiki_hidden.h_n1, wiki_num, need_h=True)
            valid_wiki_h2 = torch.index_select(wiki_hidden.h2, 1, valid_sen) # wiki_len * valid_num * (2 * eh_size)
            
            tilde_wiki_list = []
            for i in range(self.last_wiki.size(-1)):
                last_wiki = torch.index_select(self.last_wiki[:, :, i], 0, valid_sen).unsqueeze(0)  # 1, valid_num, (2 * eh)
                tilde_wiki = torch.tanh(self.tilde_linear(torch.cat([last_wiki - valid_wiki_h2, last_wiki * valid_wiki_h2], dim=-1)))
                tilde_wiki_list.append(tilde_wiki.unsqueeze(-1) * self.hist_weights[i])
            tilde_wiki = torch.cat(tilde_wiki_list, dim=-1).sum(dim=-1)
        
        query = self.attn_query(incoming.hidden.h_n)  # [valid_num, hidden]
        key = self.attn_key(torch.cat([valid_wiki_h_n1[:tilde_wiki.shape[0]], tilde_wiki], dim=-1))  # [wiki_sen_num, valid_num, hidden]
        atten_sum = self.attn_v(torch.tanh(query + key)).squeeze(-1)  # [wiki_sen_num, valid_num]
        beta = atten_sum.t() # [valid_num, wiki_len]

        mask = torch.arange(beta.shape[1], device=beta.device).long().expand(beta.shape[0], beta.shape[1]).transpose(0, 1)  # [wiki_sen_num, valid_num]
        expand_wiki_num = valid_wiki_num.unsqueeze(0).expand_as(mask)  # [wiki_sen_num, valid_num]
        reverse_mask = (expand_wiki_num <= mask).float()  # [wiki_sen_num, valid_num]

        if index == 0:
            incoming.result.atten_loss = self.atten_lossCE(beta, atten_label)
        else:
            incoming.result.atten_loss += self.atten_lossCE(beta, atten_label)

        beta = torch.t(beta) - 1e10 * reverse_mask
        alpha = self.wiki_atten(beta)  # wiki_len * valid_num
        incoming.acc.prob.append(torch.index_select(alpha.t(), 0, incoming.state.reverse_valid_sen).cpu().tolist())
        atten_indices = torch.argmax(alpha, 0)
        alpha = zeros(beta.t().shape).scatter_(1, atten_indices.unsqueeze(1), 1)
        alpha = torch.t(alpha)
        wiki_cv = torch.sum(valid_wiki_h_n1[:alpha.shape[0]] * alpha.unsqueeze(2), dim=0)  # valid_num * (2 * eh_size)
        conn.wiki_cv = wiki_cv
        conn.init_h = self.initLinearLayer(torch.cat([incoming.hidden.h_n, wiki_cv], 1))

        reverse_valid_sen = incoming.state.reverse_valid_sen
        if index == 0:
            self.last_wiki = torch.index_select(wiki_cv, 0, reverse_valid_sen).unsqueeze(-1)  # [batch, 2 * eh_size]
        else:
            self.last_wiki = torch.cat([torch.index_select(wiki_cv, 0, reverse_valid_sen).unsqueeze(-1),
                                        self.last_wiki[:, :, :self.hist_len-1]], dim=-1)

        incoming.acc.label.append(torch.index_select(atten_label, 0, reverse_valid_sen).cpu().tolist())
        incoming.acc.pred.append(torch.index_select(atten_indices, 0, reverse_valid_sen).cpu().tolist())

        atten_indices = atten_indices.unsqueeze(1)
        atten_indices = torch.cat([torch.arange(atten_indices.shape[0]).unsqueeze(1), atten_indices.cpu()], 1) # valid_num * 2
        valid_wiki_h1 = torch.transpose(valid_wiki_h1, 0, 1) # valid_num * wiki_sen_len * wiki_len * (2 * eh_size)
        valid_wiki_h1 = torch.transpose(valid_wiki_h1, 1, 2) # valid_num * wiki_len * wiki_sen_len * (2 * eh_size)
        conn.selected_wiki_h = valid_wiki_h1[atten_indices.chunk(2, 1)].squeeze(1) # valid_num * wiki_sen_len * (2 * eh_size)
        conn.selected_wiki_sen = valid_wiki_sen[atten_indices.chunk(2, 1)].squeeze(1) # valid_num * wiki_sen_len
        
    def forward_disentangle(self, incoming):
        incoming.conn = conn = Storage()
        index = incoming.state.num
        valid_sen = incoming.state.valid_sen
        valid_wiki_h_n1 = torch.index_select(incoming.wiki_hidden.h_n1, 1, valid_sen) # [wiki_sen_num, valid_num, 2 * eh_size]
        valid_wiki_sen = torch.index_select(incoming.wiki_sen, 0, valid_sen) # [valid_num, wiki_sen_num, wiki_sen_len]
        valid_wiki_h1 = torch.index_select(incoming.wiki_hidden.h1, 1, valid_sen) # [wiki_sen_len, valid_num, wiki_sen_num, 2 * eh_size]
        atten_label = torch.index_select(incoming.data.atten[:, index], 0, valid_sen)  # valid_num
        valid_wiki_num = torch.index_select(LongTensor(incoming.data.wiki_num[:, index]), 0, valid_sen)  # valid_num
        
        reverse_valid_sen = incoming.state.reverse_valid_sen
        self.beta = torch.sum(valid_wiki_h_n1 * incoming.hidden.h_n, dim = 2) # wiki_len * valid_num
        self.beta = torch.t(self.beta) # [valid_num, wiki_len]
        
        mask = torch.arange(self.beta.shape[1], device=self.beta.device).long().expand(self.beta.shape[0], self.beta.shape[1]).transpose(0, 1)  # [wiki_sen_num, valid_num]
        expand_wiki_num = valid_wiki_num.unsqueeze(0).expand_as(mask)  # [wiki_sen_num, valid_num]
        reverse_mask = (expand_wiki_num <= mask).float()  # [wiki_sen_num, valid_num]

        if index > 0:
            wiki_hidden = incoming.wiki_hidden
            wiki_num = incoming.data.wiki_num[:, index] # [batch], numpy array
            wiki_hidden.h2, wiki_hidden.h_n2 = self.compareGRU.forward(wiki_hidden.h_n1, wiki_num, need_h=True)
            valid_wiki_h2 = torch.index_select(wiki_hidden.h2, 1, valid_sen) # wiki_len * valid_num * (2 * eh_size)
            
            tilde_wiki_list = []
            for i in range(self.last_wiki.size(-1)):
                last_wiki = torch.index_select(self.last_wiki[:, :, i], 0, valid_sen).unsqueeze(0)  # 1, valid_num, (2 * eh)
                tilde_wiki = torch.tanh(self.tilde_linear(torch.cat([last_wiki - valid_wiki_h2, last_wiki * valid_wiki_h2], dim=-1)))
                tilde_wiki_list.append(tilde_wiki.unsqueeze(-1) * self.hist_weights[i])
            tilde_wiki = torch.cat(tilde_wiki_list, dim=-1).sum(dim=-1)
            # wiki_len * valid_num * (2 * eh_size)
    
            query = self.attn_query(tilde_wiki)  # [1, valid_num, hidden]
            key = self.attn_key(torch.cat([valid_wiki_h2, tilde_wiki], dim=-1))  # [wiki_sen_num, valid_num, hidden]
            atten_sum = self.attn_v(torch.tanh(query + key)).squeeze(-1)  # [wiki_sen_num, valid_num]
            
            self.beta = self.beta[:, :atten_sum.shape[0]] + torch.t(atten_sum)
        
        
        if index == 0:
            incoming.result.atten_loss = self.atten_lossCE(self.beta, #self.alpha.t().log(),
                                                           atten_label)
        else:
            incoming.result.atten_loss += self.atten_lossCE(self.beta, #self.alpha.t().log(),
                                                            atten_label)

        golden_alpha = zeros(self.beta.shape).scatter_(1, atten_label.unsqueeze(1), 1)
        golden_alpha = torch.t(golden_alpha).unsqueeze(2)
        wiki_cv = torch.sum(valid_wiki_h_n1[:golden_alpha.shape[0]] * golden_alpha, dim=0)  # valid_num * (2 * eh_size)
        conn.wiki_cv = wiki_cv
        conn.init_h = self.initLinearLayer(torch.cat([incoming.hidden.h_n, wiki_cv], 1))

        if index == 0:
            self.last_wiki = torch.index_select(wiki_cv, 0, reverse_valid_sen).unsqueeze(-1)  # [batch, 2 * eh_size]
        else:
            self.last_wiki = torch.cat([torch.index_select(wiki_cv, 0, reverse_valid_sen).unsqueeze(-1),
                                        self.last_wiki[:, :, :self.hist_len-1]], dim=-1)

        atten_indices = atten_label.unsqueeze(1) # valid_num * 1
        atten_indices = torch.cat([torch.arange(atten_indices.shape[0]).unsqueeze(1), atten_indices.cpu()], 1) # valid_num * 2
        valid_wiki_h1 = torch.transpose(valid_wiki_h1, 0, 1) # valid_num * wiki_sen_len * wiki_len * (2 * eh_size)
        valid_wiki_h1 = torch.transpose(valid_wiki_h1, 1, 2) # valid_num * wiki_len * wiki_sen_len * (2 * eh_size)
        conn.selected_wiki_h = valid_wiki_h1[atten_indices.chunk(2, 1)].squeeze(1)
        conn.selected_wiki_sen = valid_wiki_sen[atten_indices.chunk(2, 1)].squeeze(1)


    def detail_forward_disentangle(self, incoming):
        incoming.conn = conn = Storage()
        index = incoming.state.num
        valid_sen = incoming.state.valid_sen
        valid_wiki_h_n1 = torch.index_select(incoming.wiki_hidden.h_n1, 1, valid_sen) # [wiki_sen_num, valid_num, 2 * eh_size]
        valid_wiki_sen = torch.index_select(incoming.wiki_sen, 0, valid_sen)
        valid_wiki_h1 = torch.index_select(incoming.wiki_hidden.h1, 1, valid_sen)
        atten_label = torch.index_select(incoming.data.atten[:, index], 0, valid_sen)  # valid_num
        valid_wiki_num = torch.index_select(LongTensor(incoming.data.wiki_num[:, index]), 0, valid_sen)  # valid_num
        
        reverse_valid_sen = incoming.state.reverse_valid_sen
        self.beta = torch.sum(valid_wiki_h_n1 * incoming.hidden.h_n, dim = 2)
        self.beta = torch.t(self.beta) # [valid_num, wiki_len]
        
        mask = torch.arange(self.beta.shape[1], device=self.beta.device).long().expand(self.beta.shape[0], self.beta.shape[1]).transpose(0, 1)  # [wiki_sen_num, valid_num]
        expand_wiki_num = valid_wiki_num.unsqueeze(0).expand_as(mask)  # [wiki_sen_num, valid_num]
        reverse_mask = (expand_wiki_num <= mask).float()  # [wiki_sen_num, valid_num]

        if index > 0:
            wiki_hidden = incoming.wiki_hidden
            wiki_num = incoming.data.wiki_num[:, index] # [batch], numpy array
            wiki_hidden.h2, wiki_hidden.h_n2 = self.compareGRU.forward(wiki_hidden.h_n1, wiki_num, need_h=True)
            valid_wiki_h2 = torch.index_select(wiki_hidden.h2, 1, valid_sen) # wiki_len * valid_num * (2 * eh_size)
            
            tilde_wiki_list = []
            for i in range(self.last_wiki.size(-1)):
                last_wiki = torch.index_select(self.last_wiki[:, :, i], 0, valid_sen).unsqueeze(0)  # 1, valid_num, (2 * eh)
                tilde_wiki = torch.tanh(self.tilde_linear(torch.cat([last_wiki - valid_wiki_h2, last_wiki * valid_wiki_h2], dim=-1)))
                tilde_wiki_list.append(tilde_wiki.unsqueeze(-1) * self.hist_weights[i])
            tilde_wiki = torch.cat(tilde_wiki_list, dim=-1).sum(dim=-1) # wiki_len * valid_num * (2 * eh_size)
    
            query = self.attn_query(tilde_wiki)  # [1, valid_num, hidden]
            key = self.attn_key(torch.cat([valid_wiki_h2, tilde_wiki], dim=-1))  # [wiki_sen_num, valid_num, hidden]
            atten_sum = self.attn_v(torch.tanh(query + key)).squeeze(-1)  # [wiki_sen_num, valid_num]
    
            self.beta = self.beta[:, :atten_sum.shape[0]] + torch.t(atten_sum)#

        if index == 0:
            incoming.result.atten_loss = self.atten_lossCE(self.beta, #self.alpha.t().log(),
                                                           atten_label)
        else:
            incoming.result.atten_loss += self.atten_lossCE(self.beta, #self.alpha.t().log(),
                                                            atten_label)

        self.beta = torch.t(self.beta) - 1e10 * reverse_mask[:self.beta.shape[1]]
        self.alpha = self.wiki_atten(self.beta)  # wiki_len * valid_num
        incoming.acc.prob.append(torch.index_select(self.alpha.t(), 0, incoming.state.reverse_valid_sen).cpu().tolist())
        atten_indices = torch.argmax(self.alpha, 0) # valid_num
        alpha = zeros(self.beta.t().shape).scatter_(1, atten_indices.unsqueeze(1), 1)
        alpha = torch.t(alpha)
        wiki_cv = torch.sum(valid_wiki_h_n1[:alpha.shape[0]] * alpha.unsqueeze(2), dim=0)  # valid_num * (2 * eh_size)
        conn.wiki_cv = wiki_cv
        conn.init_h = self.initLinearLayer(torch.cat([incoming.hidden.h_n, wiki_cv], 1))

        if index == 0:
            self.last_wiki = torch.index_select(wiki_cv, 0, reverse_valid_sen).unsqueeze(-1)  # [batch, 2 * eh_size]
        else:
            self.last_wiki = torch.cat([torch.index_select(wiki_cv, 0, reverse_valid_sen).unsqueeze(-1),
                                        self.last_wiki[:, :, :self.hist_len-1]], dim=-1)

        incoming.acc.label.append(torch.index_select(atten_label, 0, reverse_valid_sen).cpu().tolist())
        incoming.acc.pred.append(torch.index_select(atten_indices, 0, reverse_valid_sen).cpu().tolist())

        atten_indices = atten_indices.unsqueeze(1)
        atten_indices = torch.cat([torch.arange(atten_indices.shape[0]).unsqueeze(1), atten_indices.cpu()], 1) # valid_num * 2
        valid_wiki_h1 = torch.transpose(valid_wiki_h1, 0, 1) # valid_num * wiki_sen_len * wiki_len * (2 * eh_size)
        valid_wiki_h1 = torch.transpose(valid_wiki_h1, 1, 2) # valid_num * wiki_len * wiki_sen_len * (2 * eh_size)
        conn.selected_wiki_h = valid_wiki_h1[atten_indices.chunk(2, 1)].squeeze(1) # valid_num * wiki_sen_len * (2 * eh_size)
        conn.selected_wiki_sen = valid_wiki_sen[atten_indices.chunk(2, 1)].squeeze(1) # valid_num * wiki_sen_len


class GenNetwork(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.args = args = param.args
        self.param = param

        self.GRULayer = MyGRU(args.embedding_size + 2 * args.eh_size, args.dh_size, initpara=False)
        self.wLinearLayer = nn.Linear(args.dh_size, param.volatile.dm.vocab_size)
        self.lossCE = nn.NLLLoss(ignore_index=param.volatile.dm.unk_id)
        self.wCopyLinear = nn.Linear(args.eh_size * 2, args.dh_size)
        self.drop = nn.Dropout(args.droprate)
        self.start_generate_id = 2

    def teacherForcing(self, inp, gen):
        embedding = inp.embedding # length * valid_num * embedding_dim
        length = inp.resp_length # valid_num
        wiki_cv = inp.wiki_cv # valid_num * (2 * eh_size)
        wiki_cv = wiki_cv.unsqueeze(0).repeat(embedding.shape[0], 1, 1)
        
        gen.h, gen.h_n = self.GRULayer.forward(torch.cat([embedding, wiki_cv], dim=-1), length-1, h_init=inp.init_h, need_h=True)
        
        gen.w = self.wLinearLayer(self.drop(gen.h))
        gen.w = torch.clamp(gen.w, max=5.0)
        gen.vocab_p = torch.exp(gen.w)
        wikiState = torch.transpose(torch.tanh(self.wCopyLinear(inp.wiki_hidden)), 0, 1)
        copyW = torch.exp(torch.clamp(torch.unsqueeze(torch.transpose(torch.sum(torch.unsqueeze(gen.h, 1) * torch.unsqueeze(wikiState, 0), -1), 1, 2), 2), max=5.0))
        
        inp.wiki_sen = inp.wiki_sen[:, :inp.wiki_hidden.shape[1]]
        copyHead = zeros(1, inp.wiki_sen.shape[0], inp.wiki_hidden.shape[1], self.param.volatile.dm.vocab_size).scatter_(3, torch.unsqueeze(torch.unsqueeze(inp.wiki_sen, 0), 3), 1)
        gen.copy_p = torch.matmul(copyW, copyHead).squeeze(2)
        gen.p = gen.vocab_p + gen.copy_p + 1e-10
        gen.p = gen.p / torch.unsqueeze(torch.sum(gen.p, 2), 2)
        gen.p = torch.clamp(gen.p, 1e-10, 1.0)

    def freerun(self, inp, gen, mode='max'):
        batch_size = inp.batch_size
        dm = self.param.volatile.dm

        first_emb = inp.embLayer(LongTensor([dm.go_id])).repeat(batch_size, 1)
        gen.w_pro = []
        gen.w_o = []
        gen.emb = []
        flag = zeros(batch_size).byte()
        EOSmet = []

        inp.wiki_sen = inp.wiki_sen[:, :inp.wiki_hidden.shape[1]]
        copyHead = zeros(1, inp.wiki_sen.shape[0], inp.wiki_hidden.shape[1],
                         self.param.volatile.dm.vocab_size).scatter_(3, torch.unsqueeze(torch.unsqueeze(inp.wiki_sen, 0), 3), 1)
        wikiState = torch.transpose(torch.tanh(self.wCopyLinear(inp.wiki_hidden)), 0, 1)

        next_emb = first_emb
        gru_h = inp.init_h
        gen.p = []
        
        wiki_cv = inp.wiki_cv  # valid_num * (2 * eh_size)
        
        for _ in range(self.args.max_sent_length):
            now = torch.cat([next_emb, wiki_cv], dim=-1)
            
            gru_h = self.GRULayer.cell_forward(now, gru_h)
            w = self.wLinearLayer(gru_h)
            w = torch.clamp(w, max=5.0)
            vocab_p = torch.exp(w)
            copyW = torch.exp(torch.clamp(torch.unsqueeze((torch.sum(torch.unsqueeze(gru_h, 0) * wikiState, -1).transpose_(0, 1)), 1), max=5.0)) # batch * 1 * wiki_len
            copy_p = torch.matmul(copyW, copyHead).squeeze()

            p = vocab_p + copy_p + 1e-10
            p = p / torch.unsqueeze(torch.sum(p, 1), 1)
            p = torch.clamp(p, 1e-10, 1.0)
            gen.p.append(p)

            if mode == "max":
                w_o = torch.argmax(p[:, self.start_generate_id:], dim=1) + self.start_generate_id
                next_emb = inp.embLayer(w_o)
            elif mode == "gumbel":
                w_onehot, w_o = gumbel_max(p[:, self.start_generate_id:], 1, 1)
                w_o = w_o + self.start_generate_id
                next_emb = torch.sum(torch.unsqueeze(w_onehot, -1) * inp.embLayer.weight[2:], 1)
            gen.w_o.append(w_o)
            gen.emb.append(next_emb)

            EOSmet.append(flag)
            flag = flag | (w_o == dm.eos_id).byte()
            if torch.sum(flag).detach().cpu().numpy() == batch_size:
                break

        EOSmet = 1 - torch.stack(EOSmet)
        gen.w_o = torch.stack(gen.w_o) * EOSmet.long()
        gen.emb = torch.stack(gen.emb) * EOSmet.float().unsqueeze(-1)
        gen.length = torch.sum(EOSmet, 0).detach().cpu().numpy()
        gen.h_n = gru_h

    def forward(self, incoming):
        # incoming.data.wiki_sen: batch * wiki_len * wiki_sen_len
        # incoming.wiki_hidden.h1: wiki_sen_len * (batch *wiki_len) * (eh_size * 2)
        # incoming.wiki_hidden.h_n1: wiki_len * batch * (eh_size * 2)
        # incoming.wiki_hidden.h2: wiki_len * batch * (eh_size * 2)
        # incoming.wiki_hidden.h_n2: batch * (eh_size * 2)

        i = incoming.state.num
        valid_sen = incoming.state.valid_sen
        reverse_valid_sen = incoming.state.reverse_valid_sen

        inp = Storage()
        inp.wiki_sen = incoming.conn.selected_wiki_sen
        inp.wiki_hidden = incoming.conn.selected_wiki_h
        raw_resp_length = torch.tensor(incoming.data.resp_length[:, i], dtype=torch.long)
        raw_embedding = incoming.resp.embedding

        resp_length = inp.resp_length = torch.index_select(raw_resp_length, 0, valid_sen.cpu()).numpy()
        inp.embedding = torch.index_select(raw_embedding, 0, valid_sen).transpose(0, 1) # length * valid_num * embedding_dim
        resp = torch.index_select(incoming.data.resp[:, i], 0, valid_sen).transpose(0, 1)[1:]
        inp.init_h = incoming.conn.init_h
        inp.wiki_cv = incoming.conn.wiki_cv

        incoming.gen = gen = Storage()
        self.teacherForcing(inp, gen)
        # gen.h_n: valid_num * dh_dim

        w_slice = torch.index_select(gen.w, 1, reverse_valid_sen)
        if w_slice.shape[0] < self.args.max_sent_length:
            w_slice = torch.cat([w_slice, zeros(self.args.max_sent_length - w_slice.shape[0], w_slice.shape[1], w_slice.shape[2])], 0)
        if i == 0:
            incoming.state.w_all = w_slice.unsqueeze(0)
        else:
            incoming.state.w_all = torch.cat([incoming.state.w_all, w_slice.unsqueeze(0)], 0) #state.w_all: sen_num * sen_length * batch_size * vocab_size

        w_o_f = flattenSequence(torch.log(gen.p), resp_length-1)
        data_f = flattenSequence(resp, resp_length-1)
        incoming.statistic.sen_num += incoming.state.valid_num
        now = 0
        for l in resp_length:
            loss = self.lossCE(w_o_f[now:now+l-1, :], data_f[now:now+l-1])
            if incoming.result.word_loss is None:
                incoming.result.word_loss = loss.clone()
            else:
                incoming.result.word_loss += loss.clone()
            incoming.statistic.sen_loss.append(loss.item())
            now += l - 1

        if i == incoming.state.last - 1:
            incoming.statistic.sen_loss = torch.tensor(incoming.statistic.sen_loss)
            incoming.result.perplexity = torch.mean(torch.exp(incoming.statistic.sen_loss))

    def detail_forward(self, incoming):
        
        index = i = incoming.state.num
        valid_sen = incoming.state.valid_sen
        reverse_valid_sen = incoming.state.reverse_valid_sen
        
        inp = Storage()
        inp.wiki_sen = incoming.conn.selected_wiki_sen
        inp.wiki_hidden = incoming.conn.selected_wiki_h
        inp.init_h = incoming.conn.init_h
        inp.wiki_cv = incoming.conn.wiki_cv
        
        batch_size = inp.batch_size = incoming.state.valid_num
        inp.embLayer = incoming.resp.embLayer

        incoming.gen = gen = Storage()
        self.freerun(inp, gen)

        dm = self.param.volatile.dm
        w_o = gen.w_o.detach().cpu().numpy()

        w_o_slice = torch.index_select(gen.w_o, 1, reverse_valid_sen)
        if w_o_slice.shape[0] < self.args.max_sent_length:
            w_o_slice = torch.cat([w_o_slice, zeros(self.args.max_sent_length - w_o_slice.shape[0], w_o_slice.shape[1]).to(torch.long)], 0)

        if index == 0:
            incoming.state.w_o_all = w_o_slice.unsqueeze(0)
        else:
            incoming.state.w_o_all = torch.cat([incoming.state.w_o_all, w_o_slice.unsqueeze(0)], 0) #state.w_all: sen_num * sen_length * batch_size
