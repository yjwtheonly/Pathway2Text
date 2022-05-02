#%%
from logging import exception, raiseExceptions
from typing import final
from numpy.lib.function_base import average
from packaging.version import parse
import pandas
import os
import re
import pickle as pkl
import numpy as np
import json
import dill
from torch import random
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm
import argparse
import sklearn
from argparse import ArgumentParser

from collections import OrderedDict
import seaborn as sns
    
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Dropout, Linear, ModuleList, ReLU, Softplus
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINConv, GCNConv, GATConv, global_add_pool, global_mean_pool
from torch.nn import MSELoss
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel, AutoTokenizer, BertModel, AutoConfig, AutoModel
# from torchtext.data.metrics import bleu_score
from sentence_transformers import SentenceTransformer

from collections import Counter
#%%
def make_args():
    parser = ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--graph-emb-dim', type=int, default=512, help = 'hidden graph embedding dimension ')
    parser.add_argument('--lr', type=float, default = 5e-4, help='learning rate')
    parser.add_argument('--weight-decay',type=float, default=0, help='weight decay')
    parser.add_argument('--num-layers', type = int, default=3, help='number of graph encoder layers')
    parser.add_argument('--num-beam', type = int, default = 5, help = 'number of beam search')
    # microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    parser.add_argument('--model-name', type = str, default = 'dmis-lab/biobert-base-cased-v1.2', help = 'dmis-lab/biobert-base-cased-v1.2, bert-base-cased, allenai/longformer-base-4096, roberta-base ')
    parser.add_argument('--decoder-model-name', type=str, default = 'same')
    parser.add_argument('--seed', type = int, default = 0, help = 'numpy random seed')
    parser.add_argument('--mode', type = str, default = 'test', help = 'train or test')
    parser.add_argument('--node-description', type = str, default = 'embeddings', help = 'ids, embeddings')
    parser.add_argument('--graph-position', type = str, default='', help='the position of graph encoder in the whole structure')
    parser.add_argument('--node-encoder', type=str, default='same', help = 'all-mpnet-base-v2 or same as model-name')
    parser.add_argument('--graph-encoder', type=str, default='GAT', help = 'GIN, GCN, GAT, MLP')
    parser.add_argument('--GAT-heads', type = int, default = 4)
    parser.add_argument('--label-sentence-num', type = str, default = 'all', help = '1, 3, all, trunc')
    parser.add_argument('--max-tokens', type = int, default = 500, help = '500 for roberta, 850 for longformer, only for encoder, the decoder is not limited')
    parser.add_argument('--use-method', type = str, default = 'attention', help = 'single or attention')
    parser.add_argument('--use-graph-des', action='store_true')
    parser.add_argument('--multiedge', action='store_true')
    parser.add_argument('--node-feat', type=str, default = '', help = " '' or label")
    parser.add_argument('--train-ratio', type=float, default = 0.7)
    return parser.parse_args()
#%%
def graph_flatten(batch, embeddings_n):
    size = int(batch.max().item() + 1)
    embedding_lst = []
    length = []
    for i in range(size):
        embedding_lst.append(embeddings_n[batch == i,:])
        length.append(embedding_lst[i].shape[0])
    max_len = max(length)
    embeddings_n = []
    for t in embedding_lst:
        t = F.pad(t, pad=(0,0,0,max_len - t.shape[0]), mode='constant', value=0)
        t = t.unsqueeze(0)
        embeddings_n.append(t)
    embeddings_n = torch.cat(embeddings_n, 0)
    mask = torch.zeros((embeddings_n.shape[0], embeddings_n.shape[1]), dtype=torch.long, device=embeddings_n.device)

    mask.data[embeddings_n.sum(-1) != 0] = 1
    return embeddings_n, mask

class GAT(nn.Module):

    def __init__(self, args, additional_head = False):
        super(GAT, self).__init__()
        
        self.num_layers = args.num_layers
        if(additional_head):
            self.additional_head = Sequential(Linear(args.x_dim, args.x_dim), ReLU())
        
        self.output_layer = Sequential( Linear(args.graph_emb_dim*args.num_layers, args.graph_emb_dim), 
                                        ReLU(), 
                                        Linear(args.graph_emb_dim, 768))
        self.convs = ModuleList()
        act_dim = args.graph_emb_dim // args.GAT_heads
        for i in range(self.num_layers):
            if(i):
                self.convs.append(GATConv(args.graph_emb_dim, act_dim, heads=args.GAT_heads, concat = True))
            else:
                self.convs.append(GATConv(args.x_dim, act_dim, heads=args.GAT_heads, concat = True))
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): 
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    def forward(self, x, edge_index, batch, edge_attr):
        
        if(self.additional_head):
            x = self.additional_head(x)
        
        xs = []
        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x, edge_index))
            xs.append(x)
        node_emb = torch.cat(xs, 1)
        embeddings_n = self.output_layer(node_emb)
        embeddings_g = global_mean_pool(embeddings_n, batch)
        # embeddings_n,mask = graph_flatten(batch, embeddings_n)
        return embeddings_g.unsqueeze(1), embeddings_n, None

class MLP(nn.Module):

    def __init__(self, args, additional_head = False):
        super(MLP, self).__init__()
        self.num_layers = args.num_layers
        if(additional_head):
            self.additional_head = Sequential(Linear(args.x_dim, args.x_dim), ReLU())
        self.output_layer = Sequential( Linear(args.graph_emb_dim*args.num_layers, args.graph_emb_dim), 
                                        ReLU(), 
                                        Linear(args.graph_emb_dim, 768))
        self.convs = ModuleList()
        for i in range(self.num_layers):
            if(i):
                self.convs.append(Linear(args.graph_emb_dim, args.graph_emb_dim))
            else:
                self.convs.append(Linear(args.x_dim,  args.graph_emb_dim))
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): 
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    def forward(self, x, edge_index, batch, edge_attr):
        
        if(self.additional_head):
            x = self.additional_head(x)
        
        xs = []
        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x))
            xs.append(x)
        node_emb = torch.cat(xs, 1)
        embeddings_n = self.output_layer(node_emb)
        embeddings_g = global_mean_pool(embeddings_n, batch)
        # embeddings_n,mask = graph_flatten(batch, embeddings_n)
        return embeddings_g.unsqueeze(1), embeddings_n, None

#%%
class AttentionModule(nn.Module):

    def __init__(self, args, head_num = 1, out_dim = 168):
        super(AttentionModule, self).__init__()
        self.Query = nn.Parameter(torch.Tensor(head_num, out_dim, 768*2))
        self.Key = nn.Parameter(torch.Tensor(head_num, out_dim, 768))
        torch.nn.init.xavier_uniform_(self.Query)
        torch.nn.init.xavier_uniform_(self.Key)
        self.activation = nn.LeakyReLU(negative_slope=0.4)

    def forward(self, x, bank, mask):
        #query: [batch,768*2], bank[batch,len,768], mask[batch,len]

        query = x.permute(1, 0)
        query = (torch.matmul(self.Query, query))#headnum_outdim_batch

        key = bank.permute(0,2,1)
        key = (torch.matmul(self.Key.unsqueeze(1), key)) #headnum_batch_outdim_len

        query = F.tanh(query)
        key = F.tanh(key)

        key = key.permute(0,1,3,2) #headnum_batch_len_outdim
        query = query.permute(0,2,1).unsqueeze(3) #headnum_batch_outdim_1
        
        score = torch.matmul(key, query)[:,:,:,0] #headnum_batch_len
        mask = mask.unsqueeze(0)
        score = score.masked_fill(mask == 0, -1e8)
        score = F.softmax(score, dim = -1).unsqueeze(3)

        bank = bank.unsqueeze(0)
        embedding = torch.sum(score*bank, 2) #headnum_batch_768
        embedding = embedding.permute(1,0,2) #batch_headnum_768
        return self.activation(embedding)

class LinkPrediction(nn.Module):
    
    def __init__(self, args):
        super(LinkPrediction, self).__init__()
        self.args = args
        if args.graph_encoder == 'GCN':
            self.graphEncoder = GCN(args, additional_head = True)
        elif args.graph_encoder == 'GAT':
            self.graphEncoder = GAT(args, additional_head = True)
        elif args.graph_encoder == 'MLP':
            self.graphEncoder = MLP(args, additional_head = True)
        else:
            raise Exception('wrong graph encoder')

        outdim = args.tot_arc_class+1 if args.multiedge else 2

        self.AttLayer = AttentionModule(args)
        # self.AttLayer = AttentionModule(args)
        self.projection_y = Sequential(Linear(args.x_dim, args.x_dim), nn.LeakyReLU(negative_slope=0.4))
        self.pred_layer = Sequential(Linear(args.x_dim * 3, args.x_dim), ReLU(), Linear(args.x_dim, outdim))
        self.cal_mask = None
    
    def get_cal_mask(self, mask):

        if(self.cal_mask is not None):
            return self.cal_mask
        with torch.no_grad():
            self.cal_mask = (torch.zeros_like(mask[0]) == 1)
            for i in range(0,max(1,mask.shape[1]-16),8):
                self.cal_mask.data[i] = True
        return self.cal_mask
    def forward(self, data):
        
        edge_index = data.edge_index[:,data.train_edge_mask]
        edge_attr = data.edge_attr[data.train_edge_mask, :]
        pool_emb,emb,mask = self.graphEncoder(data.x, edge_index, data.batch, edge_attr)   

        train_mask = (data.train_edge_mask != True)
        train_index = torch.tensor(range(train_mask.shape[0]), dtype = torch.long).to(train_mask.device)
        train_index = train_index[train_mask]
        if(train_index.shape[0]>100):
            idx = torch.randperm(train_index.shape[0]).to(train_mask.device)
            train_index = train_index[idx][:100]

        pred_edge_index = data.edge_index[:,train_index]
        if(self.args.use_graph_des):
            y = data.y
            y = y[data.batch[pred_edge_index[0]]]
            query = torch.cat([emb[pred_edge_index[0]], emb[pred_edge_index[1]]],-1)
            desmask = data.desmask[data.batch[pred_edge_index[0]]]
            y = self.AttLayer(query, y, desmask)[:,0,:]
        else:
            y = torch.ones_like(data.y) 
            y = y[data.batch[pred_edge_index[0]]]
            y = y[:,0,:]
        y = self.projection_y(y)
        
        edge_emb = torch.cat([emb[pred_edge_index[0]], emb[pred_edge_index[1]], y], -1)
        pred = self.pred_layer(edge_emb)
        std = data.edge_label[train_index]
        return pred, std

    # def forward(self, data):
    #     # x = x, y = y, 
    #     # edge_index = edge_index, 
    #     # edge_attr = edge_attr,
    #     # train_edge_mask = train_edge_mask,
    #     # edge_label = edge_label
        
    #     edge_index = data.edge_index[:,data.train_edge_mask]
    #     edge_attr = data.edge_attr[data.train_edge_mask, :]
    #     pool_emb,emb,mask = self.graphEncoder(data.x, edge_index, data.batch, edge_attr)   
        
    #     pred_edge_index = data.edge_index[:,data.train_edge_mask != True]
    #     if(self.args.use_graph_des):
    #         y = data.y
    #         if(self.args.use_method == 'attention'):
    #             y = y.permute(0,2,1)
    #             y_list = []
    #             for conv in self.conv_list:
    #                 out = conv(y)
    #                 max_out = torch.max(out, dim = 2) #池化
    #                 y_list.append(max_out[0])
    #             # y = self.Conv(y)
    #             y = torch.cat(y_list, 1)
    #             # y = y.permute(0,2,1)
    #             # y = torch.max(y, dim = 1)[0]
    #             # print('attention!!')
    #         else:
    #             raise Exception("can't be single")
    #             # print('single!!')
    #     else:
    #         y = torch.ones_like(data.y)
    #         y = y[:,:3,:]
    #         y = y.view(y.shape[0],-1)
    #         # print('blank')
    #     y = y[data.batch[pred_edge_index[0]]]

    #     y = self.projection_y(y)
    #     edge_emb = torch.cat([emb[pred_edge_index[0]], emb[pred_edge_index[1]], y], -1)
    #     pred = self.pred_layer(edge_emb)
    #     std = data.edge_label[data.train_edge_mask != True]
    #     return pred, std

def Train(Iter, model, optimizer, device):

    model.train()
    criterion = CrossEntropyLoss()

    tt = 0
    Loss = 0
    for i,data in enumerate(Iter):
        optimizer.zero_grad()
        data = data.to(device) 


        pred, std = model(data)
        tt += std.shape[0]
        loss = criterion(pred, std)
        Loss += loss.item() * std.shape[0]

        loss.backward()
        optimizer.step()
        if(((i+1) % 5)==0):
            print('iter:', i+1, 'batch loss:', loss.item(), 'meanloss : ', Loss / tt)

    return Loss / tt
#%%

def v_val(Iter, model, device):
    Loss = 0
    tt = 0
    Acc = 0
    Pred = []
    Std = []

    print('validating ...')
    criterion = CrossEntropyLoss()
    for data in tqdm(Iter):
        data = data.to(device)
        pred, std = model(data)
        tt += std.shape[0]
        loss = criterion(pred, std)
        Loss += loss.item() * std.shape[0]
        pred_cls = pred.max(dim=1)[1]
        Acc += pred_cls.eq(std).sum().item()
        Pred.append(pred_cls.detach().cpu())
        Std.append(std.detach().cpu())
    Std = torch.cat(Std, 0).numpy()
    Pred = torch.cat(Pred, 0).numpy()
    print('std edge num:', len(Std))
    F1 = sklearn.metrics.f1_score(Std,Pred, average = None)
    # list(set(list(Std) + list(Pred)))
    return F1, Loss / tt, Acc / tt

def Val(Iter, model, device):
    
    model.eval()
    F1, loss, acc = v_val(Iter, model, device)
    # acc_list = [acc]
    # print(loss, type(loss))
    # for _ in range(4):
    #     fF1, floss, facc = v_val(Iter, model, device)
    #     acc_list.append(facc)
    #     loss += floss
    #     acc += facc
    #     F1 = np.asarray(F1) + np.asarray(fF1)
    print('F1 score:', F1)
    # print(acc_list)
    return F1, loss, acc
#%%
class DSU(object):
    def __init__(self, n, des_list):
        super(DSU, self).__init__()
        assert n == len(des_list)
        self.node_num = n
        self.father = np.array(range(n))
        self.concat_des_list = [x.lower() for x in des_list]
        self.split_des_list = [[x] for x in self.concat_des_list]
        self.each_block_id_list = [[i] for i in range(n)]
        self.counter = {}
        for i,des in enumerate(self.concat_des_list):
            self.insert(des, i)
        self.block_num = n

    def is_unique(self):
        return len(self.counter) == self.block_num

    def find(self, a):
        if(self.father[a] == a):
            return a
        fa = self.find(self.father[a])
        self.father[a] = fa
        return fa

    def delete(self, des, node):
        assert node in self.counter[des]
        self.counter[des].remove(node)
        if(len(self.counter[des])==0):
            del self.counter[des]

    def insert(self, des, node):
        if des not in self.counter.keys():
            self.counter[des] = set()
        self.counter[des].add(node)
        
    def merge(self, a, b):

        f1 = self.find(a)
        f2 = self.find(b)
        if(f1 == f2):
            return False
        self.father[f1] = f2
        self.block_num -= 1
        
        self.split_des_list[f2] += self.split_des_list[f1]

        self.delete(self.concat_des_list[f1], f1)
        self.delete(self.concat_des_list[f2], f2)
        self.split_des_list[f2].sort()
        self.concat_des_list[f2] = ' '.join(self.split_des_list[f2])
        self.insert(self.concat_des_list[f2], f2)

        self.each_block_id_list[f2] += self.each_block_id_list[f1]
        return True
    
    def is_ambiguous(self, a):
        fa = self.find(a)
        assert self.concat_des_list[fa] in self.counter.keys()
        if(len(self.counter[self.concat_des_list[fa]])>1):
            return True
        return False

    def get_node_in_block(self, a):

        fa = self.find(a)
        return self.each_block_id_list[fa]

class ComplementGraph(object):

    def __init__(self, n, edge_index, class_list):
        super(ComplementGraph, self).__init__()
        self.node_num = n
        self.edge_set = {}
        self.class_list = class_list
        #'MACROMOLECULE', 'PROCESS', 'COMPLEX', 'SIMPLE_CHEMICAL', 'MACROMOLECULE_MULTIMER'
        for f,t in edge_index:
            self.update(f, t)

    def update(self, f, t):
        if not (f in self.edge_set.keys()):
            self.edge_set[f] = set()
        self.edge_set[f].add(t)

    def exist(self, f, t):
        if not (f in self.edge_set.keys()):
            return False
        return t in self.edge_set[f]

    def generate(self):
        while(True):
            f = np.random.choice(a=self.node_num, size=1, replace=False)[0]
            t = np.random.choice(a=self.node_num, size=1, replace=False)[0]
            if(self.class_list[f] == self.class_list[t]):
                continue
            if(self.class_list[f] == 'COMPLEX' and self.class_list[t] == 'SIMPLE_CHEMICAL'):
                continue
            if(not self.exist(f, t)):
                return f, t
#%%
def get_sentence_embedding_from_LM(model, ids, mask, full = False):

    model.eval()
    Len = len(ids)
    tokenLen = len(ids[0])
    if(full):
        print('**'*10,'important check:','tokenLen for graph des =', tokenLen)
    device = torch.device('cuda')
    ids = torch.tensor(ids, dtype = torch.long)
    mask = torch.tensor(mask, dtype = torch.long)
    model.to(device)
    retList = []
    if(full):
        with torch.no_grad():
            for l in tqdm(range(0, Len, 64)):
                r = min(Len, l + 64)
                senEmbedding = []
                for L in range(0, tokenLen, 512):
                    R = min(tokenLen, L+512)
                    batch_id = ids[l:r, L:R].to(device)
                    batch_mask = mask[l:r, L:R].to(device)
                    embedding = model(input_ids = batch_id, attention_mask = batch_mask)
                    # senEmbedding = embedding[0][:,0,:]
                    senEmbedding.append(embedding[0].detach().cpu())
                retList.append(torch.cat(senEmbedding,1))
    else:
        with torch.no_grad():
            for l in tqdm(range(0, Len, 64)):
                r = min(Len, l + 64)
                batch_id = ids[l:r].to(device)
                batch_mask = mask[l:r].to(device)
                embedding = model(input_ids = batch_id, attention_mask = batch_mask)
                senEmbedding = embedding[0][:,0,:]
                retList.append(senEmbedding.detach().cpu())
    return list((torch.cat(retList, 0)).numpy())

def get_graph_input_embeddings(seed, label_num, data, graph_list, bound_list, id_list, graph_des_list, node_des_list, des_list, en_label, label, model_name = 'all-mpnet-base-v2', each_node=None, node_concat = None):

    global ppList
    path = os.path.join('LinkP','seed'+str(seed)+'_'+model_name.replace('/','')+'_'+str(label_num)+'_'+str(args.multiedge)+'_'+args.use_method+args.node_feat+'merge')
    
    if(os.path.exists(path)):
        
        print('*'*10+'using cached graphs!!'+'*'*10)
        with open(path, 'rb') as fl:
            data_list,arc_class,tmpList = dill.load(fl)
        args.tot_arc_class = arc_class
        ppList = tmpList
        return data_list
    
    path = os.path.join('embeddings','seed'+str(seed)+'_'+model_name.replace('/','')+'_'+str(label_num)+'_'+'forLP')
    if(args.use_method == 'attention'):
        path += '_'+'attention'
    path += args.node_feat
    if(os.path.exists(path)):
        print('*'*10+'using cached embeddings!!'+'*'*10)
        with open(path, 'rb') as fl:
            graph_embeddings, node_embeddings, each_embeddings = pkl.load(fl)
        unzip_label = list(zip(*en_label))
    else:
        print('getting embeddings with '+ model_name)

        assert each_node is not None and node_concat is not None
        assert model_name == args.model_name
        config = AutoConfig.from_pretrained(args.model_name)
        config.output_hidden_states = False
        config.output_attentions = False
        model = AutoModel.from_pretrained(args.model_name, config = config)
        for p in model.parameters():
            p.requires_grad = False 
        model.eval()
        unzip_label = list(zip(*en_label))
        unzip_node_concat = list(zip(*node_concat))
        unzip_each_node = list(zip(*each_node))
        node_embeddings = get_sentence_embedding_from_LM(model, unzip_node_concat[0], unzip_node_concat[1])
        each_embeddings = get_sentence_embedding_from_LM(model, unzip_each_node[0], unzip_each_node[1])

        if args.use_method == 'single':
            model = SentenceTransformer('all-mpnet-base-v2')
            graph_embeddings = model.encode(graph_des_list)
        else:
            graph_embeddings = get_sentence_embedding_from_LM(model, unzip_label[0], unzip_label[1], full = True)

        with open(path, 'wb') as fl:
            pkl.dump([graph_embeddings, node_embeddings, each_embeddings],fl)
        print('embeddings are cached in '+path)

    graph_embeddings = np.asarray(graph_embeddings)
    node_embeddings = np.asarray(node_embeddings)
    each_embeddings = np.asarray(each_embeddings)
    each_node_embeddings = []
    each_node_des = []
    for i in range(len(graph_list)):
        l = bound_list[i]
        r = bound_list[i+1]
        each_node_des.append(dict(list(zip(id_list[l:r],des_list[l:r]))))
        each_node_embeddings.append(dict(list(zip(id_list[l:r],list(each_embeddings[l:r])))))

    arc_type = set()
    for graph in tqdm(data.values()):
        for arc in graph['Arc_list']:
            arc_type.add(arc['arc_class'])
    ppList = arc_type
    print(arc_type)
    arc_dict = dict(list(zip(list(arc_type),range(len(arc_type)))))
    tot_arc_class = len(arc_type)
    args.tot_arc_class = tot_arc_class
    print(arc_dict)

    data_list = []
    node_emb_dim = node_embeddings.shape[1]
    np.random.seed(args.seed)
    global all_graph, good_cnt, bad_cnt, bad_set, ratio_list, no_arc
    for graph, embeddings, dess, graph_e, label_tokens, label_mask in tqdm(list(zip(graph_list, each_node_embeddings, each_node_des, list(graph_embeddings), label, list(unzip_label[1])))):

        id_list = []
        precise = []
        full_node_list = []
        node_dict = graph['Node_dict']
        class_list = []
        for node in node_dict.values():
            if node['class'] != 'COMPARTMENT' and node['class'] != 'SUBMAP':
                id_list.append(node['id'])
                precise.append(node['id'].split('.')[0])
                full_node_list.append(node)
                class_list.append(node['class'])
        # L = list(set(precise))
        # new_id= list(range(len(L)))
        # tmp_dict = dict(list(zip(L, new_id)))
        # aa = [tmp_dict[precise[i]] for i in range(len(id_list))]
        # id_dict = dict(list(zip(id_list, aa)))
        aa = list(range(len(id_list)))
        id_dict = dict(list(zip(id_list, aa)))

        # Pos = {}
        # for i in range(len(id_list)):
        #     Pos[precise[i]] = i
        # id_list = [id_list[Pos[a]] for a in L]
        # full_node_list = [full_node_list[Pos[a]] for a in L]
        # class_list = [class_list[Pos[a]] for a in L]

        x = []
        in_des_list = []
        for i,iid in enumerate(id_list):
            if iid in embeddings.keys():
                in_des_list.append(dess[iid])
                x.append(embeddings[iid])
            else:
                in_des_list.append('')
                bad_set.add(full_node_list[i]['class'])
                x.append(np.ones(node_emb_dim, dtype = np.float))
        x = torch.tensor(x, dtype = torch.float)
        dsu = DSU(len(id_list), in_des_list)

        edge_index = []
        edge_attr = []
        bad_case = set()
        edge_label = []
        for edge in graph['Arc_list']:
            if(not (edge['arc_sourse'] in id_dict.keys())):
                # bad_case.add(re.sub('[0-9.]','',edge['arc_sourse']))
                pass
            elif (not (edge['arc_target'] in id_dict.keys())):
                # bad_case.add(re.sub('[0-9.]','',edge['arc_target']))
                pass
            else:
                edge_index.append([id_dict[edge['arc_sourse']], id_dict[edge['arc_target']]])
                tmp = np.zeros(tot_arc_class, dtype = np.float)
                tmp[arc_dict[edge['arc_class']]] = 1
                edge_attr.append(tmp)
                edge_label.append(arc_dict[edge['arc_class']]+1 if args.multiedge else 1)

        edge_num = len(edge_index)
        if(edge_num == 0):
            continue
        if(edge_num < 10):
            continue
        edge_id = list(range(edge_num))
        bound = int(edge_num * 0.6)
        if bound < 5:
            continue
        np.random.shuffle(edge_id)

        for edge in edge_id[:bound]:
            f,t = edge_index[edge][0], edge_index[edge][1]
            dsu.merge(f, t)
        if(dsu.is_unique() is not True):
            for i in range(bound,edge_num):
                edge = edge_id[i]
                f,t = edge_index[edge][0], edge_index[edge][1]
                if(dsu.is_ambiguous(f) or dsu.is_ambiguous(t)):
                    if(dsu.merge(f, t)):
                        edge_id[bound], edge_id[i] = edge_id[i], edge_id[bound]
                        bound += 1

        if (dsu.is_unique() is not True):
            continue
        
        ratio_list.append(bound / edge_num)
        
        C_Graph = ComplementGraph(len(id_list), edge_index, class_list)
        for i in range((edge_num - bound)):
            f,t = C_Graph.generate()
            C_Graph.update(f,t)
            edge_index.append([f, t])
            edge_attr.append(np.zeros(tot_arc_class, dtype = np.float))
            edge_label.append(0)

        train_edge_mask  = (np.zeros(len(edge_index)) == 1)
        train_edge_mask[np.asarray(edge_id[:bound])] = True
     
        train_edge_mask = torch.tensor(train_edge_mask, dtype=torch.bool)
        edge_label = torch.tensor(edge_label, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_attr = torch.tensor(edge_attr, dtype = torch.float)

        if(args.use_method == 'attention'):
            y = torch.tensor([graph_e], dtype=torch.float)
            desmask = torch.ByteTensor([label_mask])
        else:
            y = torch.tensor([[graph_e]], dtype=torch.float)
            desmask = torch.ByteTensor([[1]])
        train_edge_mask = torch.tensor(train_edge_mask, dtype=torch.bool)
        graph_data = Data(x = x, y = y, 
                        edge_index = edge_index, edge_attr = edge_attr,
                        train_edge_mask = train_edge_mask,
                        edge_label = edge_label,
                        desmask = desmask)
        data_list.append(graph_data)

    path = os.path.join('LinkP','seed'+str(seed)+'_'+model_name.replace('/','')+'_'+str(label_num)+'_'+str(args.multiedge)+'_'+args.use_method+args.node_feat+'merge')
    
    with open(path, 'wb') as fl:
        dill.dump([data_list,args.tot_arc_class, ppList], fl)
    print('data_list is cached in '+path)

    return data_list
#%%
def getInputIds(des_list, seed, label_num, model_name, add_special_tokens = True, name = '', max_tokens = 500):

    #label each_node, node_concat
    if name:
        print('tokenizing '+name+'with'+model_name)
    path = os.path.join('tokens', name+'_seed'+str(seed)+'_'+model_name.replace('/','')+'_'+str(label_num)+'_'+'forLP'+args.node_feat)
    if(os.path.exists(path)):
        print('*'*10+'use cached tokens!!'+'*'*10)
        with open(path, 'rb') as fl:
            return pkl.load(fl)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = False)
    if model_name == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token
    max_len = 0
    LenList = []
    print('getting maxlen...')
    for des in tqdm(des_list):
        Token = tokenizer.encode_plus(des)
        LenList.append(len(Token['input_ids']))
    max_len = np.max(LenList)
    sns.set()
    ax = sns.displot(LenList)

    ids = []
    masks = []
    if not ('label' in name):
        if max_len>max_tokens:
            max_len = max_tokens
    elif label_num == 'trunc':
        if max_len>500:
            max_len = 500
    print('max_len:', max_len)
    print('tokenizing...')
    for des in tqdm(des_list):
        Token = tokenizer.encode_plus(des, max_length=max_len, padding='max_length', 
                                        truncation=True, add_special_tokens=add_special_tokens)
        ids.append(Token['input_ids'])
        masks.append(Token['attention_mask'])
    print('')
    ret = list(zip(ids, masks))
    with open(path, 'wb') as fl:
        pkl.dump(ret, fl)
    print('the tokens are cached in ' + path)
    return ret

#%%
def getDatalist(seed, data, dicS, label_num, encoder_name, decoder_name, form = 'embeddings', node_encoder = 'None', max_tokens = 500):

    assert form == 'embeddings' or form == 'ids'

    node_des_list = []
    graph_des_list = []
    graph_list = []
    id_list = []
    des_list = []
    bound_list = [0]

    
    for graph_id in dicS['pathbank']:
        
        graph = data[graph_id]
        graph_list.append(graph)

        node_dict = graph['Node_dict']
        tmp_des_list = []
        for node in node_dict.values():
            if node['label'] and node['class'] != 'COMPARTMENT' and node['class'] != 'SUBMAP':
                if args.node_feat == '':
                    if(node['description']):
                        des = node['description']
                        tmp_des_list.append(des)
                        des_list.append(des)
                        id_list.append(node['id'])
                    elif (node['label']):
                        des = node['label']
                        tmp_des_list.append(des)
                        des_list.append(des)
                        id_list.append(node['id'])
                else:
                    if (node['label']):
                        des = node['label']
                        tmp_des_list.append(des)
                        des_list.append(des)
                        id_list.append(node['id'])
        bound_list.append(len(id_list))

        tmp_des_list.sort()
        node_des = ". ".join(tmp_des_list)
        node_des_list.append(node_des)
        if(label_num == 'all' or label_num == 'trunc'):
            graph_des_list.append(graph['Graph description'])
        else:
            tmp_list = graph['Graph description'].split('.')
            r = min(int(label_num), len(tmp_list))
            graph_des_list.append('. '.join(tmp_list[:r]))


    label = getInputIds(graph_des_list, seed = seed, label_num = label_num, model_name = decoder_name, add_special_tokens = True, name = 'label', max_tokens = max_tokens)
    each_node = getInputIds(des_list, seed = seed, label_num = '', model_name = encoder_name, add_special_tokens=True, name = 'node', max_tokens = max_tokens)
    node_concat = getInputIds(node_des_list, seed = seed, label_num = '', model_name = encoder_name, add_special_tokens = True, name = 'concat', max_tokens = max_tokens)

    print('label len:', len(label[0][0]))
    print('concat len:', len(node_concat[0][0]))
    print('node len:', len(each_node[0][0]))
    if(form == 'ids'):
        raise Exception ('form have to be embeddings')
    else:
        en_label = getInputIds(graph_des_list, seed = seed, label_num = label_num, model_name = node_encoder, add_special_tokens = True, name = 'label', max_tokens = max_tokens)
        return get_graph_input_embeddings(seed, label_num, data, graph_list, bound_list, id_list, graph_des_list, node_des_list, des_list, en_label, label, each_node=each_node, node_concat = node_concat, model_name=node_encoder)    
#%%
args = make_args()
assert args.use_method in {'single','attention'}
if (args.node_encoder != 'all-mpnet-base-v2'):
    args.node_encoder = args.model_name
if (args.decoder_model_name == 'same'):
    args.decoder_model_name = args.model_name
with open('finaldata/pathway2text.json', 'r') as fl:
    data = json.load(fl)
with open('finaldata/mapping_database_to_pathway2text.json', 'r') as fl:
    dicS = json.load(fl)
#%%
all_graph = 0
good_cnt = 0
bad_cnt = 0
bad_set = set()
ratio_list = []
no_arc = []
ppList = 0
np.random.seed(args.seed)
data_list = getDatalist(args.seed, data, dicS = dicS, label_num = args.label_sentence_num,
                        encoder_name = args.model_name,
                        decoder_name = args.decoder_model_name,
                        form = args.node_description, 
                        node_encoder = args.node_encoder, 
                        max_tokens = args.max_tokens)
print(sum(ratio_list), len(ratio_list), len(data_list))
print(ppList)
#%%
T_test = []
T_L_test = []
np.random.seed(args.seed)
for times in range(1):
    print('*'*30, times)
    np.random.shuffle(data_list)
    device = torch.device('cuda')
    args.x_dim = data_list[0].x.shape[1]
    print(args)
    bound  = int(len(data_list) * args.train_ratio)
    trainData = DataLoader(data_list[:bound], shuffle=True, batch_size=args.batch_size)
    valData = DataLoader(data_list[bound:], shuffle = False, batch_size=args.batch_size)
#%%
    check_name = '_'.join(['seed'+str(args.seed),args.graph_position, 
                        args.node_encoder.replace('/', ''), 
                        args.decoder_model_name.replace('/',''), 
                        args.label_sentence_num,
                        args.graph_encoder,
                        'forLP','usedes='+str(args.use_graph_des), 'multi='+str(args.multiedge),
                        'usemethod='+args.use_method, 'node_feat='+args.node_feat])
    model_params_path = os.path.join('params', check_name)
    result_path = os.path.join('result', check_name)
    print('\n\n\n\n','check name:',check_name,'\n\n\n')
    model = LinkPrediction(args)
    model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay = args.weight_decay)

    best_acc = 0
    bestF1 = None
    for epoch in range(args.epochs):

        print("="*15,'epoch:',epoch,'='*15)

        train_loss = Train(trainData, model, optimizer, device)
        F1, loss, acc = Val(valData, model, device)
        print('trainloss:', train_loss, 'valloss:', loss, 'valacc:', acc)
        if bestF1 is None:
            bestF1 = F1
        for i,a in enumerate(F1):
            if(a > bestF1[i]):
                bestF1[i] = a
        if(acc > best_acc):
            best_acc = acc
            params = model.state_dict()
            torch.save(params, model_params_path)
    ppList = list(ppList)
    nList = ['noarc'] + ppList
    goalList = ['CATALYSIS','STIMULATION','LOGIC_ARC','INHIBITION','CONSUMPTION','PRODUCTION', 'BELONG_TO']
    finall = [best_acc, bestF1[0]]
    tmpdict = dict(list(zip(nList, bestF1)))
    for x in goalList:
        finall.append(tmpdict[x])
    print(finall)
    T_test.append(finall)
print(T_test)
AA = [T[0] for T in T_test]
AA = np.asarray(AA)
print('mean acc:', np.mean(AA))