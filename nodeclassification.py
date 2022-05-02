#%%
from logging import exception
from typing import final
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
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
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
    parser.add_argument('--max-tokens', type = int, default = 500, help = '500 for roberta, 1200 for longformer, only for encoder, the decoder is not limited')
    parser.add_argument('--use-method', type = str, default = 'attention', help = 'single or attention')
    parser.add_argument('--use-graph-des', action='store_true')
    parser.add_argument('--node-feat', type=str, default = '', help = " '' or label")
    parser.add_argument('--chosen-class', type=str, default = 'MACROMOLECULE_MULTIMER', help= 'MACROMOLECULE, SIMPLE_CHEMICAL, MACROMOLECULE_MULTIMER')
    return parser.parse_args()
#%%
def filter_for_bos_eos(tokens, bos='[CLS]', eos='[SEP]'):
    try:
        b = tokens.index(bos) + 1
    except ValueError:
        b = 0
    try: 
        e = tokens.index(eos)
    except ValueError:
        e = len(tokens)
    return tokens[b:e]

class GCN(nn.Module):

    def __init__(self, args, additional_head = False):

        super(GCN, self).__init__()

        self.num_layers = args.num_layers

        if(additional_head):
            self.additional_head = Sequential(Linear(args.x_dim, args.x_dim), ReLU())

        self.output_layer = Sequential( Linear(args.graph_emb_dim, args.graph_emb_dim), 
                                        ReLU(), 
                                        Linear(args.graph_emb_dim, 768))
        self.convs = ModuleList()
        for i in range(self.num_layers):
            if(i):
                self.convs.append(GCNConv(args.graph_emb_dim, args.graph_emb_dim))
            else:
                self.convs.append(GCNConv(args.x_dim,  args.graph_emb_dim))
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
            xs.append(x.unsqueeze(1))
        node_emb = torch.cat(xs, 1)
        embeddings_n = self.output_layer(node_emb)
        embeddings_g = global_mean_pool(xs[-1][:,0,:], batch)
        # embeddings_n,mask = graph_flatten(batch, embeddings_n)
        return embeddings_g.unsqueeze(1), embeddings_n, None

class GAT(nn.Module):

    def __init__(self, args, additional_head = False):
        super(GAT, self).__init__()
        
        self.num_layers = args.num_layers
        if(additional_head):
            self.additional_head = Sequential(Linear(args.x_dim, args.x_dim), ReLU())
        
        self.output_layer = Sequential( Linear(args.graph_emb_dim, args.graph_emb_dim), 
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
            xs.append(x.unsqueeze(1))
        node_emb = torch.cat(xs, 1)
        embeddings_n = self.output_layer(node_emb)
        embeddings_g = global_mean_pool(xs[-1][:,0,:], batch)
        # embeddings_n,mask = graph_flatten(batch, embeddings_n)
        return embeddings_g.unsqueeze(1), embeddings_n, None
#%%
class AttentionModule(nn.Module):

    def __init__(self, args, head_num = 1, out_dim = 128):
        super(AttentionModule, self).__init__()
        self.Query = nn.Parameter(torch.Tensor(head_num, out_dim, args.num_layers*768))
        self.Key = nn.Parameter(torch.Tensor(head_num, out_dim, 768))
        torch.nn.init.xavier_uniform_(self.Query)
        torch.nn.init.xavier_uniform_(self.Key)

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
        return embedding

def getTransformer(args, position = '', ignore_encoder = False):

    config_encoder = AutoConfig.from_pretrained(args.model_name)

    config_decoder = AutoConfig.from_pretrained(args.decoder_model_name)
    config_decoder.is_decoder = True
    config_decoder.add_cross_attention = True
    config_decoder.bos_token_id=args.bos_token_id
    config_decoder.eos_token_id=args.eos_token_id
    if (position):
        config_encoder.position_embedding_type = position

    transformer = EncoderDecoderModel.from_encoder_decoder_pretrained(args.model_name, args.decoder_model_name, 
                                                                encoder_config = config_encoder, 
                                                                decoder_config = config_decoder)
    if(ignore_encoder):
        transformer.encoder = None
    return transformer
class NodeClassification(nn.Module):
    
    def __init__(self, args):
        super(NodeClassification, self).__init__()
        self.args = args
        if args.graph_encoder == 'GCN':
            self.graphEncoder = GCN(args, additional_head = True)
        elif args.graph_encoder == 'GAT':
            self.graphEncoder = GAT(args, additional_head = True)
        # elif args.graph_encoder == 'MLP':
        #     self.graphEncoder = MLP(args, additional_head = True)
        else:
            raise Exception('wrong graph encoder')

        outdim = args.tot_node_class

        # if args.use_graph_des:
        #     if args.use_method == 'attention':
        if(args.use_method == 'attention'):
            # self.Conv = nn.Conv1d(args.x_dim, args.x_dim, 16, stride = 8)
            self.AttLayer = AttentionModule(args)
        self.projection_y = Sequential(Linear(args.x_dim, args.x_dim), Sigmoid())
        self.pred_layer = Sequential(Linear(args.x_dim * (args.num_layers+1), args.x_dim), ReLU(), Linear(args.x_dim, outdim))
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
                
        x = data.x 
        train_mask = data.train_node_mask
        train_index = torch.tensor(range(train_mask.shape[0]), dtype = torch.long).to(train_mask.device)
        train_index = train_index[train_mask]
        if(train_index.shape[0]>x.shape[0] // 10 and x.shape[0] // 10 > 0):
            idx = torch.randperm(train_index.shape[0]).to(train_mask.device)
            train_index = train_index[idx][:x.shape[0] // 10]
        x[train_index] = 0
        pool_emb,emb,_ = self.graphEncoder(x, data.edge_index, data.batch, data.edge_attr)   

        node_emb = emb[train_index]
        if self.args.use_graph_des:
            y = data.y[data.batch[train_index]]
        else:
            # print('blank!')
            y = torch.ones_like(data.y[data.batch[train_index]])
        mask = data.desmask[data.batch[train_index]]
        y = self.AttLayer(node_emb.view(node_emb.shape[0], -1), y, mask)
        y = y[:,0,:]

        y = self.projection_y(y)
        node_emb = torch.cat([node_emb.view(node_emb.shape[0], -1), y], -1)
        pred = self.pred_layer(node_emb)
        std = data.node_type[train_index]
        return pred, std


#%%
def Train(Iter, model, optimizer, device):

    model.train()
    criterion = CrossEntropyLoss()

    tt = 0
    Loss = 0
    for i,data in enumerate(Iter):
        optimizer.zero_grad()
        data = data.to(device) #no need for DataListLoader

        pred, std = model(data)
        tt += std.shape[0]
        loss = criterion(pred, std)
        Loss += loss.item() * std.shape[0]

        loss.backward()
        optimizer.step()
        if(((i+1) % 5)==0):
            print('iter:', i+1, 'batch loss:', loss.item(), 'meanloss : ', Loss / tt, 'node num:', std.shape[0])

    return Loss / tt
#%%
def Val(Iter, model, device):
    
    model.eval()
    Loss = 0
    tt = 0
    Acc = 0
    Pred = []
    Std = []
    acclist = []

    print('validating ...')
    criterion = CrossEntropyLoss()
    for _ in range(5):
        innertt = 0
        inneracc = 0
        for data in tqdm(Iter):
            data = data.to(device)
            pred, std = model(data)
            # pred = torch.rand(pred.shape).to(pred.device)
            tt += std.shape[0]
            innertt += std.shape[0]
            loss = criterion(pred, std)
            Loss += loss.item() * std.shape[0]
            pred_cls = pred.max(dim=1)[1]
            Acc += pred_cls.eq(std).sum().item()
            inneracc += pred_cls.eq(std).sum().item()
            # print(pred_cls)
            # print(std)
            # print(Acc)
            Pred.append(pred_cls.detach().cpu())
            Std.append(std.detach().cpu())
        acclist.append(inneracc / innertt)
    Std = torch.cat(Std, 0).numpy()
    Pred = torch.cat(Pred, 0).numpy()

    print('pred label num:', len(set(list(Pred))))
    print('std label num:', len(set(list(Std))))
    print('acc list:', acclist)
    
    return Loss / tt, Acc / tt, acclist
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

def get_graph_input_embeddings( seed, 
                                label_num, data, graph_list, 
                                bound_list, id_list, graph_des_list, 
                                node_des_list, des_list, en_label, label, 
                                model_name = 'all-mpnet-base-v2', each_node=None, 
                                node_concat = None, node_target = None):

    path = os.path.join('NodeClassification','seed'+str(seed)+'_'+model_name.replace('/','')+'_'+str(label_num)+'_'+args.use_method+args.node_feat+'_'+args.chosen_class)
    
    if(os.path.exists(path)):
        print('*'*10+'using cached graphs!!'+'*'*10)
        with open(path, 'rb') as fl:
            data_list,tot_node_class = dill.load(fl)
        args.tot_node_class = tot_node_class
        return data_list
    
    path = os.path.join('embeddings','seed'+str(seed)+'_'+model_name.replace('/','')+'_'+str(label_num)+'_'+'forNode')
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
        # graph_embeddings = get_sentence_embedding_from_LM(model, unzip_label[0], unzip_label[1])
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
    each_node_target = []
    for i in range(len(graph_list)):
        l = bound_list[i]
        r = bound_list[i+1]
        each_node_target.append(dict(list(zip(id_list[l:r],node_target[l:r]))))
        each_node_embeddings.append(dict(list(zip(id_list[l:r],list(each_embeddings[l:r])))))

    arc_type = set()
    add = 0
    all_node = 0
    stadict = {}
    for graph in tqdm(graph_list):
        for arc in graph['Arc_list']:
            arc_type.add(arc['arc_class'])
        for node in graph['Node_dict'].values():
            if(node['class'] != 'COMPARTMENT' and node['class'] != 'SUBMAP'):
                if node['label'] is not None:
                    if not (node['class'] in stadict.keys()):
                        stadict[node['class']]=[set(),0]
                    add += 1
                    stadict[node['class']][0].add(node['label'])
                    stadict[node['class']][1]+=1
    # print(node_class_set)

    print(arc_type)
    arc_dict = dict(list(zip(list(arc_type),range(len(arc_type)))))
    node_class_dict = dict(list(zip(list(stadict[args.chosen_class][0]), range(len(stadict[args.chosen_class][0])))))
    cal_node_class = dict(list(zip(list(stadict[args.chosen_class][0]), [0]*len(stadict[args.chosen_class][0]))))
    tot_node_class = len(node_class_dict)
    tot_arc_class = len(arc_dict)
    args.tot_node_class = tot_node_class
    print('all nodes num:', add)
    print('node label num:', tot_node_class, 'tot node:', stadict[args.chosen_class][1])

    data_list = []
    node_emb_dim = node_embeddings.shape[1]
    target_token_len = len(node_target[0][0])

    tot_node = 0
    np.random.seed(args.seed)
    global all_graph, good_cnt, bad_cnt, bad_set, ratio_list, no_arc
    for graph, embeddings, node_ids, graph_e, label_tokens, label_mask in tqdm(list(zip(graph_list, each_node_embeddings, each_node_target, list(graph_embeddings), label, list(unzip_label[1])))):

        id_list = []
        full_node_list = []
        node_dict = graph['Node_dict']
        class_list = []
        node_type = []
        has_target = []
        for node in node_dict.values():
            if node['class'] != 'SUBMAP':
                id_list.append(node['id'])
                full_node_list.append(node)
                class_list.append(node['class'])
                # node_type.append(node_class_dict[node['class']])
                if node['class'] ==args.chosen_class:
                    node_type.append(node_class_dict[node['label']])
                    has_target.append(1)
                    # cal_node_class[node['label']] += 1
                else:
                    node_type.append(2)
                    has_target.append(0)
        # L = list(id_list)
        # new_id= list(range(len(L)))
        # tmp_dict = dict(list(zip(L, new_id)))
        # aa = [tmp_dict[precise[i]] for i in range(len(id_list))]
        aa = list(range(len(id_list)))
        id_dict = dict(list(zip(id_list, aa)))
    
        # Pos = {}
        # for i in range(len(id_list)):
        #     Pos[precise[i]] = i
        # id_list = [id_list[Pos[a]] for a in L]
        # full_node_list = [full_node_list[Pos[a]] for a in L]
        # class_list = [class_list[Pos[a]] for a in L]
        # node_type = [node_type[Pos[a]] for a in L]
        # has_target = [has_target[Pos[a]] for a in L]
        
        for i,node in enumerate(full_node_list):
            if (has_target[i] == 1):
                cal_node_class[node['label']] += 1
        all_node += len(id_list)

        x = []
        for i,iid in enumerate(id_list):
            if iid in embeddings.keys():
                x.append(embeddings[iid])
            else:
                bad_set.add(full_node_list[i]['class'])
                x.append(np.ones(node_emb_dim, dtype = np.float))
        x = torch.tensor(x, dtype = torch.float)

        edge_index = []
        edge_attr = []
        bad_case = set()
        for edge in graph['Arc_list']:
            if(not (edge['arc_sourse'] in id_dict.keys())):
                # bad_case.add(re.sub('[0-9.]','',edge['arc_sourse']))
                pass
            elif (not (edge['arc_target'] in id_dict.keys())):
                # bad_case.add(re.sub('[0-9.]','',edge['arc_target']))
                pass
            else:
                edge_index.append([id_dict[edge['arc_sourse']], id_dict[edge['arc_target']]])
                edge_index.append([id_dict[edge['arc_target']], id_dict[edge['arc_sourse']]])
                tmp = np.zeros(tot_arc_class, dtype = np.float)
                tmp[arc_dict[edge['arc_class']]] = 1
                edge_attr.append(tmp)
                edge_attr.append(np.zeros(tot_arc_class, dtype = np.float))
        
        if(len(edge_index) == 0):
            continue
        train_node_mask = np.asarray(has_target, dtype = np.long)
        train_node_mask = torch.tensor(train_node_mask, dtype = torch.long)
        train_node_mask = (train_node_mask == 1)
        if(torch.sum(train_node_mask) == 0):
            continue
        tot_node+=torch.sum(train_node_mask).item()
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_attr = torch.tensor(edge_attr, dtype = torch.float)

        if(args.use_method == 'attention'):
            y = torch.tensor([graph_e], dtype=torch.float)
            desmask = torch.ByteTensor([label_mask])
        else:
            y = torch.tensor([[graph_e]], dtype=torch.float)
            desmask = torch.ByteTensor([[1]])

        node_type = torch.tensor(node_type, dtype=torch.long)

        graph_data = Data(x = x, y = y, 
                        edge_index = edge_index, edge_attr = edge_attr,
                        train_node_mask = train_node_mask,
                        node_type = node_type,
                        desmask = desmask)
        data_list.append(graph_data)

    path = os.path.join('NodeClassification','seed'+str(seed)+'_'+model_name.replace('/','')+'_'+str(label_num)+'_'+args.use_method+args.node_feat+'_'+args.chosen_class)
    
    with open(path, 'wb') as fl:
        dill.dump([data_list, args.tot_node_class], fl)
    print('data_list is cached in '+path)
    print('tot node num:', tot_node, 'all node:', all_node)

    maxx = 0
    st = ''
    for k ,v in cal_node_class.items():
        if(v > maxx):
            maxx = v
            st = k
    print('major label:', st, 'ratio:', maxx / tot_node)

    return data_list
#%%
def getInputIds(des_list, seed, label_num, model_name, add_special_tokens = True, name = '', max_tokens = 500):

    #label each_node, node_concat
    if name:
        print('tokenizing '+name+'with'+model_name)
    path = os.path.join('tokens', name+'_seed'+str(seed)+'_'+model_name.replace('/','')+'_'+str(label_num)+'_'+'forNode'+args.node_feat)
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
    target_des_list = []


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

                        tmp_list = graph['Graph description'].split('.')
                        target_des_list.append(tmp_list[0])

                        id_list.append(node['id'])
                    elif (node['label']):
                        des = node['label']
                        tmp_des_list.append(des)
                        des_list.append(des)
                        target_des_list.append(des)
                        id_list.append(node['id'])
                else:
                    raise Exception('please full node description')
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
    target_each_node =  getInputIds(target_des_list, seed = seed, label_num = '', model_name = encoder_name, add_special_tokens=True, name = 'targetnode', max_tokens = max_tokens)

    print('label len:', len(label[0][0]))
    print('concat len:', len(node_concat[0][0]))
    print('node len:', len(each_node[0][0]))
    print('target node len:', len(target_each_node[0][0]))
    if(form == 'ids'):
        raise Exception ('form have to be embeddings')
    else:
        en_label = getInputIds(graph_des_list, seed = seed, label_num = label_num, model_name = node_encoder, add_special_tokens = True, name = 'label', max_tokens = max_tokens)
        return get_graph_input_embeddings(  seed, label_num, data, graph_list, bound_list, 
                                            id_list, graph_des_list, node_des_list, des_list, 
                                            en_label, label, each_node=each_node, 
                                            node_concat = node_concat, model_name=node_encoder, node_target = target_each_node)    
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
np.random.seed(args.seed)
data_list = getDatalist(args.seed, data, dicS = dicS, label_num = args.label_sentence_num,
                        encoder_name = args.model_name,
                        decoder_name = args.decoder_model_name,
                        form = args.node_description, 
                        node_encoder = args.node_encoder, 
                        max_tokens = args.max_tokens)
print(len(data_list))
#%%
device = torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained(args.decoder_model_name, use_fast = False)

if(tokenizer.eos_token_id is not None):
    args.eos_token_id = tokenizer.eos_token_id
else:
    args.eos_token_id = tokenizer._convert_token_to_id('[SEP]')

if(tokenizer.bos_token_id is not None):
    args.bos_token_id = tokenizer.bos_token_id
else:
    args.bos_token_id = tokenizer._convert_token_to_id('[CLS]')

if(tokenizer.pad_token_id is not None):
    args.pad_token_id = tokenizer.pad_token_id
else:
    tokenizer.pad_token = tokenizer.eos_token
    args.pad_token_id = tokenizer._convert_token_to_id('[PAD]')

args.x_dim = data_list[0].x.shape[1]
print(args)
bound  = int(len(data_list) * 0.75)
#%%
check_name = '_'.join(['seed'+str(args.seed),args.graph_position, 
                     args.node_encoder.replace('/', ''), 
                     args.decoder_model_name.replace('/',''), 
                     args.label_sentence_num,
                     args.graph_encoder,
                     'forNodeClass','usedes='+str(args.use_graph_des),
                     'usemethod='+args.use_method, 'node_feat='+args.node_feat])
model_params_path = os.path.join('params', check_name)
result_path = os.path.join('result', check_name)
print('\n\n\n\n','check name:',check_name,'\n\n\n')
print('GPU num:', torch.cuda.device_count())
T_test = []
T_L_test = []
np.random.seed(args.seed)

for times in range(1):
    print('*'*30, times)
    np.random.shuffle(data_list)
    trainData = DataLoader(data_list[:bound], shuffle=True, batch_size=1)
    valData = DataLoader(data_list[bound:], shuffle = False, batch_size=1)

    model = NodeClassification(args)
    model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay = args.weight_decay)

    best_acc = 0
    best_list = None
    for epoch in range(args.epochs):

        print("="*15,'epoch:',epoch,'='*15)

        train_loss = Train(trainData, model, optimizer, device)
        loss, acc, acc_list = Val(valData, model, device)
        print('trainloss:', train_loss, 'valloss:', loss, 'valacc:', acc)
        if(acc > best_acc):
            best_acc = acc
            best_list = acc_list
            params = model.state_dict()
            torch.save(params, model_params_path)
    print('best acc:', best_acc)
    T_test.append(best_acc)
    print(best_list)
    T_L_test += best_list
print('T_test:', T_test)