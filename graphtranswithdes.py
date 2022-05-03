#%%
from logging import exception
import pandas
import os
import re
import pickle as pkl
import numpy as np
import json
import dill
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm
import argparse
from argparse import ArgumentParser

from collections import OrderedDict
import seaborn as sns
    
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import corpus_nist

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Dropout, Linear, ModuleList, ReLU, Softplus, GELU
# from sentence_transformers import SentenceTransformer
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINConv, GCNConv, GATConv, global_add_pool, global_mean_pool
from torch.nn import MSELoss
from transformers import TFEncoderDecoderModel, BertConfig, EncoderDecoderConfig, EncoderDecoderModel, AutoTokenizer, BertModel, AutoConfig, AutoModel
from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration
#%%
def make_args():
    parser = ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--epochs', type=int, default=12, help='epochs')
    parser.add_argument('--graph-emb-dim', type=int, default=512, help = 'hidden graph embedding dimension ')
    parser.add_argument('--lr', type=float, default = 5e-5, help='learning rate')
    parser.add_argument('--weight-decay',type=float, default=0, help='weight decay')
    parser.add_argument('--num-layers', type = int, default=3, help='number of graph encoder layers')
    parser.add_argument('--num-beam', type = int, default = 3, help = 'number of beam search')
    # microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    parser.add_argument('--model-name', type = str, default = 'dmis-lab/biobert-base-cased-v1.2', help = 'dmis-lab/biobert-base-cased-v1.2, bert-base-cased, allenai/longformer-base-4096, roberta-base ')
    parser.add_argument('--decoder-model-name', type=str, default = 'same')
    parser.add_argument('--seed', type = int, default = 0, help = 'numpy random seed')
    parser.add_argument('--mode', type = str, default = 'test', help = 'train or test')
    parser.add_argument('--model', type = str, default = 'GraphTransformer', help='text2text, GraphTransformer')
    parser.add_argument('--node-description', type = str, default = 'embeddings', help = 'ids, embeddings')
    parser.add_argument('--graph-position', type = str, default='', help='the position of graph encoder in the whole structure')
    parser.add_argument('--node-encoder', type=str, default='same', help = 'all-mpnet-base-v2 or same as model-name')
    parser.add_argument('--graph-encoder', type=str, default='GAT', help = 'GIN, GCN, GAT, MLP')
    parser.add_argument('--GAT-heads', type = int, default = 4)
    parser.add_argument('--label-sentence-num', type = str, default = '3', help = '1, 3, all, trunc')
    parser.add_argument('--node-feat', type = str, default = 'labeldes', help = 'label, des, labeldes, none')
    parser.add_argument('--max-tokens', type = int, default = 500, help = '500 for roberta, 1200 for longformer, only for encoder, the decoder is not limited')
    parser.add_argument('--use-method', type = str, default = 'full', help = ' "" or full')
    parser.add_argument('--used-part', type = str, default = 'graph', help = 'graph, des or graphdes')
    parser.add_argument('--query-dim', type = int, default = 32, help = 'dimention of the query in attention module')
    return parser.parse_args()
#%%
def filter_for_bos_eos(tokens, bos=None, eos=None):
    tokens = list(tokens.cpu().numpy())
    try:
        b = tokens.index(bos) + 1
    except ValueError:
        b = 0
    try: 
        e = tokens[b:].index(eos)
        e += b
    except ValueError:
        e = len(tokens)
    if(e-b <= 1 and 'gpt' in args.decoder_model_name):
        e = b-1
        b = 0
    return tokens[b:e]

class ShiftSoftplus(Softplus):

    def __init__(self, beta=1, shift=2, threshold=20):
        super().__init__(beta, threshold)
        self.shift = shift
        self.softplus = Softplus(beta, threshold)

    def forward(self, input):
        return self.softplus(input) - np.log(float(self.shift))

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

class GCN(nn.Module):

    def __init__(self, args, additional_head = True):

        super(GCN, self).__init__()
        self.args = args
        self.num_layers = args.num_layers

        self.args = args
        if(additional_head):
            self.additional_head = Sequential(Linear(args.x_dim*2, args.x_dim), ReLU())

        self.output_layer = Sequential( Linear(args.graph_emb_dim*args.num_layers, args.graph_emb_dim), 
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

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        old_x = x[:,0,:]
        x = x.view(x.shape[0], -1)
        if(self.additional_head):
            x = self.additional_head(x)
        # if(self.args.node_feat == 'labeldes'):
        #     x = torch.mean(x, 1)
        xs = []
        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x, edge_index))
            xs.append(x)
        node_emb = torch.cat(xs, 1)
        embeddings_n = self.output_layer(node_emb)
        embeddings_g = global_mean_pool(embeddings_n, batch)
        if self.args.node_feat == 'labeldes':
            embeddings_n = embeddings_n + old_x
        embeddings_n,mask = graph_flatten(batch, embeddings_n)
        return embeddings_g.unsqueeze(1), embeddings_n, mask

class GAT(nn.Module):

    def __init__(self, args, additional_head = False):
        super(GAT, self).__init__()
        
        self.args = args
        self.num_layers = args.num_layers
        if(additional_head):
            self.additional_head = Sequential(Linear(args.x_dim*2, args.x_dim), ReLU())
        
        # self.output_layer = Sequential( Linear(args.graph_emb_dim*args.num_layers, 768), 
        #                                 GELU())
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
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        old_x = x[:,0,:]
        x = x.view(x.shape[0], -1)
        if(self.additional_head):
            x = self.additional_head(x)

        xs = []
        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x, edge_index))
            xs.append(x)
        node_emb = torch.cat(xs, 1)
        embeddings_n = self.output_layer(node_emb)
        embeddings_g = global_mean_pool(embeddings_n, batch)
        embeddings_n = embeddings_n + old_x
        embeddings_n,mask = graph_flatten(batch, embeddings_n)
        return embeddings_g.unsqueeze(1), embeddings_n, mask

class MLP(nn.Module):

    def __init__(self, args, additional_head = False):
        super(MLP, self).__init__()
        self.num_layers = args.num_layers
        self.args = args
        if(additional_head):
            self.additional_head = Sequential(Linear(args.x_dim*2, args.x_dim), ReLU())

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): 
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    def forward(self, data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        old_x = x[:,0,:]
        x = x.view(x.shape[0], -1)
        if(self.additional_head):
            x = self.additional_head(x)
    
        embeddings_g = global_mean_pool(x, batch)
        embeddings_n,mask = graph_flatten(batch, x)
        return embeddings_g.unsqueeze(1), embeddings_n, mask

class GIN(nn.Module):

    def __init__(self, args, additional_head = False):
        super(GIN, self).__init__()
        self.args = args
        self.num_layers = args.num_layers
        # self.node_classifier = nn.Linear(args.graph_emb_dim*args.num_layers, args.x_dim)
        if(additional_head):
            self.additional_head = Sequential(Linear(args.x_dim*2, args.x_dim), ReLU())
        self.convs = ModuleList()
        self.bns = ModuleList()
        self.activation = ShiftSoftplus()

        self.dense_layer = Linear(args.graph_emb_dim*args.num_layers, args.graph_emb_dim)
        self.projection_layer = Linear(args.graph_emb_dim, 768)

        self.mask_ratio = 0.1
        for i in range(self.num_layers):
            if i:
                tmpn = Sequential(Linear(args.graph_emb_dim, args.graph_emb_dim), ReLU(), Linear(args.graph_emb_dim, args.graph_emb_dim))
            else:
                tmpn = Sequential(Linear(args.x_dim, args.graph_emb_dim), ReLU(), Linear(args.graph_emb_dim, args.graph_emb_dim))
            conv = GINConv(tmpn)
            bn = nn.BatchNorm1d(args.graph_emb_dim)

            self.convs.append(conv)
            self.bns.append(bn)
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): 
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        

        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        old_x = x[:,0,:]
        x = x.view(x.shape[0], -1)
        if(self.additional_head):
            x = self.additional_head(x)
        
        xs = []
        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)
        node_emb = torch.cat(xs, 1)
        node_emb = self.dense_layer(node_emb)
        act_node = self.activation(node_emb)
        
        embeddings_n = self.projection_layer(act_node)
        embeddings_g = global_mean_pool(embeddings_n, batch)
        embeddings_n = embeddings_n + old_x
        embeddings_n,mask = graph_flatten(batch, embeddings_n)
        return embeddings_g.unsqueeze(1), embeddings_n, mask

class encoderOut(nn.Module):
    def __init__(self, last_hidden_state):
        super(encoderOut, self).__init__()
        self.lst = [last_hidden_state, None, None]
    @property
    def last_hidden_state(self):
        return self.lst[0]
    @property
    def hidden_states(self):
        return self.lst[1]    
    @property
    def attentions(self):
        return self.lst[2]
    def __getitem__(self, x):
        return self.lst[x]

def getTransformer(args, position = '', ignore_encoder = False):
    
    # model_path = os.path.join('..', 'model', model_name)

    # model_path = 'dmis-lab/biobert-base-cased-v1.1'
    if('bart' in args.decoder_model_name):
        model = BartForConditionalGeneration.from_pretrained(args.decoder_model_name)
        return model

    config_encoder = AutoConfig.from_pretrained(args.model_name)
    # config_encoder.bos_token_id=args.bos_token_id
    # config_encoder.eos_token_id=args.eos_token_id
    # config_encoder.max_position_embeddings = 1024

    config_decoder = AutoConfig.from_pretrained(args.decoder_model_name)
    config_decoder.is_decoder = True
    config_decoder.add_cross_attention = True
    config_decoder.bos_token_id=args.bos_token_id
    config_decoder.eos_token_id=args.eos_token_id
    # config_decoder.max_position_embeddings = 1024
    if (position):
        config_encoder.position_embedding_type = position
        # config_decoder.position_embedding_type = position

    # Encoder = BertModel.from_pretrained(model_path, config = config_encoder, from_tf = True)
    # Decoder = BertModel.from_pretrained(model_path, config = config_decoder, from_tf = True)
    # return  EncoderDecoderModel(encoder = Encoder, decoder = Decoder)
    # config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config = config_encoder, decoder_config = config_decoder)
    transformer = EncoderDecoderModel.from_encoder_decoder_pretrained(args.model_name, args.decoder_model_name, 
                                                                encoder_config = config_encoder, 
                                                                decoder_config = config_decoder)
    if(ignore_encoder):
        transformer.encoder = None
    return transformer


class AttentionModule(nn.Module):

    def __init__(self, args, head_num = 128, out_dim = 128):
        super(AttentionModule, self).__init__()
        self.Query = nn.Parameter(torch.Tensor(head_num, out_dim, 768))
        self.Key = nn.Parameter(torch.Tensor(head_num, out_dim, 768))
        self.out = Linear(out_dim ,768)
        torch.nn.init.xavier_uniform_(self.Query)
        torch.nn.init.xavier_uniform_(self.Key)
        self.activation = nn.LeakyReLU(negative_slope=0.4)
        for m in self.modules():
            if isinstance(m, nn.Linear): 
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, bank, mask):
        #query: [batch,768], bank[batch,len,768], mask[batch,len]

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
        
class GraphTransformer(nn.Module):

    def __init__(self, args):
        super(GraphTransformer, self).__init__()
        
        if args.graph_encoder == 'GIN':
            self.graphEncoder = GIN(args, additional_head = True)
        elif args.graph_encoder == 'GCN':
            self.graphEncoder = GCN(args, additional_head = True)
        elif args.graph_encoder == 'GAT':
            self.graphEncoder = GAT(args, additional_head = True)
        elif args.graph_encoder == 'MLP':
            self.graphEncoder = MLP(args, additional_head = True)
        else:
            raise Exception('wrong graph encoder!!')
        self.args = args
        # model_name = 'biobert_v1.1_pubmed' #E & e is different
        if(args.graph_position == 'middle'):
            self.transformer = getTransformer(args, position='None', ignore_encoder=True)
        else:
            self.transformer = getTransformer(args, position='None', ignore_encoder=False)
        self.graphAttlayer = None
        self.desAttlayer = None
        self.graphAttlayer = AttentionModule(args)
        self.desAttlayer = Sequential(Linear(768,768), nn.LayerNorm(768), nn.LeakyReLU(negative_slope=0.4))
                # self.desAttlayer = AttentionModule(args)
        self.init_emb()

    def init_emb(self):
        if self.desAttlayer is None:
            return 
        for m in self.desAttlayer.modules():
            if isinstance(m, nn.Linear): 
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def get_graph_embedding(self, data):       
        pool_emb ,emb,mask = self.graphEncoder(data)
        return pool_emb

    def forward(self, data, use_single = False):        
        pool_emb ,emb,mask = self.graphEncoder(data)

        if use_single:
            emb = pool_emb
            mask = torch.ones_like(mask[:,:1])
        else:
            if 'des' in self.args.used_part:
                emb_graph = None
                emb_des = None
                if self.graphAttlayer is not None:
                    query = pool_emb[:,0,:]
                    emb_graph =self.graphAttlayer(query, emb, mask)
                if self.desAttlayer is not None:
                    # query = pool_emb[:,0,:]
                    # emb_des = self.desAttlayer(query, data.y, data.y_mask)
                    bound = 128
                    emb_des = data.y[:,:bound,:]
                    emb_des = self.desAttlayer(emb_des)

                if emb_graph is None:
                    emb = emb_des
                    mask = data.y_mask[:,:bound]
                    # mask = torch.ones_like(emb[:,:,0])
                elif emb_des is None:
                    emb = emb_graph
                    mask = torch.ones_like(emb[:,:,0])
                else:
                    emb = torch.cat([emb_graph, emb_des], 1)
                    # mask = torch.ones_like(emb[:,:,0])
                    mask = torch.ones_like(emb_graph[:,:,0])
                    mask = torch.cat([mask, data.y_mask[:,:bound]], -1)
            else:
                pass
        # print(emb.shape)
        if self.args.graph_position == 'middle':
            encoder_outputs  =  encoderOut(emb)
            return self.transformer(encoder_outputs = encoder_outputs,
                                    attention_mask = mask,
                                    decoder_input_ids=data.labels, 
                                    labels=data.labels, 
                                    decoder_attention_mask = data.labels_mask)
        else:
            return self.transformer(inputs_embeds = emb, 
                                attention_mask = mask, 
                                decoder_input_ids=data.labels, 
                                labels=data.labels, 
                                decoder_attention_mask = data.labels_mask)

    def predict(self, data):
        pool_emb, emb, mask = self.graphEncoder(data)
        emb_graph = None
        emb_des = None
        if self.graphAttlayer is not None:
            query = pool_emb[:,0,:]
            emb_graph =self.graphAttlayer(query, emb, mask)
        if 'des' in self.args.used_part:
            if self.desAttlayer is not None:
                # query = pool_emb[:,0,:]
                # emb_des = self.desAttlayer(query, data.y, data.y_mask)
                bound = 128
                emb_des = data.y[:,:bound,:]
                emb_des = self.desAttlayer(emb_des)

            if emb_graph is None:
                emb = emb_des
                mask = data.y_mask[:,:bound]
                # mask = torch.ones_like(emb[:,:,0])
            elif emb_des is None:
                emb = emb_graph
                mask = torch.ones_like(emb[:,:,0])
            else:
                emb = torch.cat([emb_graph, emb_des], 1)
                # mask = torch.ones_like(emb[:,:,0])
                mask = torch.ones_like(emb_graph[:,:,0])
                mask = torch.cat([mask, data.y_mask[:,:bound]], -1)
        else:
            pass
        # print(emb)
        retIds = self.transformer.generate(
                            num_beams = self.args.num_beam*2,
                            max_length = 500,
                            inputs_embeds = emb, 
                            attention_mask = mask,
                            eos_token_id = self.args.eos_token_id,
                            pad_token_id = self.args.pad_token_id,
                            bos_token_id = self.args.bos_token_id,
                            early_stopping = True,
                            num_beam_groups = self.args.num_beam,
                            diversity_penalty = 0.1)
        return retIds

class Text2text(nn.Module):

    def __init__(self, args):
        super(Text2text, self).__init__()

        self.args = args
        self.transformer = getTransformer(args)
    
    def forward(self, data):
        # print(data.labels_mask)
        return self.transformer(input_ids = data.concat_ids,
                                attention_mask = data.concat_mask,
                                decoder_input_ids = data.labels, 
                                labels = data.labels,
                                decoder_attention_mask = data.labels_mask)

    def predict(self, data):
        
        return self.transformer.generate(
                            num_beams = self.args.num_beam,
                            max_length = 500,
                            input_ids = data.concat_ids,
                            attention_mask = data.concat_mask,
                            eos_token_id = self.args.eos_token_id,
                            pad_token_id = self.args.pad_token_id,
                            bos_token_id = self.args.bos_token_id)
# def get_max_node()

def getInputIds(des_list, seed, label_num, model_name, add_special_tokens = True, name = '', max_tokens = 500):

    #label each_node, node_concat
    if name:
        print('tokenizing '+name+'with'+model_name)
    path = os.path.join('tokens', name+'_seed'+str(seed)+'_'+model_name.replace('/','')+'_'+str(label_num)+'_')
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
def get_graph_input_id(tokenizer, data, graph_list, bound_list, id_list, each_node, node_concat, label):

    each_node_tokens = []
    for i in range(len(graph_list)):
        l = bound_list[i]
        r = bound_list[i+1]
        each_node_tokens.append(dict(list(zip(id_list[l:r],each_node[l:r]))))

    arc_type = set()
    for graph in tqdm(data.values()):
        for arc in graph['Arc_list']:
            arc_type.add(arc['arc_class'])
    print(arc_type)
    arc_dict = dict(list(zip(list(arc_type),range(len(arc_type)))))
    tot_arc_class = len(arc_type)
    default_id = tokenizer._convert_token_to_id('[CLS]')
    # print(len(each_node))
    node_token_length = len(each_node[0][0])
    print(node_token_length)
    data_list = []
    print('processing geometric data ...')
    for graph, node_tokens, concat_tokens, label_tokens in tqdm(list(zip(graph_list, each_node_tokens, node_concat, label))):

        id_list = []
        node_dict = graph['Node_dict']
        for node in node_dict.values():
            if node['class'] != 'COMPARTMENT' and node['class'] != 'SUBMAP':
                id_list.append(node['id'])
        new_id= list(range(len(id_list)))
        id_dict = dict(list(zip(id_list, new_id)))
        
        if(len(new_id)>400):
            continue
        x = [] 
        node_mask = []
        for iid in id_list:
            if iid in node_tokens.keys():
                x.append(node_tokens[iid][0])
                # print(type(node_tokens[iid][0]), type(node_tokens[iid]))
                # raise Exception('no!!')
                node_mask.append(node_tokens[iid][1])
            else:
                x.append([default_id] + [0]*(node_token_length-1))
                node_mask.append([1]+[0]*(node_token_length-1))
        x = torch.tensor(x, dtype = torch.long)
        node_mask = torch.tensor(node_mask, dtype = torch.long)

        edge_index = []
        edge_attr = []
        # bad_case = set()
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
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_attr = torch.tensor(edge_attr, dtype = torch.float)

        labels = torch.tensor([label_tokens[0]], dtype = torch.long)
        labels_mask = torch.tensor([label_tokens[1]], dtype = torch.long)

        concat_ids = torch.tensor([concat_tokens[0]], dtype = torch.long)   
        concat_mask = torch.tensor([concat_tokens[1]], dtype = torch.long)

        graph_data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr, 
                        node_mask = node_mask,
                        labels = labels,
                        labels_mask = labels_mask,
                        concat_ids = concat_ids,
                        concat_mask = concat_mask)
        data_list.append(graph_data)
    return data_list

def get_sentence_embedding_from_LM(model, ids, mask, full=''):

    model.eval()
    Len = len(ids)
    tokenLen = len(ids[0])
    device = torch.device('cuda')
    ids = torch.tensor(ids, dtype = torch.long)
    mask = torch.tensor(mask, dtype = torch.long)
    model.to(device)
    retList = []
    if(full == 'full'):
        print('**'*10,'important check:','tokenLen for graph des =', tokenLen)
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
    elif(full == ''):
        with torch.no_grad():
            for l in tqdm(range(0, Len, 64)):
                r = min(Len, l + 64)
                batch_id = ids[l:r].to(device)
                batch_mask = mask[l:r].to(device)
                embedding = model(input_ids = batch_id, attention_mask = batch_mask)
                senEmbedding = embedding[0][:,0,:]
                retList.append(senEmbedding.detach().cpu())
    else:
        raise Exception('wrong use method')
    return list((torch.cat(retList, 0)).numpy())
#%%
# a = [(1,2), (3,4), (5,6)]
# print(list(zip(*a)))
#%%

def get_graph_input_embeddings(seed, label_num, data, graph_list, graph_des_list, en_label, label, each_node_des, each_node_label, 
                                des_bound_list=None, des_id_list=None, des_des_list = None,
                                label_bound_list = None, label_id_list = None, label_des_list = None, 
                                model_name=None):

    path = os.path.join('embeddings','seed'+str(seed)+'_'+model_name.replace('/','')+'_'+str(label_num)+'labeldes'+'_'+args.use_method)
    if(os.path.exists(path)):
        print('*'*10+'using cached embeddings!!'+'*'*10)
        with open(path, 'rb') as fl:
            graph_embeddings, node_label_embeddings, node_des_embeddings = pkl.load(fl)
    else:
        print('getting embeddings with '+ model_name)
        if(model_name == 'all-mpnet-base-v2'):
            raise Exception('no!!')
        else:
            assert model_name == args.model_name
            config = AutoConfig.from_pretrained(args.model_name)
            config.output_hidden_states = False
            config.output_attentions = False
            model = AutoModel.from_pretrained(args.model_name, config = config)
            for p in model.parameters():
                p.requires_grad = False 
            model.eval()
            unzip_label = list(zip(*en_label))
            unzip_node_label = list(zip(*each_node_label))
            unzip_node_des = list(zip(*each_node_des))
            graph_embeddings = get_sentence_embedding_from_LM(model, unzip_label[0], unzip_label[1], full=args.use_method)
            node_label_embeddings = get_sentence_embedding_from_LM(model, unzip_node_label[0], unzip_node_label[1])
            node_des_embeddings = get_sentence_embedding_from_LM(model, unzip_node_des [0], unzip_node_des [1])

        with open(path, 'wb') as fl:
            pkl.dump([graph_embeddings, node_label_embeddings, node_des_embeddings],fl)
        print('embeddings are cached in '+path)
    unzip_label = list(zip(*en_label))
    graph_embeddings = np.asarray(graph_embeddings)
    node_label_embeddings = np.asarray(node_label_embeddings)
    node_des_embeddings = np.asarray(node_des_embeddings)

    each_node_label_embeddings = []
    each_node_des_embeddings = []
    for i in range(len(graph_list)):
        l = label_bound_list[i]
        r = label_bound_list[i+1]
        each_node_label_embeddings.append(dict(list(zip(label_id_list[l:r],list(node_label_embeddings[l:r])))))
        l = des_bound_list[i]
        r = des_bound_list[i+1]
        each_node_des_embeddings.append(dict(list(zip(des_id_list[l:r],list(node_des_embeddings[l:r])))))

    arc_type = set()
    for graph in tqdm(data.values()):
        for arc in graph['Arc_list']:
            arc_type.add(arc['arc_class'])
    print(arc_type)
    arc_dict = dict(list(zip(list(arc_type),range(len(arc_type)))))
    tot_arc_class = len(arc_type)
    tot_arc_class += 1

    data_list = []
    node_emb_dim = node_des_embeddings.shape[1]
    for graph, label_embeddings, des_embeddings, graph_e, label_tokens, y_mask in tqdm(list(zip(graph_list, each_node_label_embeddings, each_node_des_embeddings, list(graph_embeddings), label, list(unzip_label[1])))):

        id_list = []
        full_node_list = []
        node_dict = graph['Node_dict']
        for node in node_dict.values():
            if node['class'] != 'SUBMAP':
                id_list.append(node['id'])
                full_node_list.append(node)
        new_id= list(range(len(id_list)))
        id_dict = dict(list(zip(id_list, new_id)))

        x = []
        for iid in id_list:
            if iid in label_embeddings.keys():
                x1 = label_embeddings[iid]
            else:
                x1 = np.ones(node_emb_dim, dtype = np.float)
            if iid in des_embeddings.keys():
                x2 = des_embeddings[iid]
            else:
                x2 = np.ones(node_emb_dim, dtype = np.float)

            if(args.node_feat == 'label'):
                x.append(x1)
            elif (args.node_feat == 'des'):
                x.append(x2)
            elif (args.node_feat == 'labeldes'):
                x.append([x1,x2])
            elif (args.node_feat == 'none'):
                x.append(np.ones(node_emb_dim, dtype = np.float))
            else:
                raise Exception('wrong node feature')
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
                tmp = np.zeros(tot_arc_class, dtype = np.float)
                tmp[arc_dict[edge['arc_class']]] = 1
                edge_attr.append(tmp)
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_attr = torch.tensor(edge_attr, dtype = torch.float)

        labels = torch.tensor([label_tokens[0]], dtype = torch.long)
        labels_mask = torch.tensor([label_tokens[1]], dtype = torch.long)

        if(args.use_method == 'full'):
            y = torch.tensor([graph_e], dtype=torch.float)
            y_mask = torch.ByteTensor([y_mask])
        else:
            y = torch.tensor([[graph_e]], dtype=torch.float)
            y_mask = torch.ByteTensor([[1]])
        graph_data = Data(x = x, y = y, 
                        y_mask = y_mask,
                        edge_index = edge_index, edge_attr = edge_attr,
                        labels = labels,
                        labels_mask = labels_mask)
        data_list.append(graph_data)
    return data_list

#%%
def getDatalist(seed, data, dicS, label_num, encoder_name, decoder_name, form = 'embeddings', node_encoder = 'None', max_tokens = 500):

    assert form == 'embeddings' or form == 'ids'

    # node_des_list = []
    graph_des_list = []
    graph_list = []
    des_id_list = []
    label_id_list = []
    des_des_list = []
    label_des_list = []
    des_bound_list = [0]
    label_bound_list = [0]

    for graph_id in dicS['pathbank']:
        
        graph = data[graph_id]
        graph_list.append(graph)

        node_dict = graph['Node_dict']
        for node in node_dict.values():
            if node['label'] and node['class'] != 'COMPARTMENT' and node['class'] != 'SUBMAP':
                if(node['description']):
                    des_des_list.append(node['description'])
                    des_id_list.append(node['id'])
                if(node['label']):
                    label_des_list.append(node['label'])
                    label_id_list.append(node['id'])
        des_bound_list.append(len(des_id_list))
        label_bound_list.append(len(label_id_list))

        if(label_num == 'all' or label_num == 'trunc'):
            graph_des_list.append(graph['Graph description'])
        else:
            tmp_list = graph['Graph description'].split('.')
            r = min(int(label_num), len(tmp_list))
            graph_des_list.append('. '.join(tmp_list[:r]))

    label = getInputIds(graph_des_list, seed = seed, label_num = label_num, model_name = decoder_name, add_special_tokens = True, name = 'label2', max_tokens = max_tokens)
    each_node_des = getInputIds(des_des_list, seed = seed, label_num = '', model_name = encoder_name, add_special_tokens=True, name = 'nodedes', max_tokens = max_tokens)
    each_node_label = getInputIds(label_des_list, seed = seed, label_num = '', model_name = encoder_name, add_special_tokens=True, name = 'nodelabel', max_tokens = max_tokens)
    
    # with open(os.path.join('tokens', str(seed)+'allenailongformer-base-4096'), 'rb') as fl:
    #     _,each_node,node_concat = pkl.load(fl)
    # print(type(label[0]), len(label[0]))
    # raise Exception('no!!')
    print('label len:', len(label[0][0]))
    print('des len:', len(each_node_des[0][0]))
    print('node label len:', len(each_node_label[0][0]))
    if(form == 'ids'):
        raise Exception('text2text')
    else:
        en_label = getInputIds(graph_des_list, seed = seed, label_num = label_num, model_name = encoder_name, add_special_tokens = True, name = 'label2', max_tokens = max_tokens)
        if(len(en_label[0][0]) > max_tokens):
            for id, mask in en_label:
                id = id[:max_tokens]
                mask = mask[:max_tokens]
        return get_graph_input_embeddings(  seed, label_num, data, graph_list, graph_des_list, en_label, label, each_node_des, each_node_label,
                                            des_bound_list=des_bound_list, des_id_list=des_id_list, des_des_list = des_des_list,
                                            label_bound_list = label_bound_list, label_id_list = label_id_list, label_des_list = label_des_list, 
                                            model_name=node_encoder)
#%%
def Train(Iter, model, optimizer, device, single = False):

    model.train()
    # criterion = MSELoss()
    criterion = CrossEntropyLoss()
    Loss = 0
    tt = 0

    for i,data in enumerate(Iter):
        optimizer.zero_grad()
        data = data.to(device) 

        tt += data.labels.shape[0]
        ret = model(data,use_single = single)
        shape = ret.loss.shape
        shifted_prediction_scores = ret.logits[:, :-1, :].contiguous()
        labels = data.labels[:, 1:].contiguous()
        loss = criterion(shifted_prediction_scores.view(-1, shifted_prediction_scores.shape[-1]), labels.view(-1))
        Loss += loss.item() * data.labels.shape[0]

        loss.backward()
        optimizer.step()
        if(((i+1) % 5)==0):
            print(ret.logits.shape)
            print(ret.logits.max(-1))
            print(data.labels.shape)
            print(data.labels)
            print('iter:', i+1, 'loss shape:', shape, 'batch loss:', loss.item(), 'meanloss : ', Loss / tt)

    return Loss / tt
#%%
def Test(Iter, model, device):

    model.eval()
    ret = []
    std = []
    for i,data in enumerate(Iter):
        print(i,'/',len(Iter))
        data = data.to(device)
        pred = model.predict(data)
        # print(pred[:,:20])
        # print(data.labels[:,:20])

        ret.append(pred.cpu())
        std.append(data.labels.cpu())
    return ret, std

def Val(Iter, model, device, single = False):

    model.eval()
    Loss = 0
    tt = 0
    criterion = CrossEntropyLoss()

    print('validating ...')
    with torch.no_grad():
        for data in tqdm(Iter):
            data = data.to(device)
            tt += data.labels.shape[0]
            ret = model(data, use_single = single)
            shifted_prediction_scores = ret.logits[:, :-1, :].contiguous()
            labels = data.labels[:, 1:].contiguous()
            loss = criterion(shifted_prediction_scores.view(-1, shifted_prediction_scores.shape[-1]), labels.view(-1))
            Loss += loss.item() * data.labels.shape[0]
    return Loss / tt

def find_best_case(token_std, token_gen):
    maxx = 0
    std = None
    pred = None
    for x1, x2 in zip(token_std, token_gen):
        w = corpus_bleu([x1],[x2], weights=(1,))
        if(maxx < w):
            maxx = w
            std = x1
            pred = x2
    print('*'*10,'best case:', '*'*10)
    print(maxx)
    print(std)
    print(pred)

def Train_for_epoch(model, lr, weight_decay, epochs, single = False):

    trans_p = set(list(map(id, list(model.transformer.parameters()))))
    all_parameters = [p for p in model.parameters() if p.requires_grad]
    transformer_parameters = [p for p in all_parameters if id(p) in trans_p]
    others_parameters = [p for p in all_parameters if id(p) not in trans_p]
    # optimizer = torch.optim.Adam([{'params': transformer_parameters, 'lr': lr}, 
    #                             {'params': others_parameters, 'lr': lr}],
    #                             weight_decay = weight_decay)
    optimizer = torch.optim.Adam(all_parameters, lr = lr, weight_decay = weight_decay)
    # optimizer = nn.DataParallel(optimizer)

    print('parameters length:', len(all_parameters), len(list(transformer_parameters)), len(list(others_parameters)))
    print('device:',device)

    best_loss = float(1 << 30)
    train_path = model_params_path
    if single:
        train_path += '_single'
    for epoch in range(epochs):

        print("="*15,'epoch:',epoch,'='*15)

        train_loss = Train(trainData, model, optimizer, device, single)
        loss = Val(valData, model, device, single)
        print('trainloss:', train_loss, 'valloss:', loss)
        if(loss < best_loss):
            best_loss = loss
            params = model.state_dict()
            new_params = OrderedDict()
            for k,v in params.items():
                if('module.' in k):
                    k = k.replace('module.', '')
                new_params[k] = v
            torch.save(new_params, train_path)

def getGraphEmbedding(Iter, model, device):

    ret = []
    print('embedding ...')
    for data in tqdm(Iter):
        data = data.to(device)
        ret.append(model.get_graph_embedding(data).detach().cpu()[:,0,:])
    return torch.cat(ret, 0)


def findKNN(data_list, bound, model, device):

    model.eval()
    print('finding kNN ... ')

    trainData = DataLoader(data_list[:bound], shuffle= False, batch_size=args.batch_size)
    testData = DataLoader(data_list[bound:], shuffle = False, batch_size=args.batch_size)

    bank = getGraphEmbedding(trainData, model, device)
    valEmbedding = getGraphEmbedding(testData, model, device)

    query = torch.cat([bank, valEmbedding], 0)

    NN = []
    for i in tqdm(range(query.shape[0])):
        delMat = (query[i:i+1, :] - bank)
        dis = torch.norm(delMat, dim=1)
        if(i < bank.shape[0]):
            dis.data[i] = (1e20)
        best = torch.min(dis, dim = 0)[1]
        assert best.item() != i
        NN.append(best.item())
    new_data_list = []
    for data,des_id in zip(data_list,NN):
        x = data.x
        edge_index = data.edge_index 
        edge_attr = data.edge_attr
        labels = data.labels
        labels_mask = data.labels_mask
        y = data_list[des_id].y
        y_mask = data_list[des_id].y_mask
        y_label = data_list[des_id].labels
        graph = Data(x = x, y = y, y_label = y_label,
                    y_mask = y_mask,
                    edge_index = edge_index, edge_attr = edge_attr,
                    labels = labels,
                    labels_mask = labels_mask)
        new_data_list.append(graph)
    return new_data_list, NN

#%%
args = make_args()
if (args.node_encoder != 'all-mpnet-base-v2'):
    args.node_encoder = args.model_name
if (args.decoder_model_name == 'same'):
    args.decoder_model_name = args.model_name
assert args.use_method == 'full'
np.random.seed(args.seed)
with open('finaldata/pathway2text.json', 'r') as fl:
    data = json.load(fl)
with open('finaldata/mapping_database_to_pathway2text.json', 'r') as fl:
    dicS = json.load(fl)
data_list = getDatalist(args.seed, data, dicS = dicS, label_num = args.label_sentence_num,
                        encoder_name = args.model_name,
                        decoder_name = args.decoder_model_name,
                        form = args.node_description, 
                        node_encoder = args.node_encoder, 
                        max_tokens = args.max_tokens)
print('num of graphs:', len(data_list))
np.random.seed(args.seed)
np.random.shuffle(data_list)
# print(data_list[0].labels)
# print(data_list[0].x)
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

if(args.node_feat != 'labeldes'):
    args.x_dim = data_list[0].x.shape[1]
else:
    args.x_dim = data_list[0].x.shape[2]
print(args)

bound  = int(len(data_list) * 0.75)
v_bound = int(bound * 0.8)
trainData = DataLoader(data_list[:v_bound], shuffle=True, batch_size=args.batch_size)
valData = DataLoader(data_list[v_bound:bound], shuffle=True, batch_size=args.batch_size)
testData = DataLoader(data_list[bound:], shuffle = False, batch_size=args.batch_size)

if args.model == 'Text2text':
    model =Text2text(args)
    check_name = '_'.join([args.model,'seed'+str(args.seed),args.graph_position,args.model_name.replace('/', ''),args.decoder_model_name.replace('/',''), args.label_sentence_num, args.node_feat,args.used_part])
elif args.model=='GraphTransformer':
    model = GraphTransformer(args)
    check_name = '_'.join([args.model,'seed'+str(args.seed),args.graph_position, args.node_encoder.replace('/', ''), args.decoder_model_name.replace('/',''), args.label_sentence_num, args.node_feat, args.used_part])
    # if(args.graph_encoder != 'GIN'):
    check_name += args.graph_encoder
model_params_path = os.path.join('params', check_name)
result_path = os.path.join('result', check_name)
data_list_path = os.path.join('datalist', check_name)
print('\n\n\n\n','check name:',check_name,'\n\n\n')
print('GPU num:', torch.cuda.device_count())

params = torch.load(model_params_path+'_single')
model.load_state_dict(params)
model.to(device)
data_list, NN = findKNN(data_list, bound, model, device)
trainData = DataLoader(data_list[:bound], shuffle=True, batch_size=args.batch_size)
testData = DataLoader(data_list[bound:], shuffle = False, batch_size=args.batch_size)

model = GraphTransformer(args)
params = torch.load(model_params_path)
model.load_state_dict(params)
model.to(device)
print('loss:', Val(testData,model, device))
ret,std = Test(testData, model, device)

token_gen = []
token_std = []
print('converting tokens to strings')
if(tokenizer.bos_token_id is not None):
    bos = tokenizer.bos_token
if(tokenizer.eos_token_id is not None):
    eos = tokenizer.eos_token
bo = True
for batch, batch_std in tqdm(list(zip(ret,std))):
    for i in range(len(batch)):
        token_gen.append(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(filter_for_bos_eos(batch[i], args.bos_token_id, args.eos_token_id)))) 
        if(bo):
            # print(tokenizer.convert_ids_to_tokens(batch[i]))
            # print(token_gen)
            bo = False
        token_std.append([tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(filter_for_bos_eos(batch_std[i], args.bos_token_id, args.eos_token_id)))])
print(len(set(token_gen)))
bo = True
meteor_list = []
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rougeL_list = []
for xLi, x in zip(token_std, token_gen):
    if bo:
        bo = False
        print('check meteor_score ...')
    meteor_list.append(meteor_score(xLi, x))
    scores = scorer.score(xLi[0], x)
    rougeL_list.append(scores['rougeL'][1])
print('meteor_score:', np.mean(np.asarray(meteor_list)))
print('rougeL:', np.mean(np.asarray(rougeL_list)))

token_std = [[x.split()] for xLi in token_std for x in xLi]
token_gen = [x.split() for x in token_gen]
print('nist score:', corpus_nist(token_std, token_gen, n = 3))
print(corpus_bleu(token_std, token_gen, weights=(1,)), 
        corpus_bleu(token_std, token_gen, weights=(0,1)), 
        corpus_bleu(token_std, token_gen, weights=(0,0,1)),
        corpus_bleu(token_std, token_gen, weights=(0,0,0,1)),
        corpus_bleu(token_std, token_gen, weights=(0.25,0.25,0.25,0.25)))
