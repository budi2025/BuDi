'''
***********************************************************************
Enhancing Bundle Recommendation via Bundle-Item Relations and User Individuality

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: model.py
- classes of models (SASRec_2, PopCon for reranking, FFN)

Version: 1.0
***********************************************************************
'''
from collections import Counter
import random
import numpy as np
import torch
from scipy.sparse import csr_matrix
from custom_transformer import MultiheadAttention as MyMultiheadAttention
from torch import nn
import torch.nn.functional as F
from utils import *
from tqdm import tqdm

'''
feed forawrd network for BuDi

input:
    * hidden_units: dimension of hidden vectors
    * dropout_rate: ratio for drop out technique

output:
    * outputs: layer output
'''
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

'''
Backbone model of BuDi: SASRec with bundle-encoder

input:
* user_num: number of users
* bundle_num: number of bundles
* item_num: number of items
* bundle_item_list_dict: dictionary of bundle-item affiliation 
* args: pre-defined arguments using argparse
'''
class SASRec_2(torch.nn.Module):
    # initialization of model
    def __init__(self, user_num, bundle_num, item_num, bundle_item_list_dict, args):
        super(SASRec_2, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.bundle_num = bundle_num
        self.dev = args.device
        self.bundle_item_list_dict =bundle_item_list_dict
        self.args = args
        self.infonce=nn.CrossEntropyLoss(reduction="sum")

        self.N = 2 * args.batch_size
        self.mask = torch.ones((self.N, self.N), dtype=bool)
        self.mask = self.mask.fill_diagonal_(0)
        for i in range(args.batch_size):
            self.mask[i, args.batch_size + i] = 0
            self.mask[args.batch_size + i, i] = 0

        indices = []  
        values = []
        for bundle_id, item_list in self.bundle_item_list_dict.items():
            if bundle_id > self.bundle_num:
                continue
            for item_id in item_list:
                indices.append([bundle_id, item_id])
                values.append(1)    
        indices=torch.tensor(indices, dtype=torch.long).t()
        values = torch.tensor(values, dtype=torch.float32)
        self.affiliation_matrix  = torch.sparse_coo_tensor(indices, values, (self.bundle_num+1, self.item_num+1)).to(self.dev) # (bundle_num+1, item_num+1)

        self.item_emb = torch.nn.Embedding(self.item_num+1, int((args.hidden_units)/2), padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.bundle_level_bundle_emb = torch.nn.Embedding(self.bundle_num+1, int((args.hidden_units)/2), padding_idx=0)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  MyMultiheadAttention(args.hidden_units,
                                                   args.num_heads,
                                                   args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
    
    # get bundle embeddings by concatenating bundle-view and item-view embeddings
    def bund_emb(self, ids=None):
        item_embeddings = self.item_emb.weight  # (num_items + 1, embedding_dim)
        bundle_summed_embeddings = torch.sparse.mm(self.affiliation_matrix, item_embeddings)  # shape: (bundlenum+1, embedding_dim)
        
        bundle_item_counts = torch.sparse.sum(self.affiliation_matrix, dim=1).to_dense()  # (num_bundles+1,)
        bundle_item_counts[0] = 1e-8
        bundle_avg_embeddings = bundle_summed_embeddings / bundle_item_counts.unsqueeze(1)  # (num_bundles+1, embedding_dim)

        if ids is None:
            return bundle_avg_embeddings
        
        selected_bundle_embeddings = bundle_avg_embeddings[ids]  # (batch_size, seq_length, embedding_dim)

        return selected_bundle_embeddings
    
    # get bundle embeddings for item-level masked sequence
    def masked_bund_emb(self, masked_logs_seqs):
        masked_embs = self.item_emb(masked_logs_seqs)
        sum_embeddings = torch.sum(masked_embs, dim=2)
        count_nonzero = torch.count_nonzero(masked_logs_seqs, dim=2).clamp(min=1).float()
        avg_embeddings = sum_embeddings / count_nonzero.unsqueeze(-1)
        
        return avg_embeddings
    
    # gets sequence representation using self-attention
    def log2feats(self, log_seqs, seqs):
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) 

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask,
                                            key_padding_mask=timeline_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) 

        return log_feats
    
    '''
    forward propagation of the model

    input:
        * log_seqs: sequence history
        * pos_seqs: positive bundles
        * neg_seqs: sampled negative bundles
        * augment: perform item-level masking or not
        * masked_seq: masked sequence using item-level masking

    output:
        * pos_logits: positive logits
        * neg_logits: negative logits
    '''  
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, augment=False, masked_seq=None):        
        bund_embs = self.bund_emb().to(self.dev)
        if augment == True:
            # item-level masking
            item_level_seqs = self.masked_bund_emb(masked_seq)
        else:
            item_level_seqs = bund_embs[torch.LongTensor(log_seqs).to(self.dev)]
        bundle_level_seqs = self.bundle_level_bundle_emb(torch.LongTensor(log_seqs).to(self.dev)) 

        # twolevel_seqs = item_level_seqs + bundle_level_seqs
        twolevel_seqs = torch.cat((item_level_seqs, bundle_level_seqs), dim=2)
        twolevel_log_feats = self.log2feats(log_seqs, twolevel_seqs)

        pos_embs = torch.cat((bund_embs[(torch.LongTensor(pos_seqs).to(self.dev))], self.bundle_level_bundle_emb(torch.LongTensor(pos_seqs).to(self.dev))), dim=2)
        neg_embs = torch.cat((bund_embs[(torch.LongTensor(neg_seqs).to(self.dev))], self.bundle_level_bundle_emb(torch.LongTensor(neg_seqs).to(self.dev))), dim=2)

        pos_logits = (twolevel_log_feats * pos_embs).sum(dim=-1) 
        neg_logits = (twolevel_log_feats * neg_embs).sum(dim=-1) 

        return pos_logits, neg_logits

    '''
    predicts the recommendation score between all users and given bundle_indices when sequence history(log_seqs) is provided 

    input:
        * log_seqs: sequence history
        * bundle_indices: candidate bundle ids

    * logits: recommendation score
    '''   
    def predict(self, user_ids, log_seqs, bundle_indices): # for inference
        bund_embs = self.bund_emb().to(self.dev)
        item_level_seqs = bund_embs[torch.LongTensor(log_seqs).to(self.dev)]
        bundle_level_seqs = self.bundle_level_bundle_emb(torch.LongTensor(log_seqs).to(self.dev)) 

        # twolevel_seqs = item_level_seqs + bundle_level_seqs
        twolevel_seqs = torch.cat((item_level_seqs, bundle_level_seqs), dim=2)
        twolevel_log_feats = self.log2feats(log_seqs, twolevel_seqs)

        twolevel_final_feat = twolevel_log_feats[:, -1, :]
    
        twolevel_bund_embs = torch.cat((bund_embs[(torch.LongTensor(bundle_indices).to(self.dev))], self.bundle_level_bundle_emb(torch.LongTensor(bundle_indices).to(self.dev))),dim=1)

        logits = twolevel_bund_embs.matmul(twolevel_final_feat.unsqueeze(-1)).squeeze(-1) 

        return logits # preds # (U, I)

'''
Reranking model PopCon (https://github.com/snudatalab/PopCon)
Functions for PopCon are in utils.py.
Jeon, H., Kim, J., Lee, J., Lee, J., Kang, U.: 
Aggregately diversified bundle recommendation via popularity debiasing and configuration-aware reranking. 
In: PAKDD (2023)
'''
class PopCon(object):
    # Initialize the class
    def __init__(self, beta, n):
        super(PopCon, self).__init__()
        self.beta = beta
        self.n = n

    # Get dataset
    def get_dataset(self, n_user, n_item, n_bundle, bundle_item, user_item,
                    user_bundle_trn, user_bundle_vld, vld_user_idx, user_bundle_test,
                    user_bundle_test_mask):
        self.n_user = n_user
        self.n_item = n_item
        self.n_bundle = n_bundle
        self.bundle_item = bundle_item
        self.user_item = user_item
        self.user_bundle_trn = user_bundle_trn
        self.user_bundle_vld = user_bundle_vld
        self.vld_user_idx = vld_user_idx
        self.user_bundle_test = user_bundle_test
        self.user_bundle_test_mask = user_bundle_test_mask

        self.bundle_item_dense_tensor = spy_sparse2torch_sparse(self.bundle_item).to_dense()
        self.max_ent = torch.log2(torch.tensor(self.n_item))

    # Get gains of entropy and coverage
    def delta_bundle_batch(self, cur_item_freq, cand_idx_batch):
        cur_ent = self.get_entropy(cur_item_freq.unsqueeze(0))
        bi = self.bundle_item_dense_tensor[cand_idx_batch.flatten()]
        nex_item_freq = cur_item_freq.repeat(bi.shape[0], 1) + bi
        nex_ent = self.get_entropy(nex_item_freq)
        delta_bundle_ent = nex_ent - cur_ent.repeat(bi.shape[0])
        delta_bundle_ent /= self.max_ent

        cur_cov = self.get_coverage(cur_item_freq.unsqueeze(0))
        nex_cov = self.get_coverage(nex_item_freq)
        delta_bundle_cov = nex_cov - cur_cov.repeat(bi.shape[0])
        return delta_bundle_ent.reshape(
            cand_idx_batch.shape), delta_bundle_cov.reshape(
            cand_idx_batch.shape)

    # Compute entropy
    def get_entropy(self, item_freq):
        prob = item_freq / item_freq.sum(dim=1).unsqueeze(1)
        ent = -prob * torch.log2(prob)
        ent = torch.sum(ent, dim=1)
        return ent

    # Compute coverage
    def get_coverage(self, item_freq):
        num_nz = (item_freq >= 1).sum(dim=1)
        cov = num_nz / item_freq.shape[1]
        return cov

    # Reranking algorithm
    def rerank(self, results, ks):
        cand_scores, cand_idxs = torch.topk(results, dim=1, k=self.n)
        cand_scores_sigmoid = torch.sigmoid(cand_scores)
        cur_item_freq = torch.zeros(self.n_item) + 1e-9
        rec_list = []
        user_batch_size = 1024
        adjust = torch.zeros_like(cand_scores_sigmoid)
        for i in range(1, max(ks)+1):
            user_idx = list(range(cand_scores_sigmoid.shape[0]))
            np.random.shuffle(user_idx)
            rec_list_one = torch.zeros(len(user_idx)).long()
            for batch_idx, start_idx in tqdm(enumerate(range(0, len(user_idx), user_batch_size))):
                end_idx = min(start_idx + user_batch_size, len(user_idx))
                u_batch = user_idx[start_idx:end_idx]
                cand_score_batch = cand_scores_sigmoid[u_batch]
                cand_idxs_batch = cand_idxs[u_batch]
                adjust_batch = adjust[u_batch]
                cand_div_ent_batch, cand_div_cov_batch = self.delta_bundle_batch(cur_item_freq, cand_idxs_batch)
                cand_score_batch_scaled = torch.pow(cand_score_batch, self.beta)
                total_score_batch = cand_score_batch_scaled +\
                                    (1 - cand_score_batch_scaled) * (cand_div_ent_batch + cand_div_cov_batch) +\
                                    adjust_batch
                rec_idx_rel = torch.argmax(total_score_batch, axis=1).unsqueeze(1)
                rec_idx_org = torch.gather(cand_idxs_batch, dim=1, index=rec_idx_rel)
                freq_gain = self.bundle_item[rec_idx_org.squeeze()].sum(0)
                cur_item_freq += torch.tensor(freq_gain).squeeze()
                adjust[u_batch, rec_idx_rel.squeeze()] = -np.inf
                rec_list_one[u_batch] = rec_idx_org.squeeze()
            rec_list.append(rec_list_one.unsqueeze(1))
        rec_list = torch.cat(rec_list, dim=1)
        return rec_list

    # Evaluate the results
    def evaluate_test(self, results, ks, div=True):
        rec_list = self.rerank(results, ks)
        recall_list, map_list, freq_list, ndcg_list = [], [], [], []
        user_idx, _ = np.nonzero(np.sum(self.user_bundle_test, 1))
        test_pos_idx = np.nonzero(self.user_bundle_test[user_idx].toarray())[1]
        pos_idx = torch.LongTensor(test_pos_idx).unsqueeze(1)
        batch_size = 1000
        for batch_idx, start_idx in tqdm(enumerate(range(0, rec_list.shape[0], batch_size))):
            end_idx = min(start_idx + batch_size, rec_list.shape[0])
            result = rec_list[start_idx:end_idx]
            pos = pos_idx[start_idx:end_idx]
            recalls, maps, freqs, ndcgs = evaluate_metrics(result, pos, self.bundle_item, ks=ks, div=True, score=False)
            recall_list.append(recalls)
            map_list.append(maps)
            freq_list.append(freqs)
            ndcg_list.append(ndcgs)
        recalls = list(np.array(recall_list).sum(axis=0) / len(user_idx)) # hit rate
        maps = list(np.array(map_list).sum(axis=0) / len(user_idx))
        freqs = torch.stack(freq_list).sum(dim=0)
        ndcgs = list(np.array(ndcg_list).sum(axis=0) / len(user_idx))
        covs, ents, ginis = evaluate_diversities(freqs, div=div)
        return recalls, maps, covs, ents, ginis, ndcgs