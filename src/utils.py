'''
***********************************************************************
DEnhancing Bundle Recommendation via Bundle-Item Relations and User Individuality

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: utils.py
- functions for dataset preparation and main ideas of BuDi

Version: 1.0
***********************************************************************
'''
import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict, Counter
from multiprocessing import Process, Queue
from scipy.stats import entropy
from tqdm import tqdm
import pandas as pd
import itertools
import math
import pickle
from sklearn.cluster import KMeans

"""
Calculate global and local popularity

Input:
    * user_train: user histories, 
    * bundle_item_list_dict: bundle-item affiliation dictionary
    * b_l_bundle_freq: bundle level popularity
    * u_i_entropy: entropy of user-item interaction
    * u_b_entropy: entropy of user-bundle interaction
    * i_l_bundle_freq: item level bundle popularity
    * args: predefined arguments

Output:
    * user_sampling_prob: probability of negative sampling for users
"""
def global_local_popularity(user_train, bundle_item_list_dict, b_l_bundle_freq, u_i_entropy, u_b_entropy, i_l_bundle_freq, args):
    global_bundle_freq={key: b_l_bundle_freq[key] * (u_i_entropy/u_b_entropy) + i_l_bundle_freq[key] * (u_b_entropy/u_i_entropy) for key in b_l_bundle_freq}

    global_bundle_freq=normalize_freq(global_bundle_freq)
    user_sampling_prob={}
    for user_key, values in user_train.items():
        global_bundle_freq_copy = global_bundle_freq.copy()
        item_counter = Counter(list(itertools.chain.from_iterable(bundle_item_list_dict[bundle] for bundle in values)))
        item_bundle_score = calculate_bundle_scores(bundle_item_list_dict, item_counter) 
        item_bundle_score=normalize_freq(item_bundle_score)
        local_bundle_view={}
        for bundle in values:
            if bundle in local_bundle_view.keys():
                local_bundle_view[bundle] += 1
            else:
                local_bundle_view[bundle] = 1
        local_bundle_view=normalize_freq(local_bundle_view)
        for key in local_bundle_view.keys():
            item_bundle_score[key] +=local_bundle_view[key]
        item_bundle_score=normalize_freq(item_bundle_score)
        user_sampling_prob[user_key] = {key: global_bundle_freq_copy[key] + args.local_weight*item_bundle_score[key] for key in global_bundle_freq_copy}
        for key in user_sampling_prob[user_key]:
            if user_sampling_prob[user_key][key] <= 0:
                user_sampling_prob[user_key][key] = 1e-10
    return user_sampling_prob

'''
increase the synthesis proportion gradually (curriculum learning)

input:
* epoch: epoch
* max_val: max value of synthesis
* sharp: control increase rate
* start_curri: epoch that starts curriculumn learning

returns:
synthesis propotion 
'''
def curriculum_function(epoch, max_val, sharp, start_curri):
    """
    Curriculum learning
    """
    return max_val*(1-np.exp(- (epoch- start_curri)/sharp))

'''
Popularity-based negative sampling for one user

input:
* bundle_ids: bundle ids
* b_i_bundle_freq: popularity of bundles
* ts: user sequence 

returns:
* sampled negative items
'''
def my_neg_sampling(ts, bundle_ids, b_i_bundle_freq):
    t = np.random.choice(bundle_ids, p=b_i_bundle_freq/b_i_bundle_freq.sum())
    while t == ts:
        t = np.random.choice(bundle_ids, p=b_i_bundle_freq/b_i_bundle_freq.sum())
    return t

'''
Sample function of all user sequences

input:
* user_train: interaction history of each user
* usernum: number of users
* itemnum: number of items
* batch_size: size of batch
* maxlen: maximum length of user sequence
* result_queue: queue to save sampling result
* SEED: random seed
* user_sampling_prob: probabilties of negative sampling for users
'''
def new_sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED, user_sampling_prob):
    def sample(uid): 
        while len(user_train[uid]) <= 1: 
            uid = np.random.randint(1, usernum + 1)

        ts = user_train[uid]
        bundle_ids=np.array(list(user_sampling_prob[uid].keys()))
        u_bundle_freq = np.array(list(user_sampling_prob[uid].values()))
        neg = np.zeros([maxlen], dtype=np.int32)

        if len(ts) <= maxlen:
            seq = np.pad(ts[:-1], (maxlen - len(ts[:-1]), 0), mode='constant', constant_values=0)
            pos = np.pad(ts[1:], (maxlen - len(ts[1:]), 0), mode='constant', constant_values=0)
            for i in range(maxlen - len(ts[1:]), maxlen):
                neg[i] = my_neg_sampling(pos[i], bundle_ids, u_bundle_freq)
        else: # len(ts) >= maxlen
            start_point = np.random.randint(0, len(ts)-maxlen)
            seq = user_train[uid][start_point:start_point+maxlen]
            pos = user_train[uid][start_point+1:start_point+maxlen+1]
            for i in range(0, maxlen):
                neg[i] = my_neg_sampling(pos[i], bundle_ids, u_bundle_freq)

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))
    
'''
Wrap Sampler to get all train sequences

input:
* user_train: interaction history of each user
* usernum: number of users
* itemnum: number of items
* batch_size: size of batch
* maxlen: maximum length of user sequence
* n_workers: number of workers to use in sampling
* alpha: aplha to control I3. Adjusted negative sampling

returns:
* user train sequences
'''
class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, user_sampling_prob, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=new_sample_function, args=(User,
                                                    usernum,
                                                    itemnum,
                                                    batch_size,
                                                    maxlen,
                                                    self.result_queue,
                                                    np.random.randint(2e9),
                                                    user_sampling_prob
                                                    )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

'''
Train and test data partition function

input:
* fname: file name of dataset

returns:
* train and test data with information of dataset
'''
def data_partition(fname):
    usernum = 0
    bundlenum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    
    f = open('./dataset/'+fname+'/user-bundle.txt', 'r')
    for line in f:
        try:
            u, i, _ = line.rstrip().split('\t')
        except:
            u, i = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        bundlenum = max(i, bundlenum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, bundlenum] 

# Evaluate Gini-coefficient
def evaluate_gini(freq, eps=1e-7):
    freq += eps
    freq = np.sort(freq)
    n = freq.shape[0]
    idx = np.arange(1, n + 1)
    return (np.sum((2 * idx - n - 1) * freq)) / (n * np.sum(freq))

'''
Evaluation of predicted results(top 10)

input:
* model: model to evaluate
* dataset: dataset ot evaluate on
* args: model details
* bundle_item_list_dict: bundle-item affiliation dictionary
* item_num: number of items
* topklist: lis of top k

Output:
* results: evaluation score (nDCG, HitRate, Coverage, Entropy, Gini)
* results tensor: recommendation score for all users and bundles.
'''
def evaluate(model, dataset, args, bundle_item_list_dict, item_num, topklist, folder_dir=None, final=False):
    """
    Evaluate the performance of the model
    """
    [train, valid, test, usernum, bundlenum] = copy.deepcopy(dataset)

    max_k = topklist[-1]
    NDCG_list = [0.0] * len(topklist)
    HT_list = [0.0] * len(topklist)
    coverage_list = [0.0] * len(topklist)
    entropy_list = [0.0] * len(topklist)
    gini_list = [0.0] * len(topklist)

    valid_user = 0.0

    total_items_list = [[] for _ in topklist]
    freq_items_list = [np.zeros(item_num + 1) for _ in topklist]
    freq_bundles_list = [np.zeros(bundlenum + 1) for _ in topklist]
    
    result_tensor=[]
    users = range(1, usernum+1)
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: 
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        item_idx = list(set(range(1,bundlenum+1)))

        predictions = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        result_tensor.append(predictions.cpu().detach())
        predictions = predictions[0] 
        _, topk_max_indices = torch.topk(predictions, max_k)
        topk_max_cpu = np.array(item_idx)[topk_max_indices.cpu()]
        
        valid_user += 1

        for idx, k in enumerate(topklist):
            topk_cpu = topk_max_cpu[:k] 

            if test[u][0] in topk_cpu:
                HT_list[idx] += 1
                rank = np.where(topk_cpu == test[u][0])[0][0]
                NDCG_list[idx] += 1 / np.log2(rank + 2)

            for item in topk_cpu:
                total_items_list[idx].extend(bundle_item_list_dict[item])
                freq_bundles_list[idx][item] += 1
                for i in bundle_item_list_dict[item]:
                    freq_items_list[idx][i] += 1
    results_tensor=torch.cat(result_tensor, dim=0).to('cpu')

    for idx, k in enumerate(topklist):
        coverage_list[idx] = len(Counter(total_items_list[idx]).keys()) / item_num
        p_k = (freq_items_list[idx] / freq_items_list[idx].sum())[1:]
        entropy_list[idx] = -np.sum(p_k * np.log2(p_k + 1e-9))
        gini_list[idx] = evaluate_gini(freq_items_list[idx][1:])
        

    if final and folder_dir:
        for idx, k in enumerate(topklist):
            with open(file=f"{folder_dir}/{args.dataset}_freq_items_k{k}.pkl", mode='wb') as f:
                pickle.dump(freq_items_list[idx], f)
            with open(file=f"{folder_dir}/{args.dataset}_freq_bundles_k{k}.pkl", mode='wb') as f:
                pickle.dump(freq_bundles_list[idx], f)

    results = {}
    for idx, k in enumerate(topklist):
        results.update({
            f"NDCG@{k}": NDCG_list[idx] / valid_user,
            f"HT@{k}": HT_list[idx] / valid_user,
            f"Coverage@{k}": coverage_list[idx],
            f"Entropy@{k}": entropy_list[idx],
            f"Gini@{k}": gini_list[idx]
        })

    return results, results_tensor

"""
Bundle interaction frequency dictionary

Input:
* datasetname: name of dataset

Output:
* i_l_bundle_freq_dict: item level bundle frequency
* b_l_bundle_freq_dict: bundle level bundle frequency
* u_b_entropy; entropy of user-bundle interaction 
* u_i_entropy: entropy of user-item interaction
"""
def generate_bundle_freq_dict(datasetname):
    interaction = pd.read_csv("./dataset/"+datasetname+"/user-bundle.txt", sep="\t", header=None)
    interaction=interaction[[0,1]]
    interaction.columns=['user_id', 'bundle_id']
    b_l_bundle_freq_dict = Counter(interaction.iloc[:,1])
    u_b_entropy = entropy(list(b_l_bundle_freq_dict.values()), base=2)
    affil = pd.read_csv("./dataset/"+datasetname+"/bundle-item.txt", sep="\t", header=None)
    affil.columns=["bundle_id", "item_id"]
    u_b_i=interaction.merge(affil, on='bundle_id', how='inner')
    item_freq_dict = Counter(u_b_i.iloc[:,2])
    u_i_entropy=entropy(list(item_freq_dict.values()), base=2)
    affil['item_freq']=1
    for i in range(len(affil)):
        affil.iloc[i, 2] = item_freq_dict[affil.iloc[i, 1]]
    bundle_freq=affil.groupby("bundle_id")["item_freq"].mean()
    i_l_bundle_freq_dict = bundle_freq.to_dict()
    return i_l_bundle_freq_dict, b_l_bundle_freq_dict, u_b_entropy, u_i_entropy

"""
Cluster bundles based on embeddings

Input: 
* i_l_bundle_freq_dict: item level bundle frequency
* b_l_bundle_freq_dict: bundle level bundle frequency
* model: pretrained model
* bundlenum: number of bundles
* u_b_entropy; entropy of user-bundle interaction 
* u_i_entropy: entropy of user-item interaction
* args: predefined arguments

Output:
* clusters: cluster dictionary (key: cluster id, value: [bundle_id, bundle's popularity])
* assignments: assigned cluster ids
"""
def cluster_bundle(i_l_bundle_freq, b_l_bundle_freq, model, bundlenum, u_b_entropy, u_i_entropy, args):
    bundle_embeddings = model.bundle_level_bundle_emb.weight + model.bund_emb()
    kmeans = KMeans(n_clusters=args.num_cluster, init='k-means++', n_init='auto').fit(bundle_embeddings[1:].detach().cpu().numpy())
    assignments = kmeans.labels_
    clusters = {k_i: [[],[]] for k_i in range(args.num_cluster)}
    for i in range(1, bundlenum+1):
        clusters[assignments[i-1]][0].append(i)
        clusters[assignments[i-1]][1].append((u_i_entropy/u_b_entropy)*(b_l_bundle_freq[i]) + (u_b_entropy/u_i_entropy)*(i_l_bundle_freq[i]))
    return  clusters, assignments

"""
Preference-aware replacement

input:
* pos: original sequence
* replace_ratio: replacement ratio
* maxlen: max length of sequence
* clusters: cluster information dictionary
* assignments: assigned cluster ids

return: 
* seq: replaced sequence
"""
def pseudo_groundtruth_replace(replace_ratio, pos, maxlen, clusters, assignments):
    samplenum = int(maxlen* replace_ratio)
    for i in range(len(pos)):
        for j in random.sample(range(maxlen), samplenum): 
            if pos[i][j] == 0:
                continue
            if len(clusters[assignments[pos[i][j]-1]][0]) == 0:
                continue
            freq = np.array(clusters[assignments[pos[i][j]-1]][1])
            freq[freq == 0] = max(freq)
            inv_freq= 1/(freq)
            selected_index = np.random.choice(range(len(clusters[assignments[pos[i][j]-1]][0])), p=(inv_freq/ inv_freq.sum()))
            
            pos[i][j] = clusters[assignments[pos[i][j]-1]][0][selected_index]
            
            clusters[assignments[pos[i][j] - 1]][0].pop(selected_index) 
            clusters[assignments[pos[i][j] - 1]][1].pop(selected_index)  

    return pos

"""
min-max scaling

Input: 
* bundle_freq_dict: bundle frequency dictionary

Output: 
* normalized_bundle_freq: normalized bundle frequency
"""
def normalize_freq(bundle_freq_dict, level=None):
    values = list(bundle_freq_dict.values())
    min_value = min(values)
    max_value = max(values)
    if level =="item":
        normalized_bundle_freq = {
            key: 0.2 + ((value - min_value) / (max_value - min_value))
            for key, value in bundle_freq_dict.items()
        }
    elif level=="bundle":
        normalized_bundle_freq = {
            key: 0.2 + ((value - min_value) / (max_value - min_value))
            for key, value in bundle_freq_dict.items()
        }
    else:
        normalized_bundle_freq = {
            key: 0.2 + ((value - min_value) / (max_value - min_value+1e-10)) 
            for key, value in bundle_freq_dict.items()
        }
    return normalized_bundle_freq

"""
Get top-k items in each user sequence

Input:
* logs_seqs_items: user history (Bundles decomposed into items)
* k: top k

Output:
* top_items: top k popular items in each user sequence
"""
def get_top_k_items(logs_seqs_items, k):
    top_items = []

    for seq in logs_seqs_items:
        flattened_seq = seq.flatten()
        non_zero_items = flattened_seq[flattened_seq != 0]  
        
        unique_items, counts = torch.unique(non_zero_items, return_counts=True)
        if k < len(counts):
            top_k = unique_items[torch.topk(counts, k).indices]
            top_items.append(top_k)
        else:
            k=1
            top_k = unique_items[torch.topk(counts, k).indices]
            top_items.append(top_k)
    
    return top_items

"""
Mask top-k items in each user sequence

Input:
* logs_seqs_items: original user sequences
* top_items: top k popular items for each user
* mask_ratio: masking ratio

Output: 
* masked_logs_seqs: masked user sequences
"""
def mask_top_items(logs_seqs_items, top_items, mask_ratio):
    mask_num= int(logs_seqs_items[0].shape[0] * mask_ratio)
    masked_logs_seqs = logs_seqs_items.clone() 
    for i, seq in enumerate(logs_seqs_items):
        random_bundle_indices = torch.randperm(seq.shape[0])[:mask_num]

        for random_bundle_idx in random_bundle_indices:
            masked_logs_seqs[i][random_bundle_idx] = torch.where(
                torch.isin(masked_logs_seqs[i][random_bundle_idx], top_items[i]), 
                torch.tensor(0, device='cuda'), 
                masked_logs_seqs[i][random_bundle_idx]
            )
            
    return masked_logs_seqs

"""
Item-level masking

Input:
* seq: user sequences
* bundle_item_list_dict: bundle-item affiliation dictionary
* args: predefined arguments

Output: 
* masked_logs_seqs: masked user sequences
"""
def mask_item_level(seq, bundle_item_list_dict, args):
    logs_seqs_items= [[bundle_item_list_dict[item] if item in bundle_item_list_dict else [0] for item in s ] for s in seq]
    max_len = max(max(len(inner_list) for inner_list in seq) for seq in logs_seqs_items)
    padded_logs_seqs = [
    [inner_list + [0] * (max_len - len(inner_list)) for inner_list in seq] 
    for seq in logs_seqs_items
    ]
    top_items = get_top_k_items(torch.LongTensor(padded_logs_seqs).to(args.device), args.top_items_per_seq) #
    masked_logs_seqs = mask_top_items(torch.LongTensor(padded_logs_seqs).to(args.device), top_items, args.mask_ratio)
    return masked_logs_seqs

"""
Calculate global item-level bundle scores

Input:
* bundle_item_list_dict: bundle-item affiliation dictionary
* item_counter: item popularity

Output:
* bundle_scores: global item-level bundle popularity
"""
def calculate_bundle_scores(bundle_item_list_dict, item_counter):
    bundle_scores = {}

    for bundle_id, item_list in bundle_item_list_dict.items():
        score = sum(item_counter.get(item_id, 0) for item_id in item_list)
        bundle_scores[bundle_id] = score

    return bundle_scores

# format of logs (function for PopCon)
def form_content(epoch, losses, recalls, maps, covs, ents, ginis, ndcgs, elapses):
    content = f'{epoch:7d}| {losses[0]:10.4f} {losses[1]:10.4f} |'
    content += f' recall'
    for item in recalls:
        content += f' {item:.4f} '
    content += '|'
    content += f' map'
    for item in maps:
        content += f' {item:.4f} '
    content += '|'
    content += f' cov'
    for item in covs:
        content += f' {item:.4f} '
    content += '|'
    content += f' ent'
    for item in ents:
        content += f' {item:.4f} '
    content += '|'
    content += f' gini'
    for item in ginis:
        content += f' {item:.4f} '
    content += '|'
    content += f' ndcg'
    for item in ndcgs:
        content += f' {item:.4f} '
    content += f'| {elapses[0]:7.1f} {elapses[1]:7.1f} {elapses[2]:7.1f} |'
    return content

# Load pickle file (function for PopCon)
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

# Aggregate ground-truth targets and negative targets (function for PopCon)
def user_filtering(csr, neg):
    idx, _ = np.nonzero(np.sum(csr, 1))
    pos = np.nonzero(csr[idx].toarray())[1]
    pos = pos[:, np.newaxis]
    neg = neg[idx]
    arr = np.concatenate((pos, neg), axis=1)
    return arr, idx

# Load dataset (function for PopCon)
def load_mat_dataset(dataname):
    path = f'./dataset/{dataname}'
    user_bundle_trn = load_obj(f'{path}/train.pkl')
    user_bundle_vld = load_obj(f'{path}/valid.pkl')
    user_bundle_test = load_obj(f'{path}/test.pkl')
    user_item = load_obj(f'{path}/user_item.pkl')
    bundle_item = load_obj(f'{path}/bundle_item.pkl')
    user_bundle_neg = np.array(load_obj(f'{path}/neg.pkl'))
    n_user, n_item = user_item.shape
    n_bundle, _ = bundle_item.shape

    user_bundle_test_mask = user_bundle_trn + user_bundle_vld

    # filtering
    user_bundle_vld, vld_user_idx = user_filtering(user_bundle_vld,
                                                   user_bundle_neg)

    return n_user, n_item, n_bundle, bundle_item, user_item,\
           user_bundle_trn, user_bundle_vld, vld_user_idx, user_bundle_test,\
           user_bundle_test_mask

# Set random seed (function for PopCon)
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Get evaluation metrics (function for PopCon)
def get_metrics(pred_rank, pos_idx, k, bundle_item, div: bool):
    pos = torch.eq(pred_rank, pos_idx).float()
    # ndcg
    nonzero_indices = torch.nonzero(pos[:,:k])
    ndcg = 1 / torch.log2(nonzero_indices[:, 1].float() + 2)
    ndcg=ndcg.sum().item()
    # recall and map
    recall = pos[:, :k].sum().item()
    idxs = torch.nonzero(pos[:, :k], as_tuple=True)[1]
    map = (1 / (idxs + 1).float()).sum().item()
    # frequency
    if div:
        freq = torch.tensor(
            bundle_item[pred_rank[:, :k].flatten().cpu()].sum(axis=0)).squeeze()
    else:
        freq = torch.zeros(bundle_item.shape[1])
    return recall, map, freq, ndcg

# Evaluate diversities (function for PopCon)
def evaluate_diversities(freqs, div: bool):
    covs, ents, ginis = [], [], []
    if div:
        for freq in freqs:
            cov = torch.count_nonzero(freq).item()
            covs.append(cov/freqs.shape[1])
            prob = freq/freq.sum()
            prob = prob.clamp(min=1e-9)
            ent = -prob*torch.log2(prob)
            ent = torch.sum(ent)
            ents.append(ent)
            gini = evaluate_gini(freq.float()).item()
            ginis.append(gini)
        return covs, ents, ginis
    else:
        covs = [0., 0., 0.]
        ents = [0., 0., 0.]
        ginis = [0., 0., 0.]
        return covs, ents, ginis

# Evaluate performance in terms of recalls, maps, and frequencies (function for PopCon)
def evaluate_metrics(pred, pos_idx, bundle_item, ks: list, div: bool, score=True):
    recalls, maps, freqs, ndcgs = [], [], [], []
    if score:
        pred_rank = torch.topk(pred, max(ks), dim=1, sorted=True)[1]
    else:
        pred_rank = pred
    for k in ks:
        recall, map, freq, ndcg = get_metrics(pred_rank, pos_idx, k, bundle_item, div)
        recalls.append(recall)
        maps.append(map)
        freqs.append(freq)
        ndcgs.append(ndcg)
    return recalls, maps, torch.stack(freqs), ndcgs

# Transform scipy sparse tensor to torch sparse tensor (function for PopCon)
def spy_sparse2torch_sparse(data):
    samples = data.shape[0]
    features = data.shape[1]
    values = data.data
    coo_data = data.tocoo()
    indices = torch.LongTensor(np.array([coo_data.row, coo_data.col]))
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return t

