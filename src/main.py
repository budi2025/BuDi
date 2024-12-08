'''
***********************************************************************
Enhancing Bundle Recommendation via Bundle-Item Relations and User Individuality

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: main.py
- A main class for training and evaluation of BuDi.

Version: 1.0.0
***********************************************************************
'''
import os
import time
import torch
import argparse
from tqdm import tqdm
import pickle
from torch.utils.tensorboard import SummaryWriter
from model import *
from utils import *

from collections import defaultdict

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="chess", type=str)
parser.add_argument('--train_dir', default="test1", type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=500, type=int)
parser.add_argument('--hidden_units', default=256, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs1', default=200, type=int)
parser.add_argument('--num_epochs2', default=300, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.4, type=float)
parser.add_argument('--l2_emb', default=0.0001, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--gpu', default="1", type=str)
parser.add_argument('--seed', default=0, type=int) # None
parser.add_argument('--topklist', nargs='+', default=[1, 3, 5, 10, 20], type=int, help='List of integers')
parser.add_argument('--eval_inter', default=10, type=int) # None
parser.add_argument('--num_cluster', default=30, type=int) # None
parser.add_argument('--pos_re', default=5, type=int) # None
parser.add_argument('--seq_re', default=7, type=int) # None
parser.add_argument('--masking_freq', default=9, type=int) # None
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=10000)
parser.add_argument('--local_weight', default=1.0, type=float) 
parser.add_argument('--replace_ratio', default=0.4, type=float)
parser.add_argument('--top_items_per_seq', default=1, type=int) 
parser.add_argument('--mask_ratio', default=0.2, type=float) 

args = parser.parse_args()


folder_dir="log/"+args.dataset+"/"+args.dataset + '_' + args.train_dir
if not os.path.isdir(folder_dir):
    os.makedirs(folder_dir, exist_ok=True)
with open(os.path.join(folder_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()
writer = SummaryWriter('runs/'+args.dataset + '_' + args.train_dir)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  

if args.seed is not None:
    deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
 

if __name__ == '__main__':

    print(args)

    keys_order = ["HT", "NDCG", "Coverage", "Entropy", "Gini"]

    b_i_file = open('./dataset/'+args.dataset+'/bundle-item.txt', 'r')
    item_num=0
    bundle_item_list_dict= defaultdict(list)
    for line in b_i_file:
        bundle_id, item_id = line.rstrip().split('\t')
        bundle_id = int(bundle_id)
        item_id = int(item_id)
        item_num = max(item_id, item_num)
        bundle_item_list_dict[bundle_id].append(item_id)

    dataset = data_partition(args.dataset)
    i_l_bundle_freq_dict, b_l_bundle_freq_dict, u_b_entropy, u_i_entropy = generate_bundle_freq_dict(args.dataset)
    i_l_bundle_freq = normalize_freq(i_l_bundle_freq_dict, "item")
    b_l_bundle_freq = normalize_freq(b_l_bundle_freq_dict,"bundle")

    [user_train, user_valid, user_test, usernum, bundlenum] = dataset
    num_batch = len(user_train) // args.batch_size 
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    f = open(os.path.join(folder_dir, 'log.txt'), 'w')
    
    user_sampling_prob = global_local_popularity(user_train, bundle_item_list_dict, b_l_bundle_freq, u_i_entropy, u_b_entropy, i_l_bundle_freq, args)

    sampler = WarpSampler(user_train, usernum, bundlenum, user_sampling_prob, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, bundlenum, item_num, bundle_item_list_dict, args).to(args.device) 
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass 
    model.train() 
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            print("load success")
        except: 
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
            
    if args.inference_only:
        inf = open(os.path.join(folder_dir, 'inference_log.txt'), 'w')
        model.eval()
        print("start eval")
        t_test = evaluate(model, dataset, args, bundle_item_list_dict, item_num, args.topklist)          
        f.write(str(t_test)+"\n") 
        for key in keys_order:
            for k, value in t_test.items():
                if k.startswith(key):
                    inf.write(f"{k}: {value}\n")
        inf.close()
    
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()
    best_performance=0.0
    early_stop=0
    best_epoch=0
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs1 + 1)):
        if args.inference_only: break 
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'): 
            augment=False
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            if step % args.pos_re == 0:
                masked_seq = mask_item_level(seq, bundle_item_list_dict, args)
                augment=True
                pos_logits, neg_logits = model(u, seq, pos, neg, augment, masked_seq)
            else:
                pos_logits, neg_logits = model(u, seq, pos, neg, augment)
            
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            adam_optimizer.zero_grad()

            loss = bce_criterion(pos_logits, pos_labels)
            loss += bce_criterion(neg_logits, neg_labels)

            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.bundle_level_bundle_emb.parameters(): loss += args.l2_emb * torch.norm(param)

            loss.backward()
            adam_optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            t_test, results_tensor = evaluate(model, dataset, args, bundle_item_list_dict, item_num, args.topklist) 
            f.write(f"epoch{epoch}")
            f.write(str(t_test)+"\n") 
            for key in keys_order:
                for k, value in t_test.items():
                    if k.startswith(key):
                        f.write(f"{k}: {value}\n")  
            f.flush()
            t0 = time.time()

            if t_test[f"NDCG@{args.topklist[-2]}"] > best_performance:
                fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(args.num_epochs2, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                torch.save(model.state_dict(), os.path.join(folder_dir, fname))
                best_test_dict= t_test
                best_performance = t_test[f"NDCG@{args.topklist[0]}"]
                best_epoch = epoch
                early_stop=0
            else:
                early_stop+=1

            if early_stop > 10:
                t0 = time.time()
                model.train()
                f.write('###Early stop####')
                f.write("best epoch"+str(best_epoch)+'\n'+"best performance:"+str(best_test_dict) + '\n')
                f.flush()
                break

            model.train()

    print("end pre-training")
    f.write("end first training\n")

    model.load_state_dict(torch.load(os.path.join(folder_dir, fname), map_location=torch.device(args.device)))
    clusters, assignments = cluster_bundle(i_l_bundle_freq, b_l_bundle_freq, model, bundlenum, u_b_entropy, u_i_entropy, args)
    model = SASRec(usernum, bundlenum, item_num, bundle_item_list_dict, args).to(args.device)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass
    T = 0.0
    t0 = time.time()
    best_performance=0.0
    early_stop=0
    best_epoch=0
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs2 + 1)):
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'): 
            augment=False
            u, seq, pos, neg = sampler.next_batch() 
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            if step % args.pos_re == 0:
                ratio=curriculum_function(epoch, args.replace_ratio, 100, 1)
                pos = pseudo_groundtruth_replace(ratio, pos, args.maxlen, clusters, assignments)
            if step % args.seq_re == 0:
                seq = pseudo_groundtruth_replace(ratio, seq, args.maxlen, clusters, assignments)
            if step % args.masking_freq == 0:
                masked_seq = mask_item_level(seq, bundle_item_list_dict, args)
                augment=True
                pos_logits, neg_logits = model(u, seq, pos, neg, augment, masked_seq)
            else:
                pos_logits, neg_logits = model(u, seq, pos, neg)

            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            adam_optimizer.zero_grad()

            loss = bce_criterion(pos_logits, pos_labels)
            loss += bce_criterion(neg_logits, neg_labels)
    
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.bundle_level_bundle_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()

        if epoch % args.eval_inter == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            t_test, results_tensor = evaluate(model, dataset, args, bundle_item_list_dict, item_num, args.topklist) 
            f.write(f"epoch{epoch}")
            f.write(str(t_test)+"\n") 
            for key in keys_order:
                for k, value in t_test.items():
                    if k.startswith(key):
                        f.write(f"{k}: {value}\n")   
            ##popcon reranking
            set_seed(args.seed)
            n_user, n_item, n_bundle, bundle_item, user_item,\
            user_bundle_trn, user_bundle_vld, vld_user_idx, user_bundle_test,\
            user_bundle_test_mask = load_mat_dataset(args.dataset)
            ks = [1, 3, 5, 10, 20]
            
            print('=========================== LOADED ===========================')
            popcon = PopCon(beta=args.beta, n=args.n)
            popcon.get_dataset(n_user, n_item, n_bundle, bundle_item, user_item,
                                user_bundle_trn, user_bundle_vld, vld_user_idx,
                                user_bundle_test, user_bundle_test_mask)
            test_start_time = time.time()
            f.write('popcon reranking\n')
            test_recalls, test_maps, test_covs, test_ents, test_ginis, test_ndcgs = popcon.evaluate_test(results_tensor, ks, div=True)
            test_elapsed = time.time() - test_start_time

            test_content = form_content(0, [0, 0],
                                        test_recalls, test_maps, test_covs, test_ents, test_ginis, test_ndcgs,
                                        [0, test_elapsed, test_elapsed])
            f.write(f'beta={args.beta}\n')
            for item in test_recalls:
                f.write(f' {item:.10f}\n')
            for item in test_ndcgs:
                f.write(f' {item:.10f}\n')
            for item in test_covs:
                f.write(f' {item:.10f}\n')
            for item in test_ents:
                f.write(f' {item:.10f}\n')
            for item in test_ginis:
                f.write(f' {item:.10f}\n') 
            f.flush()
            t0 = time.time()

            if t_test[f"Entropy@{args.topklist[-2]}"] > best_performance:
                fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(args.num_epochs2, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                torch.save(model.state_dict(), os.path.join(folder_dir, fname))
                best_test_dict= t_test
                best_performance = t_test[f"NDCG@{args.topklist[-2]}"]
                best_epoch = epoch
                early_stop=0
            else:
                early_stop+=1

            if early_stop > 10:
                t0 = time.time()
                model.train()
                f.write('###Early stop####')
                f.write("best epoch"+str(best_epoch)+'\n'+"best performance:"+str(best_test_dict) + '\n')
                f.flush()
                break

            model.train()
                
    print("best performance %.4f at %d epoch"%(best_performance, best_epoch))
    f.write("best epoch"+str(best_epoch)+'\n'+"best performance:"+str(best_test_dict) + '\n')
    f.flush()
    f.close()
    sampler.close()
    print("Done")
