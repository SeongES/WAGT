from loader import MoleculeDataset
from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model_scaling import GNN_graphpred
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

import os
import shutil
from util import count_parameters
import time
import copy 
from gtot_weight import IntermediateLayerGetter, GTOTRegularization

from parser import get_parser
import wandb

criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train(args, model, device, loader, optimizer, gtot_regularization, gtot_weight, target_getter, source_getter):
    model.train()
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        # rep, pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        intermediate_output_s, output_s = source_getter(batch.x, batch.edge_index, batch.edge_attr, batch.batch)  
        intermediate_output_t, output_t = target_getter(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        pred = output_t
        loss_reg_gtot = gtot_regularization(model, intermediate_output_s, intermediate_output_t, batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        is_valid = y**2 > 0
        loss_mat = criterion(pred.double(), (y+1)/2)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid) 
        loss = loss + gtot_weight*loss_reg_gtot
        
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list)


def main():
    args = get_parser()
    print(args)

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    print(dataset)
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    source_model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)

    if args.type=='frozen':
        for name, p_param in model.named_parameters():
            if name != 'style_feature':
                p_param.requires_grad= False

    elif args.type== 'linear_probing':
        for name, param in model.named_parameters():
            param.requires_grad_(False)

            if name == 'style_feature' or 'graph_pred' in name:
                param.requires_grad= True

    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file, type=args.type)
        source_model.from_pretrained(args.input_model_file, type='ft')


    model.to(device)
    source_model.to(device)

    for param in source_model.parameters():
        param.requires_grad = False
    source_model.eval()


    source_getter = IntermediateLayerGetter(source_model, return_layers=['gnn.gnns.4.mlp.2'])
    target_getter = IntermediateLayerGetter(model, return_layers=['gnn.gnns.4.mlp.2'])

    gnn_param = count_parameters(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    gtot_regularization = GTOTRegularization(order=args.gtot_order, args=args)
    
    # Optimizer
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr": args.lr * args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    input_model_file_name = args.input_model_file.split('/')[1][:-4]
    # save model
    os.makedirs(os.path.join(args.save_path, args.tags), exist_ok=True)
    save_model_name = os.path.join(args.save_path, args.tags, f'{args.gnn_type}_{args.dataset}_{input_model_file_name}_{args.runseed}.pt')

    gtot_weight = args.gtot_weight
    
    start = time.time()

    if args.store==True:
        wandb_log = wandb.init(project= 'gtot_0', entity='emilyseong', name=f'{args.dataset}_{input_model_file_name}_{args.runseed}', config=args, tags = [args.tags])

    for epoch in tqdm(range(1, args.epochs+1)):
        # print("====epoch " + str(epoch))

        train(args, model, device, train_loader, optimizer, gtot_regularization, gtot_weight, target_getter, source_getter)
        # print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)
        else:
            train_acc = 0
            # print("ommitting training evaluation")

        val_acc = eval(args, model, device, val_loader)
        test_acc = eval(args, model, device, test_loader)

        if args.store==True:
            wandb_log.log({'epoch': epoch, 'train':train_acc*100, 'val':val_acc*100, 'test':test_acc*100})

    os.makedirs('logs/sider_clintox/gtot0/', exist_ok=True)
    with open('logs/sider_clintox/gtot0/{}_{}.log'.format(args.input_model_file.split('/')[-1][:-4], args.dataset), 'a+') as f:
        f.write(str(args.runseed) + ' ' + str(test_acc*100) +' lr: ' + str(args.lr) + ' gtot_w: ' + str(args.gtot_weight))
        f.write('\n')
if __name__ == "__main__":
    main()
