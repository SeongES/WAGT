import argparse
from loader import BioDataset
from dataloader import DataLoaderFinetune
from splitters import random_split, species_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model_scaling import GNN_graphpred
from sklearn.metrics import roc_auc_score

import pandas as pd

import os
import shutil
from util import count_parameters
import time
import copy 
from gtot_weight import IntermediateLayerGetter, GTOTRegularization
import wandb

criterion = nn.BCEWithLogitsLoss()

def train(args, model, device, loader, optimizer, gtot_regularization, gtot_weight, target_getter, source_getter):
    model.train()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        # rep, pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        intermediate_output_s, output_s = source_getter(batch)  
        intermediate_output_t, output_t = target_getter(batch)

        pred = output_t
        loss_reg_gtot = gtot_regularization(model, intermediate_output_s, intermediate_output_t, batch)
        y = batch.go_target_downstream.view(pred.shape).to(torch.float64)

        optimizer.zero_grad()
        loss = criterion(pred.double(), y)
        loss = loss + gtot_weight*loss_reg_gtot
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.go_target_downstream.view(pred.shape).detach().cpu())
        y_scores.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
            roc_list.append(roc_auc_score(y_true[:,i], y_scores[:,i]))
        else:
            roc_list.append(np.nan)

    return sum(roc_list)/len(roc_list)


def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--type', type=str, default='eff', help='ft or eff or frozen')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--pred_lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1, help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0, help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300, help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean", help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last", help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'tox21', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type=str, default = "species", help='Random or species split')
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--gtot_order', default=1, type=int, help='A^{k} in graph topology OT')
    parser.add_argument('--save_path', default='saved_models', type=str, help="directory to save the finetuned model")
    parser.add_argument('--shot_number', type=int, default = None, help='Number of shots')
    parser.add_argument('--gtot_weight', type=float, default = 0.001, help='gtot weight')
    parser.add_argument('--tags', type=str, default = 'tmp', help='tag for wandb')
    parser.add_argument('--store', action= 'store_true', help='store at wandb')
    args = parser.parse_args()



    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)


    root_supervised = 'dataset/supervised'

    dataset = BioDataset(root_supervised, data_type='supervised')

    print(dataset)

    if args.split == "random":
        print("random splitting")
        train_dataset, valid_dataset, test_dataset = random_split(dataset, seed = args.seed) 
    elif args.split == "species":
        trainval_dataset, test_dataset = species_split(dataset)
        train_dataset, valid_dataset, _ = random_split(trainval_dataset, seed = args.seed, frac_train=0.85, frac_valid=0.15, frac_test=0)
        test_dataset_broad, test_dataset_none, _ = random_split(test_dataset, seed = args.seed, frac_train=0.5, frac_valid=0.5, frac_test=0)
        print("species splitting")
    else:
        raise ValueError("Unknown split name.")

    train_loader = DataLoaderFinetune(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoaderFinetune(valid_dataset, batch_size=10*args.batch_size, shuffle=False, num_workers = args.num_workers)
    
    if args.split == "random":
        test_loader = DataLoaderFinetune(test_dataset, batch_size=10*args.batch_size, shuffle=False, num_workers = args.num_workers)
    else:
        ### for species splitting
        test_easy_loader = DataLoaderFinetune(test_dataset_broad, batch_size=10*args.batch_size, shuffle=False, num_workers = args.num_workers)
        # test_hard_loader = DataLoaderFinetune(test_dataset_none, batch_size=10*args.batch_size, shuffle=False, num_workers = args.num_workers)

    num_tasks = len(dataset[0].go_target_downstream)

    print(train_dataset[0])

    #set up model
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


    source_getter = IntermediateLayerGetter(source_model, return_layers=['gnn.gnns.4.mlp.3'])
    target_getter = IntermediateLayerGetter(model, return_layers=['gnn.gnns.4.mlp.3'])

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
        name = args.input_model_file.split('/')[-1][:-4]
        wandb_log = wandb.init(project= 'ppi', entity='emilyseong', name=f'{name}_{args.runseed}', config=args, tags = [args.tags])

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))

        train(args, model, device, train_loader, optimizer, gtot_regularization, gtot_weight, target_getter, source_getter)
        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)
        else:
            # print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader)

        if args.split == "random":
            test_acc_random = eval(args, model, device, test_loader)
        else:
            test_acc = eval(args, model, device, test_easy_loader)

        if args.store==True:
            wandb_log.log({'epoch': epoch, 'train':train_acc*100, 'val':val_acc*100, 'test':test_acc*100})
        print("")

    os.makedirs('logs/wagt/', exist_ok=True)
    with open('logs/wagt/{}.log'.format(args.input_model_file.split('/')[-1][:-4]), 'a+') as f:
        f.write(str(args.runseed) + ' ' + str(test_acc*100) +' lr: ' + str(args.lr) + ' gtot_w: ' + str(args.gtot_weight))
        f.write('\n')

if __name__ == "__main__":
    main()
