import torch
from sklearn.metrics import f1_score
import dgl
import sys
sys.path.append('../../')
import numpy as np
from utils_han import load_data, EarlyStopping
import torch.nn.functional as F
from model_hetero import HAN, HAN_freebase
import time

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1


def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1


def main(args):
    g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask, meta_paths = load_data(args, args.dataset, feat_type=0)

    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()
    from utils_han import setup

    args = args.__dict__
    args = setup(args)
    features = features.to(args['device'])
    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])
    if args['dataset'] == 'Freebase' or args['dataset'] == 'taobao':
        # Add a fc layer to calculate sparse
        model = HAN_freebase(
            meta_paths=meta_paths,
            in_size=features.shape[1],
            hidden_size=args['hidden_units'],
            out_size=num_classes,
            num_heads=args['num_heads'],
            dropout=args['dropout']).to(args['device'])
    else:
        model = HAN(
            meta_paths=meta_paths,
            in_size=features.shape[1],
            hidden_size=args['hidden_units'],
            out_size=num_classes,
            num_heads=args['num_heads'],
            dropout=args['dropout']).to(args['device'])

    g = g.to(args['device'])

    stopper = EarlyStopping(patience=args['patience'])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    for epoch in range(args['num_epochs']):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, features, labels, val_mask, loss_fcn)
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
              'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

        if early_stop:
            break

    stopper.load_checkpoint(model)
    t1 = time.time()

    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, features, labels, test_mask, loss_fcn)
    t2 = time.time()
    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1))
    print('inference time', t2-t1)
    return test_micro_f1, test_macro_f1

if __name__ == '__main__':
    import argparse


    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dataset', type=str, default='DBLP',
                        choices=['DBLP', 'ACM', 'taobao'])
    parser.add_argument('--path', type=str, default='../../data/',
                        choices=['DBLP', 'ACM', 'taobao'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--run_num', type=int, default=5)
    parser.add_argument('--run', type=int, default=5)
    parser.add_argument('--run_num_seed', type=int, default=3)
    parser.add_argument('--train_ratio', type=float, default=0.01)
    parser.add_argument('--valid_ratio', type=float, default=0.01)
    args = parser.parse_args()#.__dict__

    print(args)
    micro_res_list = []
    macro_res_list = []
    for i in range(args.run_num):
        for seed in range(args.run_num_seed):
            args.run = i + 1
            torch.cuda.manual_seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            micro_f1, macro_f1 = main(args)
            micro_res_list.append(micro_f1)
            macro_res_list.append(macro_f1)
    print(f'test avg micro f1: {np.mean(micro_res_list)}, {np.std(micro_res_list)}')   
    print(f'test avg macro f1: {np.mean(macro_res_list)}, {np.std(macro_res_list)}')   
