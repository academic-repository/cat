import argparse
import os
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from transformers import *
from torch.autograd import Variable
from torch.utils.data import Dataset
from tqdm import tqdm 

from read_data import *
from normal_bert import ClassificationBert

parser = argparse.ArgumentParser(description='PyTorch Base Models')

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=4, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--batch-size-u', default=24, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--lrmain', '--learning-rate-bert', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate for bert')
parser.add_argument('--lrlast', '--learning-rate-model', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for models')

parser.add_argument('--gpu', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--n-labeled', type=int, default=20,
                    help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=200,
                    help='Number of labeled data')


parser.add_argument('--mix-option', default=False, type=bool, metavar='N',
                    help='mix option')
parser.add_argument('--train_aug', default=False, type=bool, metavar='N',
                    help='aug for training data')


parser.add_argument('--model', type=str, default='bert-base-uncased',
                    help='pretrained model')

parser.add_argument('--mode', type=str, default='single',
                    help='single stentence or pairs')

parser.add_argument('--data-path', type=str, default='yahoo_answers_csv/',
                    help='path to data folders')




args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu num: ", n_gpu)

best_acc = 0


def main():
    global best_acc
    if args.mode == 'single':
        train_labeled_set, val_set, test_set, n_labels = get_data(
            args.data_path, args.n_labeled)
    else:
        train_labeled_set, val_set, test_set, n_labels = get_data_pair(
            args.data_path, args.n_labeled)
    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True)
    val_loader = Data.DataLoader(
        dataset=val_set, batch_size=32, shuffle=False)
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=32, shuffle=False)


    model = ClassificationBert(n_labels).cuda()
#     model = nn.DataParallel(model)
    optimizer = AdamW(
        [
            {"params": model.bert.parameters(), "lr": args.lrmain},
            {"params": model.linear.parameters(), "lr": args.lrlast},
        ])


    criterion = nn.CrossEntropyLoss()

    test_accs = []

    for epoch in range(args.epochs):
        train(labeled_trainloader, model, optimizer, criterion, epoch)

        val_loss, val_acc = validate(
            val_loader, model, criterion, epoch, mode='Valid Stats')
        print("epoch {}, val acc {}, val_loss {}".format(
            epoch, val_acc, val_loss))

        if val_acc >= best_acc:
            best_acc = val_acc
            test_loss, test_acc = validate(
                test_loader, model, criterion, epoch, mode='Test Stats ')
            test_accs.append(test_acc)
            print("epoch {}, test acc {},test loss {}".format(
                epoch, test_acc, test_loss))

    print('Best val_acc:')
    print(best_acc)

    print('Test acc:')
    print(test_accs)
    
    test_accs_str = [str(i) for i in test_accs]
    test_accs_str = '\t'.join(test_accs_str)
    best_acc_str = str(best_acc)
    with open('./data/record/' + '1121_v2_' + 'batch_size_' + str(args.batch_size) + '_' + str(args.data_path).strip('./data/').strip('/') + '_' + str(args.n_labeled)
              + '_n_examples'
              +'.txt', 'a') as fw:
        fw.write('val_acc' + '\t' + 'test_acc' + '\n')
        fw.write(best_acc_str + '\t' + test_accs_str + '\n')


def validate(valloader, model, criterion, epoch, mode):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0

        for batch_idx, (inputs, attention_mask, token_type_ids, targets, length) in enumerate(valloader):
            inputs, attention_mask, token_type_ids, targets = inputs.to(device), attention_mask.to(device), token_type_ids.to(device), targets.to(device)
            if args.mode == 'single':
                outputs = model(inputs, attention_mask=attention_mask)
            else:
                outputs = model(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)

            correct += (np.array(predicted.cpu()) ==
                        np.array(targets.cpu())).sum()
            loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

        acc_total = correct/total_sample
        loss_total = loss_total/total_sample

    return loss_total, acc_total


def train(labeled_trainloader, model, optimizer, criterion, epoch):
    model.train()

    for batch_idx, (inputs, attention_mask, token_type_ids, targets, length) in tqdm(enumerate(labeled_trainloader)):
        inputs, attention_mask, token_type_ids, targets = inputs.to(device), attention_mask.to(device), token_type_ids.to(device), targets.to(device)
        if args.mode == 'single':
            outputs = model(inputs, attention_mask=attention_mask)
        else:
            outputs = model(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        print('epoch {}, step {}, loss {}'.format(
            epoch, batch_idx, loss.item()))
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()