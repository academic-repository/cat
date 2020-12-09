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
from mixtext import MixText

parser = argparse.ArgumentParser(description='PyTorch MixText')

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=4, type=int, metavar='N',
                    help='train batchsize')
# parser.add_argument('--batch-size-u', default=24, type=int, metavar='N',
#                     help='train batchsize')

parser.add_argument('--lrmain', '--learning-rate-bert', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate for bert')
parser.add_argument('--lrlast', '--learning-rate-model', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for models')

parser.add_argument('--gpu', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--n-labeled', type=int, default=20,
                    help='number of labeled data')

parser.add_argument('--un-labeled', default=5000, type=int,
                    help='number of unlabeled data')

parser.add_argument('--val-iteration', type=int, default=200,
                    help='number of labeled data')

parser.add_argument('--adversial-training-iteration', type=int, default=30,
                    help='number of step of adversarial training')

parser.add_argument('--mix-option', default=True, type=bool, metavar='N',
                    help='mix option, whether to mix or not')
parser.add_argument('--mix-method', default=0, type=int, metavar='N',
                    help='mix method, set different mix method')
parser.add_argument('--separate-mix', default=False, type=bool, metavar='N',
                    help='mix separate from labeled data and unlabeled data')
parser.add_argument('--co', default=False, type=bool, metavar='N',
                    help='set a random choice between mix and unmix during training')
parser.add_argument('--train_aug', default=False, type=bool, metavar='N',
                    help='augment labeled training data')

parser.add_argument('--model', type=str, default='bert-base-uncased',
                    help='pretrained model')

parser.add_argument('--data-path', type=str, default='yahoo_answers_csv/',
                    help='path to data folders')

parser.add_argument('--loss-type', type=str, default='CE_max_w_pseudo_label_max_2',
                    help='couterfactual loss type')

parser.add_argument('--mix-layers-set', nargs='+',
                    default=[0, 1, 2, 3], type=int, help='define mix layer set')

parser.add_argument('--alpha', default=0.75, type=float,
                    help='alpha for beta distribution')

parser.add_argument('--gamma', default=5, type=float,
                    help='gamma for counterfactual loss')

parser.add_argument('--beta', default=0.75, type=float,
                    help='beta for beta distribution')

parser.add_argument('--lambda-u', default=1, type=float,
                    help='weight for consistency loss term of unlabeled data')
parser.add_argument('--T', default=0.5, type=float,
                    help='temperature for sharpen function')

parser.add_argument('--temp-change', default=1000000, type=int)

parser.add_argument('--margin', default=0.7, type=float, metavar='N',
                    help='margin for hinge loss')
parser.add_argument('--lambda-u-hinge', default=0, type=float,
                    help='weight for hinge loss term of unlabeled data')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
n_gpu = torch.cuda.device_count()
print("GPU num: ", n_gpu)

best_acc = 0
total_steps = 0
flag = 0
print('Whether mix: ', args.mix_option)
print("Mix layers sets: ", args.mix_layers_set)


def main():
    global best_acc
    global mu_avg_epoch
    mu_avg_epoch = 0
    #     global mu_avg_epoch
    # Read dataset and build dataloaders
    train_labeled_set, val_set, test_set, n_labels = get_data(
        args.data_path, args.n_labeled, args.un_labeled, model=args.model, train_aug=args.train_aug)
    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True)
    #     unlabeled_trainloader = Data.DataLoader(
    #         dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=True)
    val_loader = Data.DataLoader(
        dataset=val_set, batch_size=32, shuffle=False)
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=32, shuffle=False)

    # Define the model, set the optimizer
    # model = MixText(n_labels, args.mix_option).cuda()
    model = MixText(n_labels, args.mix_option).to(device)
    #     model = nn.DataParallel(model)
    #     optimizer = AdamW(
    #         [
    #             {"params": model.module.bert.parameters(), "lr": args.lrmain},
    #             {"params": model.module.linear.parameters(), "lr": args.lrlast},
    #         ])

    optimizer = AdamW(
        [
            {"params": model.bert.parameters(), "lr": args.lrmain},
            {"params": model.linear.parameters(), "lr": args.lrlast},
        ])

    num_warmup_steps = math.floor(50)
    num_total_steps = args.val_iteration

    scheduler = None
    # WarmupConstantSchedule(optimizer, warmup_steps=num_warmup_steps)

    train_criterion = CounterFactualLoss()
    criterion = nn.CrossEntropyLoss()

    test_accs = []

    # Start training
    for epoch in range(args.epochs):
        print('alpha, mu_avg_epoch:{}, {}'.format(args.alpha, mu_avg_epoch))
        train(labeled_trainloader, model, optimizer,
              scheduler, train_criterion, epoch, n_labels, args.train_aug, count=epoch)

        # scheduler.step()

        # _, train_acc = validate(labeled_trainloader,
        #                        model,  criterion, epoch, mode='Train Stats')
        # print("epoch {}, train acc {}".format(epoch, train_acc))

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

        print('Epoch: ', epoch)

        print('Best acc:')
        print(best_acc)

        print('Test acc:')
        print(test_accs)

    print("Finished training!")
    print('Best acc:')
    print(best_acc)

    print('Test acc:')
    print(test_accs)

    test_accs_str = [str(i) for i in test_accs]
    test_accs_str = '.\t.'.join(test_accs_str)
    best_acc_str = str(best_acc)
    with open('./data/record/' + 'v2_1121_new_'
              + 'batch_size_' + str(args.batch_size) + '_'
              + 'mix_layer' + str('_'.join([str(i) for i in args.mix_layers_set])) + '_'
              + str(args.data_path).strip('./data/') + '_'
              + str(args.n_labeled) + '_examples'
              + '_alpha' + str(args.alpha)
              + '_beta' + str(args.beta)
              + '_batch_size' + str(args.batch_size)
              + '_gamma' + str(args.gamma)
              + '_' + str(args.loss_type)
              + '.txt', 'a') as fw:
        fw.write('val_acc' + '\t' + 'test_acc' + '\n')
        fw.write(best_acc_str + '\t' + test_accs_str + '\n')


def train(labeled_trainloader, model, optimizer, scheduler, criterion, epoch, n_labels,
          train_aug=False, count=0):
    labeled_train_iter = iter(labeled_trainloader)
    # unlabeled_train_iter = iter(unlabeled_trainloader)
    model.train()

    global total_steps
    global flag
    global mu_avg_epoch
    if flag == 0 and total_steps > args.temp_change:
        print('Change T!')
        args.T = 0.9
        flag = 1

    '''build base classifier'''
    if count == 0:
        for _ in tqdm(range(args.val_iteration)):
            total_steps += 1

            try:
                inputs_x, attention_mask, targets_x, inputs_x_length = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, attention_mask, targets_x, inputs_x_length = labeled_train_iter.next()

            batch_size = inputs_x.size(0)
            targets_x = torch.zeros(batch_size, n_labels).scatter_(
                1, targets_x.view(-1, 1), 1)

            inputs_x, attention_mask, targets_x = inputs_x.to(device), attention_mask.to(device), targets_x.to(device)

            logits = model(inputs_x, x1_attention_mask=attention_mask)

            loss = - torch.mean(torch.sum(F.log_softmax(logits, dim=1) * targets_x, dim=1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    '''build base classifier'''
    labeled_train_iter = iter(labeled_trainloader)
    print('starting adversarial training...')
    avg_mu = []
    for batch_idx in tqdm(range(args.adversial_training_iteration)):
        try:
            inputs_x, attention_mask, targets_x, inputs_x_length = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, attention_mask, targets_x, inputs_x_length = labeled_train_iter.next()

        # for _ in range(10):
        #     print(np.random.beta(args.alpha, args.alpha))
        # l = nn.Parameter(torch.Tensor([np.random.beta(args.alpha, args.alpha)]).to(device),
        #                  requires_grad=True)  # mix_lambda

        if epoch >= 100:
            tmp_mu_avg_peoch = min(mu_avg_epoch, 0.99)
            tmp_beta = args.alpha * (1 - tmp_mu_avg_peoch) / tmp_mu_avg_peoch
            l = nn.Parameter(
                torch.Tensor(np.random.beta(args.alpha, tmp_beta, size=(inputs_x.shape[0], 1, 1))).to(device),
                requires_grad=True)  # mix_lambda, shape: batch_size * 1 * 1
        else:
            l = nn.Parameter(
                torch.Tensor(np.random.beta(args.alpha, args.beta, size=(inputs_x.shape[0], 1, 1))).to(device),
                requires_grad=True)  # mix_lambda, shape: batch_size * 1 * 1
        sgd_optimizer = torch.optim.Adam([l], lr=2e-2)
        # sgd_optimizer.add_param_group(l)
        mix_layer = np.random.choice(args.mix_layers_set, 1)[0]
        mix_layer = mix_layer - 1

        all_inputs = torch.cat(
            [inputs_x], dim=0)

        all_targets = torch.cat(
            [targets_x], dim=0)

        idx1 = torch.randperm(all_inputs.size(0))
        idx2 = torch.randperm(all_inputs.size(0) - all_inputs.size(0)) + all_inputs.size(
            0)  
        idx = torch.cat([idx1, idx2], dim=0)

        input_a, input_b = all_inputs.to(device), all_inputs[idx].to(device)
        attention_mask_a, attention_mask_b = attention_mask.to(device), attention_mask[idx].to(device)
        target_a, target_b = all_targets.to(device), all_targets[idx].to(device)
        target_a_onehot = torch.zeros(all_inputs.size(0), n_labels).to(device).scatter_(
            1, target_a.view(-1, 1), 1)

        '''factual learning'''
        logits = model(input_a, x1_attention_mask=attention_mask_a)
        loss = nn.CrossEntropyLoss()(logits, target_a)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 100 == 0:
            print("epoch {}, step {}, loss on factual data {}".format(
                epoch, batch_idx, loss.item()))

        '''counterfactual lambda learning'''
        model.eval()  
        for param in model.parameters():
            param.requires_grad = False
        for _ in range(5): 


            logits_mix = model(x1=input_a, x1_attention_mask=attention_mask_a
                               , x2=input_b, x2_attention_mask=attention_mask_b
                               , l=l, mix_layer=mix_layer)
            if args.loss_type == 'CE_max_w_pseudo_label_max':
                loss = (-l.squeeze(dim=-1).squeeze(dim=-1) - args.gamma * nn.CrossEntropyLoss(reduction='none')(
                    logits_mix, target_a)
                        ).mean()  

            if args.loss_type == 'CE_max_w_pseudo_label_max_1':
                loss = (-l.squeeze(dim=-1).squeeze(dim=-1) + args.gamma * nn.CrossEntropyLoss(reduction='none')(
                    logits_mix, target_a)
                        + 2 * args.gamma * torch.max(F.softmax(logits_mix, dim=-1), dim=-1)[
                            0]).mean()  
            if args.loss_type == 'CE_max_w_pseudo_label_max_2':
                loss = (-l.squeeze(dim=-1).squeeze(dim=-1) ** 2 - args.gamma * nn.CrossEntropyLoss(reduction='none')(
                    logits_mix, target_a)
                        - 2 * args.gamma * torch.max(F.softmax(logits_mix, dim=-1), dim=-1)[
                            0]).mean()  
 
            loss.backward()
            sgd_optimizer.step()
            sgd_optimizer.zero_grad()
        model.train()  
        for param in model.parameters():
            param.requires_grad = True
        model.zero_grad()
        avg_mu += [l.squeeze(dim=-1).squeeze(dim=-1)]
        '''counterfactual learning'''
        logits_mix = model(x1=input_a, x1_attention_mask=attention_mask_a
                               , x2=input_b, x2_attention_mask=attention_mask_b
                               , l=l, mix_layer=mix_layer)
        logits = model(input_a, x1_attention_mask=attention_mask_a)
        loss = criterion(logits=logits, logits_mix=logits_mix, targets=target_a_onehot)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    cat_mu = torch.cat(avg_mu)
    mu_avg_epoch = (cat_mu.norm(p=1) / cat_mu.shape[0]).item()
    print('mu average and variance at epoch {} : {}, {}'.format(epoch, mu_avg_epoch, torch.var(cat_mu)))


def validate(valloader, model, criterion, epoch, mode):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0

        for batch_idx, (inputs, attention_mask, targets, length) in tqdm(enumerate(valloader)):
            inputs, attention_mask, targets = inputs.to(device), attention_mask.to(device), targets.to(device)
            outputs = model(inputs, x1_attention_mask=attention_mask)
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)

            if batch_idx == 0:
                print("Sample some true labeles and predicted labels")
                print(predicted[:20])
                print(targets[:20])

            correct += (np.array(predicted.cpu()) ==
                        np.array(targets.cpu())).sum()
            loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

        acc_total = correct / total_sample
        loss_total = loss_total / total_sample

    return loss_total, acc_total


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class CounterFactualLoss(object):
    def __call__(self, logits, logits_mix, targets):
        Lx = - \
            torch.sum(F.log_softmax(
                logits, dim=1) * targets, dim=1)  ## batch_size

        #         probs_I = (F.softmax(logits_mix, dim=-1) * targets).sum(dim=-1)  ## batch_size
        probs_I = torch.max(F.softmax(logits_mix, dim=-1), dim=-1)[0]  ## batch_size

        probs = (F.softmax(logits, dim=-1) * targets).sum(dim=-1)  ## batch_size
        rates = torch.where(probs / probs_I > 10.0, torch.tensor([10.0]).to(device), probs / probs_I)  ## batch_size
        loss = torch.mean(Lx * rates)

        return loss


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, outputs_u_2, epoch, mixed=1):

        if args.mix_method == 0 or args.mix_method == 1:

            Lx = - \
                torch.mean(torch.sum(F.log_softmax(
                    outputs_x, dim=1) * targets_x, dim=1))

            probs_u = torch.softmax(outputs_u, dim=1)

            Lu = F.kl_div(probs_u.log(), targets_u, None, None, 'batchmean')

            Lu2 = torch.mean(torch.clamp(torch.sum(-F.softmax(outputs_u, dim=1)
                                                   * F.log_softmax(outputs_u, dim=1), dim=1) - args.margin, min=0))

        elif args.mix_method == 2:
            if mixed == 0:
                Lx = - \
                    torch.mean(torch.sum(F.logsigmoid(
                        outputs_x) * targets_x, dim=1))

                probs_u = torch.softmax(outputs_u, dim=1)

                Lu = F.kl_div(probs_u.log(), targets_u,
                              None, None, 'batchmean')

                Lu2 = torch.mean(torch.clamp(args.margin - torch.sum(
                    F.softmax(outputs_u_2, dim=1) * F.softmax(outputs_u_2, dim=1), dim=1), min=0))
            else:
                Lx = - \
                    torch.mean(torch.sum(F.log_softmax(
                        outputs_x, dim=1) * targets_x, dim=1))

                probs_u = torch.softmax(outputs_u, dim=1)
                Lu = F.kl_div(probs_u.log(), targets_u,
                              None, None, 'batchmean')

                Lu2 = torch.mean(torch.clamp(args.margin - torch.sum(
                    F.softmax(outputs_u, dim=1) * F.softmax(outputs_u, dim=1), dim=1), min=0))

        return Lx, Lu, args.lambda_u * linear_rampup(epoch), Lu2, args.lambda_u_hinge * linear_rampup(epoch)


if __name__ == '__main__':
    main()
