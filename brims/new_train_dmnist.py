# coding: utf-8
import numpy as np
import time
import math
import os
import torch
import torch.nn as nn
import datetime
import rnn_models
import baseline_lstm_model
import random
import mixed
from tMNISTDataset import get_tmnist_data
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
from utils import repackage_hidden, get_args, plot_grad_flow
import json
import pickle
from collections import defaultdict

# Set the random seed manually for reproducibility.
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False
np.random.seed(0)


args = get_args()
print(args)

best_test = {i: 0.0 for i in args.test_lens}
acc_test = defaultdict(list)
best_val = {i: 0.0 for i in args.test_lens}
test_lens = args.test_lens
acc_val = defaultdict(list)
acc_train = []
loss_train = []
in_size = args.in_size
inp_size = in_size
val_size = inp_size
do_multimnist = False
######## Plot Specific Details ########

colors = ['white', 'black']
cmap = LinearSegmentedColormap.from_list('name', colors)
norm = plt.Normalize(0, 1)

matplotlib.rc('xtick', labelsize=7.5)
matplotlib.rc('ytick', labelsize=7.5)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

# Get Data Loaders

train_loaders, val_loaders, test_loaders = get_tmnist_data(batch_size=args.batch_size, lens=test_lens)


train_loader = train_loaders[in_size]
print(f'len train_loader {len(train_loader)}')


eval_batch_size = 32

# create folder for current experiments
# name: args.name + current time
# includes: entire scripts for faithful reproduction, train & test logs
folder_name = str(datetime.datetime.now())[:-7]
if args.name is not None:
    folder_name = str(args.name)

if not os.path.exists(folder_name):
    os.mkdir(folder_name)
if not os.path.exists(folder_name+'/visuals/'):
    os.mkdir(folder_name+'/visuals/')

logger_args = open(os.path.join(os.getcwd(), folder_name, 'args.txt'), 'a')
logger_output = open(os.path.join(os.getcwd(), folder_name, 'output.txt'), 'a')
logger_epoch_output = open(os.path.join(os.getcwd(), folder_name, 'epoch_output.txt'), 'a')

# save args to logger
logger_args.write(str(args) + '\n')


json.dump(vars(args), open(os.path.join(os.getcwd(), folder_name, "args.json"), 'w'))


# define saved model file location
savepath = os.path.join(os.getcwd(), folder_name)

###############################################################################
# Build the model
###############################################################################

ntokens = args.ntokens
n_out = args.n_out

if args.adaptivesoftmax:
    print("Adaptive Softmax is on: the performance depends on cutoff values. check if the cutoff is properly set")
    print("Cutoffs: " + str(args.cutoffs))
    if args.cutoffs[-1] > ntokens:
        raise ValueError("the last element of cutoff list must be lower than vocab size of the dataset")
    criterion_adaptive = nn.AdaptiveLogSoftmaxWithLoss(args.nhid, ntokens, cutoffs=args.cutoffs).to(device)
else:
    criterion = nn.CrossEntropyLoss()


if args.algo == "blocks":
    rnn_mod = rnn_models.RNNModel
    model = rnn_mod(args.model, ntokens, args.emsize, args.nhid,
                            args.nlayers, n_out, args.dropout, False,
                            num_blocks = args.num_blocks, topk = args.topk,
                            use_cudnn_version=args.cudnn, use_adaptive_softmax=args.adaptivesoftmax,
                            cutoffs=args.cutoffs, discrete_input=args.discrete_input, use_inactive = args.use_inactive,
                            blocked_grad=args.blocked_grad, block_dilation=args.block_dilation,
                            layer_dilation=args.layer_dilation, num_modules_read_input=2,
                            glimpse_size=args.glimpse_size, quant_glimpses=args.quant_glimpses).to(device)
elif args.algo == "lstm":
    rnn_mod = baseline_lstm_model.RNNModel
    model = rnn_mod(args.model, ntokens, args.emsize, args.nhid,
                            args.nlayers, n_out, args.dropout, False,
                            use_cudnn_version=args.cudnn, use_adaptive_softmax=args.adaptivesoftmax,
                            cutoffs=args.cutoffs).to(device)
elif args.algo == 'mixed':
    rnn_mod = mixed.RNNModel
    model = rnn_mod(args.model, ntokens, args.emsize, args.nhid,
                            args.nlayers, args.dropout, False,
                            num_blocks = args.num_blocks, topk = args.topk,
                            use_cudnn_version=args.cudnn, use_adaptive_softmax=args.adaptivesoftmax,
                            cutoffs=args.cutoffs, use_inactive=args.use_inactive ,
                            blocked_grad=args.blocked_grad).to(device)
else:
    raise Exception("Algorithm option not found")

if os.path.exists(folder_name+'/model.pt'):
    state = torch.load(folder_name+'/model.pt')
    model.load_state_dict(state['state_dict'])
    global_epoch = state['epoch']
    best_val = state['best_val']
    loss_train = np.load(os.path.join(folder_name, 'loss_train.npy'))
    loss_train = loss_train.tolist()
    acc_train = np.load(os.path.join(folder_name, 'acc_train.npy'))
    acc_train = acc_train.tolist()
    #with open(os.path.join(folder_name, 'acc_val.json'), 'r') as f:
        #acc_val = f.read()
    #with open(os.path.join(folder_name, 'acc_test.json'), 'r') as f:
        #acc_test = f.read()
    a_file = open(folder_name + "/acc_val.pkl", "rb")
    acc_val = pickle.load(a_file)
    # json.dump(acc_test, open(folder_name + "/acc_test.json", 'w'))
    a_file = open(folder_name + "/acc_test.pkl", "rb")
    acc_test = pickle.load(a_file)

else:
    global_epoch = 1

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model Built with Total Number of Trainable Parameters: " + str(total_params))
if not args.cudnn:
    print(
        "--cudnn is set to False. the model will use RNNCell with for loop, instead of cudnn-optimzed RNN API. Expect a minor slowdown.")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

print("Model Built with Total Number of Trainable Parameters: " + str(total_params))

###############################################################################
# Training code
###############################################################################


def evaluate_(test_lens, split):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    if split is "Val":
        loader = val_loaders
    else:
        loader = test_loaders

    test_acc = {i: 0.0 for i in test_lens}
    val_loss = 0.0
    cont = 0
    for test_len in test_lens:
        total_acc = 0.0
        num_batches = 0
        #print(len(loader[test_len]))
        with torch.no_grad():
            for d,t in loader[test_len]:
                hidden = model.init_hidden(args.batch_size)
                cont += 1
                t = t.to(dtype=torch.int64)

                data = Variable(d.cuda())
                targets = Variable(t.cuda())


                num_batches += 1
                #print(data.shape)
                output, hidden, _, _, _ = model(data, hidden)

                if not args.adaptivesoftmax:
                    loss = criterion(output[-1], targets)
                    acc = torch.eq(torch.max(output[-1],dim=1)[1], targets).double().mean()
                else:
                    _, loss = criterion_adaptive(output.view(-1, args.nhid), targets)

                total_acc += acc.item()
                hidden = repackage_hidden(hidden, args)
                if test_len is val_size:
                    val_loss += loss.item()
                #if (cont == 500):
                    #break

        test_acc[test_len] = total_acc / num_batches

    if split is "Val":
        val_loss = val_loss / num_batches
        if args.scheduler:
            scheduler.step(val_loss)

    return test_acc


def train(epoch):
    global best_val
    total_loss = 0.
    forward_elapsed_time = 0.
    start_time = time.time()

    i = 0
    j = 0

    calc_mask = True
    start_epoch_time = time.time()

    for d, t in train_loader:
        #print(epoch)
        hidden = model.init_hidden(args.batch_size)
        model.train()
        i += 1

        t = t.to(dtype=torch.int64)

        data = Variable(d.cuda())
        targets = Variable(t.cuda())
        #print(f'antes de entrar no modelo {data}')
        # synchronize cuda for a proper speed benchmark
        torch.cuda.synchronize()

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        forward_start_time = time.time()

        hidden = repackage_hidden(hidden, args)
        model.zero_grad()

        output, hidden, _, _, _ = model(data, hidden, calc_mask)

        if not args.adaptivesoftmax:
            loss = criterion(output[-1], targets)
            acc = torch.eq(torch.max(output[-1],dim=1)[1], targets).double().mean()
        else:
            raise Exception('not implemented')
            _, loss = criterion_adaptive(output.view(-1, args.nhid), targets)

        #print(f'loss: {loss}')
        #exit()
        #loss = loss*1e8
        #print(f'loss: {loss}')
        total_loss += acc.item()

        # synchronize cuda for a proper speed benchmark
        torch.cuda.synchronize()

        forward_elapsed = time.time() - forward_start_time
        forward_elapsed_time += forward_elapsed

        loss.backward()
        #plot_grad_flow(model.named_parameters(), "depois_backward.jpg")
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        #plot_grad_flow(model.named_parameters(), "depois_step.jpg")
        #exit()
        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            printlog = '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | forward ms/batch {:5.2f} | average acc {:5.4f} | ppl {:8.2f}'.format(
                epoch, i, len(train_loader), optimizer.param_groups[0]['lr'],
                              elapsed * 1000 / args.log_interval, forward_elapsed_time * 1000 / args.log_interval,
                cur_loss, math.exp(cur_loss))
            # print and save the log
            acc_train.append(cur_loss)
            loss_train.append(loss.item())

            plot_grad_flow(model.named_parameters(), "apos_epoca.jpg")
            print(printlog)
            logger_output.write(printlog + '\n')
            logger_output.flush()
            total_loss = 0.
            # reset timer
            start_time = time.time()
            forward_start_time = time.time()
            forward_elapsed_time = 0.

        if i % args.log_interval == 0 and i > 0:
            j += 1


            test_acc = evaluate_(test_lens, split="Test")
            #print('avaliou o teste')
            val_acc = evaluate_(test_lens, split="Val")
            #print('avaliou a validacao')
            printlog = ''

            for key in test_acc:
                #print(val_acc[key])
                #print(acc_val[key])
                acc_val[key].append(val_acc[key])
                #print(test_acc[key])
                acc_test[key].append(test_acc[key])

                if val_acc[key] > best_val[key]:
                    best_val[key] = val_acc[key]
                    best_test[key] = test_acc[key]
                    state = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'best_val': best_val
                        }
                    torch.save(state, folder_name+f'/best_model_{key}.pt')

            for key in test_acc:
                printlog = printlog + '\n' + '|Seq_len: {} | Test Current: {} | Test Optim: {} | Val Current: {} | Val Best: {} |'.format(str(key), str(test_acc[key]), str(best_test[key]), str(val_acc[key]), str(best_val[key]))

            logger_output.write(printlog+'\n\n')
            logger_output.flush()

            print(printlog+'\n\n')
        #if (i == 500):
            #break
    state = {
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'best_val': best_val
    }
    torch.save(state, folder_name+'/model.pt')

    #json.dump(acc_val, open(folder_name+"/acc_val.json", 'w'))
    a_file = open(folder_name+"/acc_val.pkl", "wb")
    pickle.dump(acc_val, a_file)
    #json.dump(acc_test, open(folder_name + "/acc_test.json", 'w'))
    a_file = open(folder_name+"/acc_test.pkl", "wb")
    pickle.dump(acc_test, a_file)

    np.save(folder_name + "/loss_train.npy", loss_train)
    np.save(folder_name + "/acc_train.npy", acc_train)
    #print(np.load(folder_name + "/loss_train.npy", allow_pickle=True))
    #print(np.load(folder_name + "/acc_train.npy", allow_pickle=True))
    #exit()
    #with open(folder_name + "/loss_train.txt", "wb") as fp:  # Pickling
        #pickle.dump(loss_train, fp)
    #with open(folder_name + "/acc_train.txt", "wb") as fp:  # Pickling
        #pickle.dump(acc_train, fp)

    print(f'epoch time {time.time() - start_epoch_time}')


for epoch in range(global_epoch, args.epochs + 1):
    train(epoch)
