import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import rnn_models
import argparse, pathlib
import glob
import seaborn as sns
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, \
    balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import os
import pandas as pd
import numpy as np
import json
import pickle
from utils import repackage_hidden
from itertools import chain
from matplotlib.lines import Line2D
from tMNISTDataset import get_tmnist_data
from collections import defaultdict, Counter
from mpl_toolkits.axes_grid1 import ImageGrid


def evaluate_model(model, loader, test_lens, args):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    all_predictions = dict()
    all_ground_truth = dict()
    all_gcenters = defaultdict(list)

    for test_len in test_lens:
        ground_truth = []
        predictions = []
        with torch.no_grad():
            for d,t in loader[test_len]:
                hidden = model.init_hidden(args.batch_size)

                t = t.to(dtype=torch.int64)

                data = Variable(d.cuda())
                targets = Variable(t.cuda())


                # print(data.shape)
                output, hidden, out_glimpse, gcenters, sigmas = model(data, hidden, log_attention=True)

                hidden = repackage_hidden(hidden, args)
                #print(gcenters)
                #exit()
                all_gcenters[test_len].append(gcenters)

                ground_truth.append(targets.cpu().data.numpy())
                predictions.append(
                    torch.max(output[-1], dim=1)[1].cpu().data.numpy())

        all_predictions[test_len] = predictions
        all_ground_truth[test_len] = ground_truth

    return all_predictions, all_ground_truth, all_gcenters


def get_glimpses_sequence(model, loader, args, batch , path):
    # Turn on evaluation mode which disables dropout.
    glimpses = []
    model.eval()
    with torch.no_grad():
        for d,t in loader:
            hidden = model.init_hidden(batch)
            t = t.to(dtype=torch.int64)
            data = Variable(d.cuda())
            #print(torch.unique(data))
            targets = Variable(t.cuda())
            output, hidden, glimpses, gcenters, sigmas = model(data, hidden, log_glimpse=True)
            hidden = repackage_hidden(hidden, args)
            break

    #fig_att = plt.figure(figsize=(8., 8.))
    fig_att = plt.gcf()
    plt.subplot(211)
    fig_att.tight_layout()
    d = d.reshape((d.shape[2], d.shape[3]))
    plt.imshow(d.cpu().data.numpy(), cmap='gray')
    #plt.show()
    #exit()
    #plt.subplot(212)
    grid = ImageGrid(fig_att, 212,  # similar to subplot(111)
                     nrows_ncols=(2, 4),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     )
    #print(np.max(glimpses[1]))
    #print(np.rint(glimpses[1]))
    #exit()
    i = 0
    for ax, im in zip(grid, glimpses):
        # Iterating over the grid returns the Axes.
        im = np.rint(im)
        #print(im)
        im = 255*(im - np.min(im))/(np.max(im) - np.min(im) + 1)
        #print(im)
        #exit()
        ax.imshow(im, cmap= 'gray', vmin=0, vmax=255)
        ax.set_title(f't = {i}')
        i += 1

    plt.show()
    plt.draw()
    fig_att.savefig(os.path.join(path, "visuals", "attention_sequence.pdf"), bbox_inches='tight')




def metrics(all_predictions, all_labels, path, lens, split):
    metrics = dict()
    for len in lens:
        predictions = np.array(all_predictions[len])
        predictions = np.reshape(predictions,
                                 predictions.shape[0] * predictions.shape[1])

        labels = np.array(all_labels[len])
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        acc = accuracy_score(labels, predictions)
        balanced_acc = balanced_accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')
        f1 = f1_score(labels, predictions, average='weighted')

        metrics[len] = [acc, balanced_acc, precision, recall, f1]
        cm = confusion_matrix(labels, predictions,
                              labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        fig_m = plt.gcf()
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        #ax.set_title(f'Confusion Matrix {len}')
        ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        plt.show()
        plt.draw()
        if not os.path.exists(os.path.join(path, "visuals")):
            os.makedirs(os.path.join(path, "visuals"))
        fig_m.savefig(
            os.path.join(path, "visuals", f"confusion_matrix_{split}_{len}.pdf"))

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index",
                                        columns=['acc', 'balanced_acc',
                                                 'precision', 'recall', 'f1'])
    metrics_df = metrics_df.round(4)
    print(metrics_df)
    metrics_df.to_csv(os.path.join(path, f'{split}_results.csv'))



def get_file_path(dpath, tag):
    print(dpath)
    file_name = tag.replace("/", "_") + '.csv'
    folder_path = os.path.join(dpath, 'csv')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


def plot_train_loss(data, path):
    fig_loss = plt.gcf()
    plt.plot(data)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="b", lw=4)], ['Loss Train'])
    plt.xlim((0, len(data)))
    plt.ylim((np.ndarray.min(data) - 1, np.ndarray.max(data) + 1))
    plt.show()
    plt.draw()
    fig_loss.savefig(os.path.join(path, "visuals", "loss_train.pdf"), bbox_inches='tight')


def plot_train_acc(data_train, data_val, lens, path):
    fig_loss = plt.gcf()
    lines = [Line2D([0], [0], color="b", lw=4)]
    legends = ['Acc Train']
    plt.plot(data_train*100)
    colors = ["g", "r", "c", "m", "k"]
    for l, c in zip(lens, colors):
        aux = [x * 100 for x in data_val[l]]
        plt.plot(aux)
        lines.append(Line2D([0], [0], color=c, lw=4))
        legends.append(f'Acc Val - {l}')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend(lines, legends)
    plt.xlim((0, len(data_train)))
    plt.ylim((0, 100))
    plt.show()
    plt.draw()
    fig_loss.savefig(os.path.join(path, "visuals", "acc_train.pdf"), bbox_inches='tight')




def compute_glimpse_heatmap(all_gcenters, lens, path, split):

    for len_value in lens:
        htmap = np.zeros((len_value, len_value), dtype=int)
        gcenters_len = all_gcenters[len_value]
        #print(gcenters_len)
        #exit()
        gcenters_len = list(chain.from_iterable(gcenters_len))
        z = Counter(gcenters_len)
        z = dict(z)

        cont_out = 0
        for key, value in z.items():
            if key[0] >= 0 and key[1] >= 0 and key[0] < len_value and key[1] < len_value:
                htmap[key[0], key[1]] = value
            else:
                cont_out +=1
        print(f'cont_out {cont_out}')
        fig_m = plt.gcf()
        ax = plt.subplot()
        sns.heatmap(htmap, cmap='YlGnBu', annot=False, fmt='g', ax=ax, linewidth=0.2)
        plt.show()
        plt.draw()
        if not os.path.exists(os.path.join(path, "visuals")):
            os.makedirs(os.path.join(path, "visuals"))
        fig_m.savefig(
            os.path.join(path, "visuals", f"glimpses_heatmap_{split}_{len_value}.pdf"))

def function_with_args_and_default_kwargs(optional_args=None, **kwargs):
    parser = argparse.ArgumentParser()
    # add some arguments
    # add the other arguments
    for k, v in kwargs.items():
        parser.add_argument('--' + k, default=v)
    args = parser.parse_args(optional_args)
    return args

def get_model(args, device):
    rnn_mod = rnn_models.RNNModel
    model = rnn_mod(args.model, args.ntokens, args.emsize, args.nhid,
                    args.nlayers, args.n_out, args.dropout, False,
                    num_blocks=args.num_blocks, topk=args.topk,
                    use_cudnn_version=args.cudnn, use_adaptive_softmax=args.adaptivesoftmax,
                    cutoffs=args.cutoffs, discrete_input=args.discrete_input,
                    use_inactive=args.use_inactive,
                    blocked_grad=args.blocked_grad, block_dilation=args.block_dilation,
                    layer_dilation=args.layer_dilation, num_modules_read_input=2,
                    glimpse_size=args.glimpse_size, quant_glimpses=args.quant_glimpses).to(device)
    return model

def generate_results():
    folder = os.path.join(os.getcwd(), 'results', 'tMNIST_definitivo_sem_nenhum_feedback_300_10_sem_hierarquia')

    f = open(os.path.join(folder, 'args.json'), "r")
    args = json.loads(f.read())

    args = function_with_args_and_default_kwargs(**args)


    loss_train = np.load(os.path.join(folder, 'loss_train.npy'))
    plot_train_loss(loss_train, folder)

    acc_train = np.load(os.path.join(folder, 'acc_train.npy'))
    a_file = open(os.path.join(folder, 'acc_val.pkl'), "rb")
    acc_val = pickle.load(a_file)
    plot_train_acc(acc_train, acc_val, args.test_lens, folder)


    device = torch.device("cuda" if args.cuda else "cpu")
    print('iniciando as predictions')
    # avalia o modelo no conjunto de teste e val
    #train_loaders, val_loaders, test_loaders = get_glimpse_mnist(
    #    batch_size=args['batch_size'],
    #    lens=args['lens'])

    exit()
    train_loaders, val_loaders, test_loaders = get_tmnist_data(batch_size=args.batch_size, lens=args.test_lens)


    model = get_model(args, device)

    in_size = args.in_size
    checkpoint = torch.load(os.path.join(folder, f'best_model_{in_size}.pt'))

    try:
        checkpoint.eval()
    except AttributeError as error:
        print
        error

    model.load_state_dict(checkpoint['state_dict'])
    print('loaded model ...')
    print('TRAIN DATASET ...')
    '''
    predictions_train, labels_train, all_gcenters_train = evaluate_model(model, train_loaders, [in_size], args)
    metrics(predictions_train, labels_train, folder, [in_size], 'train')
    compute_glimpse_heatmap(all_gcenters_train, [in_size], folder, 'train')

    print('VALIDATION DATASET ...')

    predictions_val, labels_val, all_gcenters_val = evaluate_model(model, val_loaders, args.test_lens, args)
    metrics(predictions_val, labels_val, folder, args.test_lens, 'val')
    compute_glimpse_heatmap(all_gcenters_val, args.test_lens, folder, 'val')

    print('TEST DATASET ...')
    predictions_test, labels_test, all_gcenters_test = evaluate_model(model, test_loaders, args.test_lens, args)
    metrics(predictions_test, labels_test, folder, args.test_lens, 'test')
    compute_glimpse_heatmap(all_gcenters_test, args.test_lens, folder, 'test')
    '''

    _, av_loader, _ = get_tmnist_data(batch_size=1, lens=[in_size])
    get_glimpses_sequence(model, av_loader[in_size], args, 1, folder)

if __name__ == '__main__':
    generate_results()