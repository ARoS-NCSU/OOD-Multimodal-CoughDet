import wandb
import numpy as np
import os
from tqdm import tqdm
import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import argparse
from sklearn import metrics
import torch.nn.functional as F
import mat73

from mydataloader.mydataset import MyDataset
from models.mn.model import get_model as get_mobilenet
from models.dymn.model import get_model as get_dymn
from models.preprocess import AugmentMelSTFT
from helpers.init import worker_init_fn
from helpers.utils import NAME_TO_WIDTH, exp_warmup_linear_down, mixup

from ood_metrics import calc_metrics
from scipy.special import logsumexp, softmax
from sklearn.preprocessing import label_binarize
import random
from numpy.linalg import norm, pinv
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import pairwise_distances_argmin_min
import cupy as cp


def _mel_forward(x, mel):
    old_shape = x.size()
    x = x.reshape(-1, old_shape[2])
    x = mel(x)
    x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
    return x


def generalized_entropy(softmax_id_val, gamma, M):
    probs =  softmax_id_val 
    probs_sorted = np.sort(probs, axis=1)[:,-M:]
    scores = np.sum(probs_sorted**gamma * (1 - probs_sorted)**(gamma), axis=1)

    return -scores 

def ood_test(args):
    # Train Models for Acoustic Scene Classification

    # logging is done using wandb
    wandb.init(
        project="Mydataset",
        notes="OOD",
        tags=["Environmental Sound Classification", "OOD"],
        config=args,
        name=args.experiment_name
    )

    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    # model to preprocess waveform into mel spectrograms
    mel = AugmentMelSTFT(n_mels=args.n_mels,
                         sr=args.resample_rate,
                         win_length=args.window_size,
                         hopsize=args.hop_size,
                         n_fft=args.n_fft,
                         freqm=args.freqm,
                         timem=args.timem,
                         fmin=args.fmin,
                         fmax=args.fmax,
                         fmin_aug_range=args.fmin_aug_range,
                         fmax_aug_range=args.fmax_aug_range
                         )
    mel.to(device)

    # load prediction model
    model_name = args.model_name
    pretrained_name = model_name if args.pretrained else None
    width = NAME_TO_WIDTH(model_name) if model_name and args.pretrained else args.model_width
    if model_name.startswith("dymn"):
        model = get_dymn(width_mult=width, pretrained_name=pretrained_name,
                         pretrain_final_temp=args.pretrain_final_temp,
                         num_classes=args.num_class)
    else:
        model = get_mobilenet(width_mult=width, pretrained_name=pretrained_name,
                              head_type=args.head_type, se_dims=args.se_dims,
                              num_classes=args.num_class)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)


    # load data
    directory = '../processedData/v2'
    file_names = [f for f in os.listdir(directory) if f.endswith('.mat')]
    file_names = sorted(file_names)[:-3]
    # print(file_names)
    all_data, all_labels = [], []
    for file_name in file_names:
        mat = mat73.loadmat(os.path.join(directory, file_name))
        data = mat['audio_data'] 
        label = mat['label']
        all_data.append(np.transpose(data))
        all_labels.append(label)
    all_data = np.array(all_data, dtype=object)
    all_labels = np.array(all_labels, dtype=object)
    
    # dataloader
    start_sub, end_sub = args.start_sub, args.end_sub
    if start_sub and end_sub:
        print("test set:", file_names[start_sub:end_sub])
        traindataset = MyDataset(datasource = np.concatenate((all_data[0:start_sub], all_data[end_sub:]), axis=0),
                                labels = np.concatenate((all_labels[0:start_sub], all_labels[end_sub:]), axis=0),
                                window_size=1.5, 
                                step_size=0.5, 
                                transform=None,
                                classes_num = args.num_class)

        # evaluation loader
        evaldataset = MyDataset(datasource = all_data[start_sub:end_sub],
                                labels = all_labels[start_sub:end_sub],
                                window_size=1.5, 
                                step_size=0.5, 
                                transform=None,
                                classes_num = args.num_class)
    elif start_sub:
        print("test set:", file_names[start_sub:])
        traindataset = MyDataset(datasource = all_data[:start_sub],
                                labels = all_labels[:start_sub],
                                window_size=1.5, 
                                step_size=0.5, 
                                transform=None,
                                classes_num = args.num_class)

        # evaluation loader
        evaldataset = MyDataset(datasource = all_data[start_sub:],
                                    labels = all_labels[start_sub:],
                                    window_size=1.5, 
                                    step_size=0.5, 
                                    transform=None,
                                    classes_num = args.num_class)
    else:
        print("test set:", file_names[:end_sub])
        traindataset = MyDataset(datasource = all_data[end_sub:],
                                labels = all_labels[end_sub:],
                                window_size=1.5, 
                                step_size=0.5, 
                                transform=None,
                                classes_num = args.num_class)

        # evaluation loader
        evaldataset= MyDataset(datasource = all_data[:end_sub], 
                                labels = all_labels[:end_sub],
                                window_size=1.5, 
                                step_size=0.5, 
                                transform=None,
                                classes_num = args.num_class)

    train_indices = []
    for i, (_, label) in enumerate(traindataset):
        if np.sum(label) != 0:
            train_indices += [i]

    dl = DataLoader(dataset=Subset(traindataset, train_indices),
                    worker_init_fn=worker_init_fn,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size,
                    shuffle=True)

    eval_indices, ood_indices = [], []
    for i, (_, label) in enumerate(evaldataset):
        if np.sum(label) != 0:
            eval_indices += [i]  
        else: ood_indices += [i]  
    print("in samples:", len(eval_indices))
    print("ood samples:", len(ood_indices))
    # selected_indices = eval_indices + random.sample(ood_indices, len(eval_indices))
    eval_dl = DataLoader(dataset= evaldataset,
                         worker_init_fn=worker_init_fn,
                         num_workers=args.num_workers,
                         batch_size=args.batch_size)
    model.eval()
    mel.eval()

    targets = []
    outputs = []
    losses = []
    train_features = []
    def get_features(module, input, output):
        train_features.append(output.cpu().numpy())
    handle = model.classifier[4].register_forward_hook(get_features)

    pbar = tqdm(dl)
    pbar.set_description("Validating")
    for batch in pbar:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            x = _mel_forward(x, mel)
            y_hat, feature = model(x)
        targets.append(y.cpu().numpy())
        outputs.append(y_hat.float().cpu().numpy())
        losses.append(F.cross_entropy(y_hat, y).cpu().numpy())

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    losses = np.stack(losses)
    train_features = np.concatenate(train_features)
    handle.remove()

    test_targets = []
    test_outputs = []
    test_features = []
    def get_features(module, input, output):
        test_features.append(output.cpu().numpy())
    handle = model.classifier[4].register_forward_hook(get_features)
    pbar = tqdm(eval_dl)
    pbar.set_description("Validating")
    for batch in pbar:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            x = _mel_forward(x, mel)
            y_hat, feature = model(x)
        test_targets.append(y.cpu().numpy())
        test_outputs.append(y_hat.float().cpu().numpy())

    test_targets = np.concatenate(test_targets)
    test_outputs = np.concatenate(test_outputs)
    test_features = np.concatenate(test_features)
    handle.remove()

########################################################################################
########################### OOD #######################################################
########################################################################################
    # Extract w and b
    w = model.classifier[-1].weight.cpu().detach().numpy()
    b = model.classifier[-1].bias.cpu().detach().numpy()
    logit_id_val = test_features @ w.T + b
    logit_id_train = train_features @ w.T + b

    print('computing softmax...')
    # softmax_id_train = softmax(logit_id_train, axis=-1)
    softmax_id_val = softmax(logit_id_val, axis=-1)
    softmax_id_train = softmax(logit_id_train, axis=-1)

    u = -np.matmul(pinv(w), b)

    final_target = []
    ood_target = []
    for i in range(len(test_targets)):
        if sum(test_targets[i]) == 0:
            ood_target.append(0)
            final_target.append(2)
        else: 
            final_target.append(np.argmax(test_targets[i]))
            ood_target.append(1)
            # if np.argmax(test_targets[i]) == 2:
            #     ood_target.append(0)
            # else: ood_target.append(1)
    
    ### GEN
    # score_id = generalized_entropy(softmax_id_val, args.gamma, args.M)

    ### ViM
    def logsumexp_gpu(arr, axis=None):
        # Find the maximum value in the array to avoid numerical issues
        arr_max = cp.max(arr, axis=axis, keepdims=True)
        
        # Stable computation of log sum exp
        if axis is None:
            return arr_max + cp.log(cp.sum(cp.exp(arr - arr_max)))
        else:
            return arr_max + cp.log(cp.sum(cp.exp(arr - arr_max), axis=axis))
    
    
    DIM = 320
    print(f'{DIM=}')

    print('computing principal space...')
    train_features_gpu = cp.asarray(train_features)
    test_features_gpu = cp.asarray(test_features)
    logit_id_train_gpu = cp.asarray(logit_id_train)
    logit_id_val_gpu = cp.asarray(logit_id_val)
    u_gpu = cp.asarray(u)

    # Center the features
    centered_features = train_features_gpu - u_gpu

    # Compute covariance matrix
    cov_matrix = cp.cov(centered_features, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eig_vals, eigen_vectors = cp.linalg.eigh(cov_matrix)

    # Compute NS on the GPU
    NS_gpu = cp.ascontiguousarray((eigen_vectors.T[cp.argsort(eig_vals * -1)[DIM:]]).T)
    NS = cp.asnumpy(NS_gpu)

    print('computing alpha...')
    vlogit_id_train = norm(np.matmul(train_features - u, NS), axis=-1)
    alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
    print(f'{alpha=:.4f}')

    vlogit_id_val = norm(np.matmul(test_features - u, NS), axis=-1) * 0.5
    energy_id_val = logsumexp(logit_id_val, axis=-1)
    score_id = -vlogit_id_val + energy_id_val

    score_id_norm = (score_id-score_id.min())/(score_id.max()-score_id.min())
    final_output = []

    best_thresh = find_thresh(final_target, test_outputs, score_id_norm)

    for i in range(len(test_targets)):
        if score_id_norm[i] >= best_thresh:
            final_output.append(np.argmax(test_outputs[i]))
        else:
            final_output.append(2)

    accuracy, precision, recall, f1_scores = all_test(test_outputs.argmax(axis=1), final_target)
    ood_accuracy, ood_precision, ood_recall, ood_f1_scores = all_test(final_output, final_target)

    # Calculate probabilities for the negative class
    prob_neg_class = 1 - score_id_norm

    # Combine to form an array of shape (n_samples, 2)
    extended_probabilities = np.vstack((prob_neg_class, score_id_norm)).T
    oodmetrics = calc_metrics(np.array(score_id),np.array(ood_target))
    wandb.log({"my_custom_roc" : wandb.plot.roc_curve(np.array(ood_target),
                                 extended_probabilities),
                "fpr95": oodmetrics["fpr_at_95_tpr"],
                "detection_error": oodmetrics["detection_error"],
                "auroc": oodmetrics["auroc"],
                "aupr_in": oodmetrics["aupr_in"],
                "aupr_out": oodmetrics["aupr_out"],
                "accuracy": accuracy,
                "cough_f1": f1_scores[0],
                "speech_f1": f1_scores[1],
                "cough_pre": precision[0],
                "speech_pre": precision[1],
                "cough_rec": recall[0],
                "speech_rec": recall[1],
                "ood_accuracy": ood_accuracy,
                "ood_cough_f1": ood_f1_scores[0],
                "ood_speech_f1": ood_f1_scores[1],
                "ood_cough_pre": ood_precision[0],
                "ood_speech_pre": ood_precision[1],
                "ood_cough_rec": ood_recall[0],
                "ood_speech_rec": ood_recall[1]})

def all_test(predictions, targets):
    accuracy = metrics.accuracy_score(targets, predictions)

    conf_matrix = metrics.confusion_matrix(targets, predictions)
    # Calculating TP, FP, FN for each class
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP

    # Calculating precision and recall for each class
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # Calculating F1 score for each class
    f1_scores = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_scores

def find_thresh(targets, probabilities, ood_scores):
    best_threshold = 0
    best_f1 = 0

    # Iterate over possible thresholds
    for threshold in np.linspace(0, 1, 101):
        final_output = []
        for i in range(len(targets)):
            if ood_scores[i] >= threshold:
                final_output.append(np.argmax(probabilities[i]))
            else:
                final_output.append(2)
        ood_accuracy, ood_precision, ood_recall, ood_f1_scores = all_test(final_output, targets)
        
        # Update best threshold and F1 score
        if ood_f1_scores[0] > best_f1:
            best_f1 = ood_f1_scores[0]
            best_threshold = threshold
    return best_threshold

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--experiment_name', type=str, default="MyDataset")
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--num_class', type=int, default=9)

    # training
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--model_name', type=str, default="mn10_as")
    parser.add_argument('--pretrain_final_temp', type=float, default=1.0)  # for DyMN
    parser.add_argument('--model_width', type=float, default=1.0)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--se_dims', type=str, default="c")
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--mixup_alpha', type=float, default=0.3)
    parser.add_argument('--no_roll', action='store_true', default=False)
    parser.add_argument('--no_wavmix', action='store_true', default=False)
    parser.add_argument('--gain_augment', type=int, default=12)
    parser.add_argument('--weight_decay', type=int, default=0.0)

    # lr schedule
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--warm_up_len', type=int, default=8)
    parser.add_argument('--ramp_down_start', type=int, default=8)
    parser.add_argument('--ramp_down_len', type=int, default=25)
    parser.add_argument('--last_lr_value', type=float, default=0.001)

    # preprocessing
    parser.add_argument('--resample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--freqm', type=int, default=0)
    parser.add_argument('--timem', type=int, default=0)
    parser.add_argument('--fmin', type=int, default=0)
    parser.add_argument('--fmax', type=int, default=None)
    parser.add_argument('--fmin_aug_range', type=int, default=10)
    parser.add_argument('--fmax_aug_range', type=int, default=2000)

    # select dataset
    parser.add_argument('--start_sub', type=int, default=None)
    parser.add_argument('--end_sub', type=int, default=None)

    # ood
    parser.add_argument('--model_path', type=str, default="./saved_models/single_1")
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--M', type=int, default=2)

    args = parser.parse_args()
    ood_test(args)