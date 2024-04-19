import wandb
import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import argparse
from sklearn import metrics
import torch.nn.functional as F
import mat73

from mydataloader.mydataset_multi import MyDataset
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


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=4, padding=1)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=4, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(160, 128)
        self.fc2 = nn.Linear(128+960, 960)
        self.fc3 = nn.Linear(960, 3)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x1, x2 = x[0], x[1]
        # Apply convolutions
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = self.dropout(x1)
        x1 = F.relu(self.conv4(x1))

        # Flatten the output for the fully connected layers
        x1 = x1.view(x1.size(0), -1)

        # Fully connected layers
        x1 = F.relu(self.fc1(x1))

        x1 = self.fc2(torch.cat((x1, x2), dim=1))
        x1 = self.dropout(x1)
        yh = self.fc3(x1)
        

        return yh, x1
    

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
    imu_model = SimpleCNN()
    imu_model.load_state_dict(torch.load(args.imu_model_path))
    imu_model.to(device)


    # load data
    directory = '../processedData/v2'
    file_names = [f for f in os.listdir(directory) if f.endswith('.mat')]
    file_names = sorted(file_names)[:-3]
    # print(file_names)
    all_data, all_labels = [], []
    all_acc, all_gyro, all_mag = [], [], []
    for file_name in file_names:
        mat = mat73.loadmat(os.path.join(directory, file_name))
        data = mat['audio_data'] 
        label = mat['label']
        all_data.append(np.transpose(data))
        all_labels.append(label)
        acc_x, acc_y, acc_z = np.array(mat['acc_x']), np.array(mat['acc_y']), np.array(mat['acc_z'])
        gyro_x, gyro_y, gyro_z = np.array(mat['gyro_x']), np.array(mat['gyro_y']), np.array(mat['gyro_z'])
        mag_x, mag_y, mag_z = np.array(mat['mag_x']), np.array(mat['mag_y']), np.array(mat['mag_z'])
        all_acc.append(np.stack((acc_x, acc_y, acc_z), axis=0))
        all_gyro.append(np.stack((gyro_x, gyro_y, gyro_z), axis=0))
        all_mag.append(np.stack((mag_x, mag_y, mag_z), axis=0))
        
    all_data = np.array(all_data, dtype=object)
    all_labels = np.array(all_labels, dtype=object)
    
    # dataloader
    train_dataset, train_labels = [], []
    test_dataset, test_labels = [], []
    train_acc, train_gyro, train_mag = [], [], []
    test_acc, test_gyro, test_mag = [], [], []
    split_prop = args.split_prop
    for i in range(len(all_data)):
        sub_data = all_data[i]
        sub_label = all_labels[i]
        sub_acc, sub_gyro, sub_mag = all_acc[i], all_gyro[i], all_mag[i]
        d_split_point = int(len(sub_data)*split_prop)
        d_end_point = int(len(sub_data)*(split_prop+0.3))
        l_split_point = int(len(sub_label)*split_prop)
        l_end_point = int(len(sub_label)*(split_prop+0.3))

        acc_split_point = int(len(sub_acc[0])*split_prop)
        acc_end_point = int(len(sub_acc[0])*(split_prop+0.3))

        gyro_split_point = int(len(sub_gyro[0])*split_prop)
        gyro_end_point = int(len(sub_gyro[0])*(split_prop+0.3))

        mag_split_point = int(len(sub_mag[0])*split_prop)
        mag_end_point = int(len(sub_mag[0])*(split_prop+0.3))

        if split_prop < 0.7:
            train_dataset.append(np.concatenate((sub_data[:d_split_point], sub_data[d_end_point:]), axis=0))
            train_labels.append(np.concatenate((sub_label[:l_split_point,:], sub_label[l_end_point:,:]), axis=0))
            test_dataset.append(sub_data[d_split_point:d_end_point])
            test_labels.append(sub_label[l_split_point:l_end_point,:])

            train_acc.append(np.concatenate((sub_acc[:, :acc_split_point], sub_acc[:, acc_end_point:]), axis=1))
            test_acc.append(sub_acc[:,acc_split_point:acc_end_point])

            train_gyro.append(np.concatenate((sub_gyro[:, :gyro_split_point], sub_gyro[:, gyro_end_point:]), axis=1))
            test_gyro.append(sub_gyro[:,gyro_split_point:gyro_end_point])

            train_mag.append(np.concatenate((sub_mag[:, :mag_split_point], sub_mag[:, mag_end_point:]), axis=1))
            test_mag.append(sub_mag[:,mag_split_point:mag_end_point])
        else:
            train_dataset.append(sub_data[:d_split_point])
            train_labels.append(sub_label[:l_split_point,:])
            test_dataset.append(sub_data[d_split_point:])
            test_labels.append(sub_label[l_split_point:,:])

            train_acc.append(sub_acc[:, :acc_split_point])
            test_acc.append(sub_acc[:, acc_split_point:])

            train_gyro.append(sub_gyro[:, :gyro_split_point])
            test_gyro.append(sub_gyro[:, gyro_split_point:])

            train_mag.append(sub_mag[:, :mag_split_point])
            test_mag.append(sub_mag[:, mag_split_point:])

    traindata = np.array(train_dataset,dtype=object)
    evaldata = np.array(test_dataset,dtype=object)
    traindataset = MyDataset(datasource = [traindata, train_acc, train_gyro, train_mag],
                             labels = train_labels,
                             window_size=1.5, 
                             step_size=0.5, 
                             transform=None,
                             classes_num = args.num_class)
    evaldataset = MyDataset(datasource = [evaldata, test_acc, test_gyro, test_mag], 
                                labels = test_labels,
                                window_size=1.5, 
                                step_size=0.5, 
                                transform=None,
                                classes_num = args.num_class)

    train_indices = []
    for i, (_, _, label) in enumerate(traindataset):
        if np.sum(label) != 0:
            train_indices += [i]

    dl = DataLoader(dataset=Subset(traindataset, train_indices),
                    worker_init_fn=worker_init_fn,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size,
                    shuffle=True)
                    
    eval_indices, ood_indices = [], []
    for i, (_, _, label) in enumerate(evaldataset):
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

    pbar = tqdm(dl)
    pbar.set_description("Validating")
    for batch in pbar:
        x, i, y = batch
        x = x.to(device)
        i = i.to(device)
        y = y.to(device)
        with torch.no_grad():
            x = _mel_forward(x, mel)
            _, features = model(x)
            # print(i.shape, features.shape)
            y_hat, features_all = imu_model([i.float(),features])
        targets.append(y.cpu().numpy())
        outputs.append(y_hat.float().cpu().numpy())
        losses.append(F.cross_entropy(y_hat, y).cpu().numpy())
        train_features.append(features_all.cpu().numpy())

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    losses = np.stack(losses)
    train_features = np.concatenate(train_features)


    test_targets = []
    test_outputs = []
    test_features = []
    pbar = tqdm(eval_dl)
    pbar.set_description("Validating")
    for batch in pbar:
        x, i, y = batch
        x = x.to(device)
        i = i.to(device)
        y = y.to(device)
        with torch.no_grad():
            x = _mel_forward(x, mel)
            _, features = model(x)
            # print(i.shape, features.shape)
            y_hat, features_all = imu_model([i.float(),features])
        test_targets.append(y.cpu().numpy())
        test_outputs.append(y_hat.float().cpu().numpy())
        test_features.append(features_all.cpu().numpy())

    test_targets = np.concatenate(test_targets)
    test_outputs = np.concatenate(test_outputs)
    test_features = np.concatenate(test_features)

########################################################################################
########################### OOD #######################################################
########################################################################################
    # Extract w and b
    w = imu_model.fc3.weight.cpu().detach().numpy()
    b = imu_model.fc3.bias.cpu().detach().numpy()
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
    if len(ood_target)>10000:
        ood_target = ood_target[:10000]
        extended_probabilities = extended_probabilities[:10000]
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
    parser.add_argument('--split_prop', type=float, default=None)

    # ood
    parser.add_argument('--model_path', type=str, default="./ood/multi_1")
    parser.add_argument('--imu_model_path', type=str, default="./ood/multi_1_imu")
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--M', type=int, default=2)

    args = parser.parse_args()
    ood_test(args)