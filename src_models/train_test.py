import wandb
import numpy as np
import os
from tqdm import tqdm
import torch
import argparse
from sklearn import metrics
import torch.nn.functional as F

from models.mn.model import get_model as get_mobilenet
from models.dymn.model import get_model as get_dymn
from models.imu_model import SimpleCNN
from models.preprocess import AugmentMelSTFT
from helpers.utils import NAME_TO_WIDTH, exp_warmup_linear_down, mixup
from helpers.laod_data import load_single_modal, load_multi_modal

from sklearn.preprocessing import label_binarize

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        """
        :param patience: Number of epochs to wait after min has been hit. After this number, training stops.
        :param min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
    
def _mel_forward(x, mel):
    old_shape = x.size()
    x = x.reshape(-1, old_shape[2])
    x = mel(x)
    x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
    return x


def train(args):
    # logging is done using wandb
    wandb.init(
        project="Mydataset",
        notes="Train on my dataset.",
        tags=["Environmental Sound Classification", "Fine-Tuning"],
        group="experiment_1_in_KD",
        config=args,
        name=args.experiment_name
    )
    wandb.define_metric("val_loss", summary="min")

    # model to preprocess waveform into mel spectrograms
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
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

    # Load the saved state dictionary
    model.to(device)
    
    # Check parameters beforing data preparation
    if args.setting == "in" and args.split_prop is None:
        raise ValueError(f"Invalid value seting for args.setting: {args.setting}. Expected values for split_prop.")
    if args.setting == "cross" and args.start_sub is None and args.end_sub is None:
        raise ValueError(f"Invalid value seting for args.setting: {args.setting}. Expected values for start_sub or end_sub.")
    
    # Load data
    if args.modality == "single":
        dl, eval_dl = load_single_modal(args, directory = '../processedData/v2', type = "train_test")
    elif args.modality == "multi":
        imu_model = SimpleCNN(classnum=args.num_class)
        imu_model.to(device)
        dl, eval_dl = load_multi_modal(args, directory = '../processedData/v2', type = "train_test")
    else:
        raise ValueError(f"Invalid value for args.modality: {args.modality}. Expected 'single' or 'multi'.")
    
    # optimizer & scheduler
    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.modality == "multi":
        optimizer_imu = torch.optim.Adam(imu_model.parameters(), lr=5e-5, weight_decay=1e-7)
    # phases of lr schedule: exponential increase, constant lr, linear decrease, fine-tune
    schedule_lambda = \
        exp_warmup_linear_down(args.warm_up_len, args.ramp_down_len, args.ramp_down_start, args.last_lr_value)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)
    if args.modality == "multi":
        scheduler_imu = torch.optim.lr_scheduler.LambdaLR(optimizer_imu, schedule_lambda)

    name = None
    accuracy, val_loss = float('NaN'), float('NaN')
    min_val_loss = float("inf")
    early_stopping = EarlyStopping(patience=3, min_delta=0)

    for epoch in range(args.n_epochs):
        mel.train()
        model.train()
        if args.modality == "multi":
            imu_model.train()
        train_stats = dict(train_loss=list())
        pbar = tqdm(dl)
        pbar.set_description("Epoch {}/{}: accuracy: {:.4f}, val_loss: {:.4f}"
                             .format(epoch + 1, args.n_epochs, accuracy, val_loss))
        for batch in pbar:
            if args.modality == "single":
                x, y = batch
            else: 
                x, i, y = batch
            bs = x.size(0)
            x, y = x.to(device), y.to(device)
            if args.modality == "multi": 
                i = i.to(device)
            x = _mel_forward(x, mel)

            if args.mixup_alpha:
                rn_indices, lam = mixup(bs, args.mixup_alpha)
                lam = lam.to(x.device)
                x = x * lam.reshape(bs, 1, 1, 1) + \
                    x[rn_indices] * (1. - lam.reshape(bs, 1, 1, 1))
                y_hat, _ = model(x)
                samples_loss = (F.cross_entropy(y_hat, y, reduction="none") * lam.reshape(bs) +
                                F.cross_entropy(y_hat, y[rn_indices], reduction="none") * (
                                            1. - lam.reshape(bs)))

            else:
                y_hat, features = model(x)
                if args.modality == "multi":
                    y_hat, features_all = imu_model([i.float(),features])
                samples_loss = F.cross_entropy(y_hat, y, reduction="none")

            # loss
            loss = samples_loss.mean()

            # append training statistics
            train_stats['train_loss'].append(loss.detach().cpu().numpy())

            # Update Model
            loss.backward()
            optimizer.step()
            if args.modality == "multi":
                optimizer_imu.step()
            optimizer.zero_grad()
            if args.modality == "multi":
                optimizer_imu.zero_grad()
        # Update learning rate
        scheduler.step()
        if args.modality == "multi":
            scheduler_imu.step()

        # evaluate
        if args.modality == "single":
            accuracy, val_loss, mean_ap, precision, recall, f1_scores = _test(model, mel, eval_dl, device)
        if args.modality == "multi":
            accuracy, val_loss, mean_ap, precision, recall, f1_scores = _test(model, mel, eval_dl, device, imu_model)
        # Check early stopping conditions
        early_stopping(val_loss)
        if early_stopping.early_stop and epoch > 7:
            print("Early stopping triggered")
            break

        # log train and validation statistics
        wandb.log({"train_loss": np.mean(train_stats['train_loss']),
                   "accuracy": accuracy,
                   "val_loss": val_loss,
                   "map": mean_ap,
                   "cough_f1": f1_scores[0],
                   "speech_f1": f1_scores[1],
                   "cough_pre": precision[0],
                   "speech_pre": precision[1],
                   "cough_rec": recall[0],
                   "speech_rec": recall[1]
                   })

        # remove previous model (we try to not flood your hard disk) and save latest model
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_epoch_data = {"best_train_loss": np.mean(train_stats['train_loss']),
                   "best_accuracy": accuracy,
                   "best_val_loss": val_loss,
                   "best_map": mean_ap,
                   "best_cough_f1": f1_scores[0],
                   "best_speech_f1": f1_scores[1],
                   "best_cough_pre": precision[0],
                   "best_speech_pre": precision[1],
                   "best_cough_rec": recall[0],
                   "best_speech_rec": recall[1]
                   }
            if name is not None:
                os.remove(os.path.join(wandb.run.dir, name))
            name = f"mn{str(width).replace('.', '')}_mydata_epoch_{epoch}.pt"
            print(name)
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, name))
    wandb.log(best_epoch_data)


def _test(model, mel, eval_loader, device, imu_model = None):
    model.eval()
    mel.eval()

    targets = []
    outputs = []
    losses = []
    pbar = tqdm(eval_loader)
    pbar.set_description("Validating")
    for batch in pbar:
        if args.modality == "single":
            x, y = batch     
        else: 
            x, i, y = batch
            x, i, y = x.to(device), i.to(device), y.to(device)

        with torch.no_grad():
            x = _mel_forward(x, mel)
            y_hat, features = model(x)
            if args.modality == "multi":
                y_hat, features_all = imu_model([i.float(),features])
        targets.append(y.cpu().numpy())
        outputs.append(y_hat.float().cpu().numpy())
        losses.append(F.cross_entropy(y_hat, y).cpu().numpy())

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    losses = np.stack(losses)
    # print("Targets shape:", targets.shape)
    # print("Outputs shape:", outputs.shape)

    accuracy = metrics.accuracy_score(targets.argmax(axis=1), outputs.argmax(axis=1))
    # Binarize the output
    y_true_bin = label_binarize(targets.argmax(axis=1), classes=np.unique(targets.argmax(axis=1)))

    # Calculate average precision for each class
    aps = []
    for i in range(y_true_bin.shape[1]):
        ap = metrics.average_precision_score(y_true_bin[:, i], outputs[:, i])
        aps.append(ap)

    # Calculate mean Average Precision
    mean_ap = np.mean(aps)
    # mean_ap = metrics.average_precision_score(targets, outputs)

    conf_matrix = metrics.confusion_matrix(targets.argmax(axis=1), outputs.argmax(axis=1))
    # Calculating TP, FP, FN for each class
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP

    # Calculating precision and recall for each class
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # Calculating F1 score for each class
    f1_scores = 2 * (precision * recall) / (precision + recall)

    return accuracy, losses.mean(), mean_ap, precision, recall, f1_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--experiment_name', type=str, default="MyDataset")
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--num_class', type=int, default=9)
    parser.add_argument('--modality', type=str, default="multi") # single / multi
    parser.add_argument('--setting', type=str, default="cross") # in / cross

    # training
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--model_name', type=str, default="mn10_as")
    parser.add_argument('--pretrain_final_temp', type=float, default=1.0)  # for DyMN
    parser.add_argument('--model_width', type=float, default=1.0)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--se_dims', type=str, default="c")
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--mixup_alpha', type=float, default=0)
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
    parser.add_argument('--start_sub', type=int, default=None)
    parser.add_argument('--end_sub', type=int, default=None)

    args = parser.parse_args()
    train(args)
