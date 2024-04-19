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

class SimpleCNN(nn.Module):
    def __init__(self, classnum = 3):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=4, padding=1)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=4, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(160, 128)
        self.fc2 = nn.Linear(128+960, 960)
        self.fc3 = nn.Linear(960, classnum)
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
    

def train(args):
    # Train Models for Acoustic Scene Classification

    # logging is done using wandb
    wandb.init(
        project="Mydataset",
        notes="Train on my dataset.",
        tags=["Environmental Sound Classification", "Fine-Tuning"],
        config=args,
        name=args.experiment_name
    )
    wandb.define_metric("val_loss", summary="min")

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
    model.to(device)
    imu_model = SimpleCNN(classnum=args.num_class)
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

    eval_indices = []
    for i, (_, _, label) in enumerate(evaldataset):
        if np.sum(label) != 0:
            eval_indices += [i]  
    eval_dl = DataLoader(dataset= Subset(evaldataset, eval_indices),
                         worker_init_fn=worker_init_fn,
                         num_workers=args.num_workers,
                         batch_size=args.batch_size)

    # optimizer & scheduler
    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # phases of lr schedule: exponential increase, constant lr, linear decrease, fine-tune
    schedule_lambda = \
        exp_warmup_linear_down(args.warm_up_len, args.ramp_down_len, args.ramp_down_start, args.last_lr_value)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)

    name = None
    accuracy, val_loss = float('NaN'), float('NaN')
    min_val_loss = float("inf")
    early_stopping = EarlyStopping(patience=5, min_delta=0)

    for epoch in range(args.n_epochs):
        mel.train()
        model.train()
        imu_model.train()
        train_stats = dict(train_loss=list())
        pbar = tqdm(dl)
        pbar.set_description("Epoch {}/{}: accuracy: {:.4f}, val_loss: {:.4f}"
                             .format(epoch + 1, args.n_epochs, accuracy, val_loss))
        for batch in pbar:
            x, i, y = batch
            bs = x.size(0)
            x, i, y = x.to(device), i.to(device), y.to(device)
            x = _mel_forward(x, mel)
            _, features = model(x)
            y_hat, features_all = imu_model([i.float(),features])
 
            samples_loss = F.cross_entropy(y_hat, y, reduction="none")

            # loss
            loss = samples_loss.mean()

            # append training statistics
            train_stats['train_loss'].append(loss.detach().cpu().numpy())

            # Update Model
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Update learning rate
        scheduler.step()

        # evaluate
        accuracy, val_loss, mean_ap, precision, recall, f1_scores = _test(model, imu_model, mel, eval_dl, device)
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
                os.remove(os.path.join(wandb.run.dir, imu_name))
            name = f"mn{str(width).replace('.', '')}_mydata_epoch_{epoch}.pt"
            print(name)
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, name))
            imu_name = f"mn{str(width).replace('.', '')}_mydata_epoch_{epoch}_imu.pt"
            torch.save(imu_model.state_dict(), os.path.join(wandb.run.dir, imu_name))
    wandb.log(best_epoch_data)


def _mel_forward(x, mel):
    old_shape = x.size()
    x = x.reshape(-1, old_shape[2])
    x = mel(x)
    x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
    return x


def _test(model, imu_model, mel, eval_loader, device):
    model.eval()
    imu_model.eval()
    mel.eval()

    targets = []
    outputs = []
    losses = []
    pbar = tqdm(eval_loader)
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
    parser.add_argument('--warm_up_len', type=int, default=5)
    parser.add_argument('--ramp_down_start', type=int, default=5)
    parser.add_argument('--ramp_down_len', type=int, default=20)
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
    args = parser.parse_args()
    train(args)
