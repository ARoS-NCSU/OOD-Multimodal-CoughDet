import os
import mat73
import numpy as np
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from helpers.init import worker_init_fn
from mydataloader.mydataset import MyDataset
from mydataloader.mydataset_multi import MyDataset as MyDatasetMulti


def load_single_modal(args, directory = '../processedData/v2', type = "train_test"):
    file_names = [f for f in os.listdir(directory) if f.endswith('.mat')]
    all_data, all_labels = [], []
    file_names = sorted(file_names)[:-3]
    for file_name in file_names:
        mat = mat73.loadmat(os.path.join(directory, file_name))
        data = mat['audio_data'] 
        label = mat['label']
        all_data.append(np.transpose(data))
        all_labels.append(label)
    all_data = np.array(all_data, dtype=object)
    all_labels = np.array(all_labels, dtype=object)

    # dataloader
    if args.setting == "in":
        train_dataset, train_labels = [], []
        test_dataset, test_labels = [], []
        split_prop = args.split_prop
        for i in range(len(all_data)):
            sub_data = all_data[i]
            sub_label = all_labels[i]
            d_split_point = int(len(sub_data)*split_prop)
            d_end_point = int(len(sub_data)*(split_prop+0.3))
            l_split_point = int(len(sub_label)*split_prop)
            l_end_point = int(len(sub_label)*(split_prop+0.3))
            if split_prop < 0.7:
                train_dataset.append(np.concatenate((sub_data[:d_split_point], sub_data[d_end_point:]), axis=0))
                train_labels.append(np.concatenate((sub_label[:l_split_point,:], sub_label[l_end_point:,:]), axis=0))
                test_dataset.append(sub_data[d_split_point:d_end_point])
                test_labels.append(sub_label[l_split_point:l_end_point,:])
            else:
                train_dataset.append(sub_data[:d_split_point])
                train_labels.append(sub_label[:l_split_point,:])
                test_dataset.append(sub_data[d_split_point:])
                test_labels.append(sub_label[l_split_point:,:])

        traindata = np.array(train_dataset,dtype=object)
        evaldata = np.array(test_dataset,dtype=object)
        traindataset = MyDataset(datasource = traindata,
                                    labels = train_labels,
                                    window_size=1.5, 
                                    step_size=0.5, 
                                    transform=None,
                                    classes_num = args.num_class)
        evaldataset = MyDataset(datasource = evaldata, 
                                    labels = test_labels,
                                    window_size=1.5, 
                                    step_size=0.5, 
                                    transform=None,
                                    classes_num = args.num_class)
    elif args.setting == "cross":
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
    else:
        raise ValueError(f"Invalid value for args.modality: {args.setting}. Expected 'in' or 'cross'.")

    train_indices = []
    for i, (_, label) in enumerate(traindataset):
        if np.sum(label) != 0:
            train_indices += [i]

    dl = DataLoader(dataset=Subset(traindataset, train_indices),
                    worker_init_fn=worker_init_fn,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size,
                    shuffle=True)
    if type == "train_test":
        eval_indices = []
        for i, (_, label) in enumerate(evaldataset):
            if np.sum(label) != 0:
                eval_indices += [i]  
        eval_dl = DataLoader(dataset= Subset(evaldataset, eval_indices),
                                worker_init_fn=worker_init_fn,
                                num_workers=args.num_workers,
                                
                                batch_size=args.batch_size)
    elif type == "ood":
        eval_dl = DataLoader(dataset= evaldataset,
                         worker_init_fn=worker_init_fn,
                         num_workers=args.num_workers,
                         batch_size=args.batch_size)
    
    return dl, eval_dl

def load_multi_modal(args, directory = '../processedData/v2', type = "train_test"):
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
    if args.setting == "in":
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
        traindataset = MyDatasetMulti(datasource = [traindata, train_acc, train_gyro, train_mag],
                                labels = train_labels,
                                window_size=1.5, 
                                step_size=0.5, 
                                transform=None,
                                classes_num = args.num_class)
        evaldataset = MyDatasetMulti(datasource = [evaldata, test_acc, test_gyro, test_mag], 
                                    labels = test_labels,
                                    window_size=1.5, 
                                    step_size=0.5, 
                                    transform=None,
                                    classes_num = args.num_class)
    elif args.setting == "cross":
        start_sub, end_sub = args.start_sub, args.end_sub
        if start_sub and end_sub:
            print("test set:", file_names[start_sub:end_sub])
            traindataset = MyDatasetMulti(datasource = [np.concatenate((all_data[0:start_sub], all_data[end_sub:]), axis=0),
                                                all_acc[0:start_sub] + all_acc[end_sub:],
                                                all_gyro[0:start_sub] + all_gyro[end_sub:],
                                                all_mag[0:start_sub] + all_mag[end_sub:]],
                                    labels = np.concatenate((all_labels[0:start_sub], all_labels[end_sub:]), axis=0),
                                    window_size=1.5, 
                                    step_size=0.5, 
                                    transform=None,
                                    classes_num = args.num_class)

            # evaluation loader
            evaldataset = MyDatasetMulti(datasource = [all_data[start_sub:end_sub], all_acc[start_sub:end_sub],
                                                all_gyro[start_sub:end_sub],all_mag[start_sub:end_sub]],
                                    labels = all_labels[start_sub:end_sub],
                                    window_size=1.5, 
                                    step_size=0.5, 
                                    transform=None,
                                    classes_num = args.num_class)
        elif start_sub:
            print("test set:", file_names[start_sub:])
            traindataset = MyDatasetMulti(datasource = [all_data[:start_sub], all_acc[:start_sub], all_gyro[:start_sub], all_mag[:start_sub]],
                                    labels = all_labels[:start_sub],
                                    window_size=1.5, 
                                    step_size=0.5, 
                                    transform=None,
                                    classes_num = args.num_class)

            # evaluation loader
            evaldataset = MyDatasetMulti(datasource = [all_data[start_sub:], all_acc[start_sub:],all_gyro[start_sub:],all_mag[start_sub:]],
                                        labels = all_labels[start_sub:],
                                        window_size=1.5, 
                                        step_size=0.5, 
                                        transform=None,
                                        classes_num = args.num_class)
        else:
            print("test set:", file_names[:end_sub])
            traindataset = MyDatasetMulti(datasource = [all_data[end_sub:],all_acc[end_sub:],all_gyro[end_sub:],all_mag[end_sub:]],
                                    labels = all_labels[end_sub:],
                                    window_size=1.5, 
                                    step_size=0.5, 
                                    transform=None,
                                    classes_num = args.num_class)

            # evaluation loader
            evaldataset= MyDatasetMulti(datasource = [all_data[:end_sub],all_acc[:end_sub],all_gyro[:end_sub],all_mag[:end_sub]], 
                                    labels = all_labels[:end_sub],
                                    window_size=1.5, 
                                    step_size=0.5, 
                                    transform=None,
                                classes_num = args.num_class)
    else:
        raise ValueError(f"Invalid value for args.modality: {args.setting}. Expected 'in' or 'cross'.")
    
    train_indices = []
    for i, (_, _, label) in enumerate(traindataset):
        if np.sum(label) != 0:
            train_indices += [i]
    
    dl = DataLoader(dataset=Subset(traindataset, train_indices),
                    worker_init_fn=worker_init_fn,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size,
                    shuffle=True)

    if type == "train_test":
        eval_indices = []
        for i, (_, _, label) in enumerate(evaldataset):
            if np.sum(label) != 0:
                eval_indices += [i]  
        eval_dl = DataLoader(dataset= Subset(evaldataset, eval_indices),
                            worker_init_fn=worker_init_fn,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size)
    
    elif type == "ood":
        eval_dl = DataLoader(dataset= evaldataset,
                         worker_init_fn=worker_init_fn,
                         num_workers=args.num_workers,
                         batch_size=args.batch_size)

    return dl, eval_dl