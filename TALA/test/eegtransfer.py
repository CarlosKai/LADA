import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split

import torch


#Adjust the seed for controlling the randomness
seed=1234

# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)
main_folder = "C:\\Users\89368\Desktop\LALA\data"

if not os.path.exists(main_folder):
    os.mkdir(main_folder)

save_folder = main_folder + "/EEG"
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

for i in range(20):
    path = "C:\\Users\89368\Desktop\LALA\data\EEG/train_" + str(i) + ".pt"
    checkpoint = torch.load(path)
    path_test = "C:\\Users\89368\Desktop\LALA\data\EEG/test_" + str(i) + ".pt"
    checkpoint_test = torch.load(path_test)

    all_X_test = checkpoint_test['samples']
    all_y_test = checkpoint_test['labels']

    # check the unique labels, if exists, we will add them to the test split (but not val).
    unique_labels, unique_counts = np.unique(checkpoint_test['labels'], return_counts=True)

    if 1 in unique_counts:
        labels_one = unique_labels[unique_counts == 1]
        print("Problem at Subject " + str(i) + " for " + str(len(labels_one)) + " classes.")
        for label in labels_one:
            all_X_test = all_X_test[all_y_test != label]
            all_y_test = all_y_test[all_y_test != label]

    X_val, X_test, y_val, y_test = train_test_split(all_X_test, all_y_test, stratify=all_y_test, test_size=0.5,
                                                    random_state=seed)

    if 1 in unique_counts:
        labels_one = unique_labels[unique_counts == 1]
        for label in labels_one:
            X_test = np.concatenate((X_test, checkpoint_test['samples'][checkpoint_test['labels'] == label]))
            y_test = np.concatenate((y_test, checkpoint_test['labels'][checkpoint_test['labels'] == label]))

    subject_folder = save_folder + "/subject_" + str(i)

    if not os.path.exists(subject_folder):
        os.mkdir(subject_folder)

    # Save all the splits
    # train
    np.save(subject_folder + "/timeseries_train.npy", checkpoint['samples'])
    np.save(subject_folder + "/label_train.npy", checkpoint['labels'])
    # val
    np.save(subject_folder + "/timeseries_val.npy", X_val)
    np.save(subject_folder + "/label_val.npy", y_val)
    # test
    np.save(subject_folder + "/timeseries_test.npy", X_test)
    np.save(subject_folder + "/label_test.npy", y_test)


