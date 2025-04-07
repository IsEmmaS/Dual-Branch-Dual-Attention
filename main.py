import numpy as np
import time
import collections
import torch
import datetime
import argparse

from sklearn import metrics, preprocessing
from torch import optim
from pathlib import Path
from network import DBDA_network_MISH
from train import train
from record import record_output
from process import (
    aa_and_each_accuracy,
    sampling,
    load_dataset,
    generate_png,
    get_dataloader,
)

PWD = Path(__file__).resolve().parent
IMAGE_FOLDER = f"{PWD}/records/figures"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser=argparse.ArgumentParser(description="DBDA Network Training Script")
parser.add_argument("dataset", type=str, help="Dataset name (IN, UP, BS, SV, PC, DN, DN_1, WHL, HC, HH, KSC)")
args = parser.parse_args()

seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]

day_str = datetime.datetime.now().strftime("%m_%d_%H_%M")


dataset_str = args.dataset.upper()
Dataset = dataset_str.upper()
data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT = load_dataset(Dataset)
print(data_hsi.shape)

image_x, image_y, BAND = data_hsi.shape
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]))
CLASSES_NUM = max(gt)
print("The class numbers of the HSI data is:", CLASSES_NUM)
print("Importing Setting Parameters")
ITER = 5
PATCH_LENGTH = 4
lr, num_epochs, batch_size = 0.0005, 200, 16
loss = torch.nn.CrossEntropyLoss()
img_rows = 2 * PATCH_LENGTH + 1
img_cols = 2 * PATCH_LENGTH + 1
img_channels = data_hsi.shape[2]
ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]
VAL_SIZE = int(TRAIN_SIZE)
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
KAPPA = []
overall_acc_list = []
average_acc_list = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))
data = preprocessing.scale(data)
data_ = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
whole_data = data_
padded_data = np.lib.pad(
    whole_data,
    ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
    "constant",
    constant_values=0,
)
net = DBDA_network_MISH(BAND, CLASSES_NUM)
optimizer = optim.Adam(net.parameters(), lr=lr, amsgrad=False)
time_1 = int(time.time())
train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
_, total_indices = sampling(1, gt)
TRAIN_SIZE = len(train_indices)
print("Train size: ", TRAIN_SIZE)
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
print("Test size: ", TEST_SIZE)
VAL_SIZE = int(TRAIN_SIZE)
print("Validation size: ", VAL_SIZE)
print("Selecting Small Pieces from the Original Cube Data")
train_loader, valida_loader, test_loader, all_loader = get_dataloader(
    TRAIN_SIZE,
    train_indices,
    TEST_SIZE,
    test_indices,
    TOTAL_SIZE,
    total_indices,
    VAL_SIZE,
    whole_data,
    PATCH_LENGTH,
    padded_data,
    batch_size,
    gt,
)
for iter in range(ITER):
    torch.cuda.empty_cache()
    np.random.seed(seeds[iter])
    print("iter:", iter)
    tic1 = time.perf_counter()
    train(
        net,
        train_loader,
        valida_loader,
        loss,
        optimizer,
        device,
        dataset_str,
        IMAGE_FOLDER,
        iter,
        epochs=num_epochs,
    )
    toc1 = time.perf_counter()
    pred_test_fdssc = []
    tic2 = time.perf_counter()
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            net.eval()
            y_hat = net(X)
            pred_test_fdssc.extend(np.array(net(X).cpu().argmax(axis=1)))
    toc2 = time.perf_counter()
    collections.Counter(pred_test_fdssc)
    gt_test = gt[test_indices] - 1
    overall_acc_fdssc = metrics.accuracy_score(pred_test_fdssc, gt_test[:-VAL_SIZE])
    confusion_matrix_fdssc = metrics.confusion_matrix(
        pred_test_fdssc, gt_test[:-VAL_SIZE]
    )
    each_acc_fdssc, average_acc_fdssc = aa_and_each_accuracy(confusion_matrix_fdssc)
    kappa = metrics.cohen_kappa_score(pred_test_fdssc, gt_test[:-VAL_SIZE])
    torch.save(net.state_dict(), f"{PWD}/pth/{str(round(overall_acc_fdssc, 3))}.pt")
    KAPPA.append(kappa)
    overall_acc_list.append(overall_acc_fdssc)
    average_acc_list.append(average_acc_fdssc)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - tic2)
    ELEMENT_ACC[(iter), :] = each_acc_fdssc
print(f"{net.name}  Training Finished")
record_output(
    overall_acc_list,
    average_acc_list,
    KAPPA,
    ELEMENT_ACC,
    TRAINING_TIME,
    TESTING_TIME,
    f"{PWD}/records/log/{net.name}_{day_str}_{Dataset}"
    f"_split{VALIDATION_SPLIT}_lr{lr}.txt",
)
generate_png(all_loader, net, gt_hsi, Dataset, device, total_indices)
