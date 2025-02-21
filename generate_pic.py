import numpy as np
import matplotlib.pyplot as plt
from operator import truediv
import scipy.io as sio
import torch
import math
import extract_small_cubic
import torch.utils.data as Data
import time
from pathlib import Path

PWD = Path(__file__).resolve().parent


def load_dataset(Dataset):
    """
    根据数据集名称加载高光谱数据集。

    参数:
        Dataset(str): 数据集名称，可选值有'IN'、'UP'、'PC'、'SV'、'KSC'、'BS'、'DN'、'DN_1'、'WHL'、'HC'、'HH'等。

    返回:
        tuple: data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT
    """
    _dataset_info = {
        "IN": {
            "data_path": f"{PWD}/datasets/Indian_pines_corrected.mat",
            "gt_path": f"{PWD}/datasets/Indian_pines_gt.mat",
            "data_key": "indian_pines_corrected",
            "gt_key": "indian_pines_gt",
            "TOTAL_SIZE": 10249,
            "VALIDATION_SPLIT": 0.95,
        },
        "UP": {
            "data_path": f"{PWD}/datasets/PaviaU.mat",
            "gt_path": f"{PWD}/datasets/PaviaU_gt.mat",
            "data_key": "paviaU",
            "gt_key": "paviaU_gt",
            "TOTAL_SIZE": 42776,
            "VALIDATION_SPLIT": 0.91,
        },
        "PC": {
            "data_path": f"{PWD}/datasets/Pavia.mat",
            "gt_path": f"{PWD}/datasets/Pavia_gt.mat",
            "data_key": "pavia",
            "gt_key": "pavia_gt",
            "TOTAL_SIZE": 148152,
            "VALIDATION_SPLIT": 0.999,
        },
        "SV": {
            "data_path": f"{PWD}/datasets/Salinas_corrected.mat",
            "gt_path": f"{PWD}/datasets/Salinas_gt.mat",
            "data_key": "salinas_corrected",
            "gt_key": "salinas_gt",
            "TOTAL_SIZE": 54129,
            "VALIDATION_SPLIT": 0.98,
        },
        "KSC": {
            "data_path": f"{PWD}/datasets/KSC.mat",
            "gt_path": f"{PWD}/datasets/KSC_gt.mat",
            "data_key": "KSC",
            "gt_key": "KSC_gt",
            "TOTAL_SIZE": 5211,
            "VALIDATION_SPLIT": 0.91,
        },
        "BS": {
            "data_path": f"{PWD}/datasets/Botswana.mat",
            "gt_path": f"{PWD}/datasets/Botswana_gt.mat",
            "data_key": "Botswana",
            "gt_key": "Botswana_gt",
            "TOTAL_SIZE": 3248,
            "VALIDATION_SPLIT": 0.99,
        },
        "DN": {
            "data_path": f"{PWD}/datasets/Dioni.mat",
            "gt_path": f"{PWD}/datasets/Dioni_gt_out68.mat",
            "data_key": "ori_data",
            "gt_key": "map",
            "TOTAL_SIZE": 20024,
            "VALIDATION_SPLIT": 0.95,
        },
        "DN_1": {
            "data_path": f"{PWD}/datasets/DN_1/Dioni.mat",
            "gt_path": f"{PWD}/datasets/DN_1/Dioni_gt_out68.mat",
            "data_key": "imggt",
            "gt_key": "map",
            "TOTAL_SIZE": 20024,
            "VALIDATION_SPLIT": 0.98,
        },
        "WHL": {
            "data_path": f"{PWD}/datasets/WHL/WHU_Hi_LongKou.mat",
            "gt_path": f"{PWD}/datasets/WHL/WHU_Hi_LongKou_gt.mat",
            "data_key": "WHU_Hi_LongKou",
            "gt_key": "WHU_Hi_LongKou_gt",
            "TOTAL_SIZE": 204542,
            "VALIDATION_SPLIT": 0.99,
        },
        "HC": {
            "data_path": f"{PWD}/datasets/HC/WHU_Hi_HanChuan.mat",
            "gt_path": f"{PWD}/datasets/HC/WHU_Hi_HanChuan_gt.mat",
            "data_key": "WHU_Hi_HanChuan",
            "gt_key": "WHU_Hi_HanChuan_gt",
            "TOTAL_SIZE": 257530,
            "VALIDATION_SPLIT": 0.99,
        },
        "HH": {
            "data_path": f"{PWD}/datasets/HH/WHU_Hi_HongHu.mat",
            "gt_path": f"{PWD}/datasets/HH/WHU_Hi_HongHu_gt.mat",
            "data_key": "WHU_Hi_HongHu",
            "gt_key": "WHU_Hi_HongHu_gt",
            "TOTAL_SIZE": 386693,
            "VALIDATION_SPLIT": 0.99,
        },
    }
    if Dataset not in _dataset_info:
        raise ValueError(f"Invalid dataset name: {Dataset}")
    info = _dataset_info[Dataset]
    data_hsi = sio.loadmat(info["data_path"])[info["data_key"]]
    gt_hsi = sio.loadmat(info["gt_path"])[info["gt_key"]]
    TOTAL_SIZE = info["TOTAL_SIZE"]
    VALIDATION_SPLIT = info["VALIDATION_SPLIT"]
    TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    return data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT


def save_cmap(img, cmap, fname):
    """
    将带有指定colormap的图像保存到文件中。

    参数:
        img (numpy.ndarray): 要保存的输入图像，需带有colormap。
        cmap (str 或 matplotlib.colors.Colormap): 要应用于图像的colormap。可以是字符串（如'viridis'、'plasma'等）
            或matplotlib.colors.Colormap对象。
        fname (str): 保存图像的文件名。

    返回:
        None
     示例:
        >>> import numpy as np
        >>> img = np.random.rand(100, 100)
        >>> save_cmap(img, 'viridis', 'example.png')
    """
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])
    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, cmap=cmap)
    plt.savefig(fname, dpi=height)
    plt.close()


def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(
        ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi
    )
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)
    return 0


def list_to_colormap(x_list):
    """
    将整数列表转换为颜色映射数组。

    该函数将输入的整数列表 `x_list` 转换为一个颜色映射数组 `y`，其中每个整数对应一种特定的颜色。
    颜色映射关系通过字典 `color_map` 定义，字典的键为整数，值为对应的颜色数组（归一化后的 RGB 值）。

    Args:
        x_list (numpy.ndarray): 输入的整数列表，形状为 (n,)，其中 n 为列表长度。

    Returns:
        numpy.ndarray: 输出的颜色映射数组，形状为 (n, 3)，其中每行表示一个颜色的 RGB 值。

    Raises:
        KeyError: 如果 `x_list` 中的某个整数不在 `color_map` 中，将使用默认的黑色 (0, 0, 0)。

    Example:
        >>> x_list = np.array([0, 1, 2, 3, 4])
        >>> y = list_to_colormap(x_list)
        >>> print(y)
        [[1.         0.         0.        ]
         [0.         1.         0.        ]
         [0.         0.         1.        ]
         [1.         1.         0.        ]
         [1.         0.         1.        ]]

    Note:
        该函数假设输入的 `x_list` 是一个一维的 numpy 数组。如果输入不是 numpy 数组，可能会导致错误。
        该函数使用了默认的黑色 (0, 0, 0) 作为未定义整数的颜色，确保输出数组的完整性。

    See Also:
        numpy.array: 用于创建和操作数组。
    """
    color_map = {
        (-1): np.array([0, 0, 0]) / 255.0,
        (0): np.array([255, 0, 0]) / 255.0,
        (1): np.array([0, 255, 0]) / 255.0,
        (2): np.array([0, 0, 255]) / 255.0,
        (3): np.array([255, 255, 0]) / 255.0,
        (4): np.array([255, 0, 255]) / 255.0,
        (5): np.array([0, 255, 255]) / 255.0,
        (6): np.array([200, 100, 0]) / 255.0,
        (7): np.array([0, 200, 100]) / 255.0,
        (8): np.array([100, 0, 200]) / 255.0,
        (9): np.array([200, 0, 100]) / 255.0,
        (10): np.array([100, 200, 0]) / 255.0,
        (11): np.array([0, 100, 200]) / 255.0,
        (12): np.array([150, 75, 75]) / 255.0,
        (13): np.array([75, 150, 75]) / 255.0,
        (14): np.array([75, 75, 150]) / 255.0,
        (15): np.array([255, 100, 100]) / 255.0,
        (16): np.array([0, 0, 0]) / 255.0,
        (17): np.array([100, 100, 255]) / 255.0,
        (18): np.array([255, 150, 75]) / 255.0,
        (19): np.array([75, 255, 150]) / 255.0,
        (20): np.array([150, 75, 255]) / 255.0,
        (21): np.array([50, 50, 50]) / 255.0,
        (22): np.array([100, 100, 100]) / 255.0,
        (23): np.array([150, 150, 150]) / 255.0,
        (24): np.array([200, 200, 200]) / 255.0,
        (25): np.array([250, 250, 250]) / 255.0,
    }
    y = np.array([color_map.get(item, np.array([0, 0, 0]) / 255.0) for item in x_list])
    return y


def generate_iter(
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
    INPUT_DIMENSION,
    batch_size,
    gt,
):
    """
    生成用于训练、验证和测试的迭代器。
    该函数从给定的数据集中提取训练、测试和验证集，并将它们转换为 PyTorch 的 DataLoader 格式，以便在深度学习模型中使用。

    Args:
        TRAIN_SIZE (int): 训练集的大小。
        train_indices (numpy.ndarray): 训练集的索引。
        TEST_SIZE (int): 测试集的大小。
        test_indices (numpy.ndarray): 测试集的索引。
        TOTAL_SIZE (int): 整个数据集的大小。
        total_indices (numpy.ndarray): 整个数据集的索引。
        VAL_SIZE (int): 验证集的大小。
        whole_data (numpy.ndarray): 整个数据集的原始数据。
        PATCH_LENGTH (int): 每个样本的邻域大小。
        padded_data (numpy.ndarray): 填充后的数据。
        INPUT_DIMENSION (int): 输入数据的维度。
        batch_size (int): 每个批次的大小。
        gt (numpy.ndarray): 地面真值（标签）。

    Returns:
        tuple: 包含训练集、验证集、测试集和整个数据集的迭代器 (train_iter, valiada_iter, test_iter, all_iter)。

    Raises:
        ValueError: 如果输入参数的类型或形状不正确。

    Example:
        >>> train_iter, valiada_iter, test_iter, all_iter = generate_iter(
        ...     TRAIN_SIZE=100, train_indices=np.array([1, 2, 3]), TEST_SIZE=50, test_indices=np.array([4, 5, 6]),
        ...     TOTAL_SIZE=150, total_indices=np.array([1, 2, 3, 4, 5, 6]), VAL_SIZE=20,
        ...     whole_data=np.random.rand(150, 10, 10), PATCH_LENGTH=2, padded_data=np.random.rand(154, 10, 10),
        ...     INPUT_DIMENSION=10, batch_size=10, gt=np.array([1, 2, 3, 4, 5, 6])
        ... )
        >>> print(train_iter)
        <torch.utils.data.dataloader.DataLoader object at 0x...>

    Note:
        该函数假设输入的 `train_indices`、`test_indices` 和 `total_indices` 是一维的 numpy 数组。
        该函数使用 `extract_small_cubic.select_small_cubic` 函数提取数据，确保该函数已正确实现。

    See Also:
        extract_small_cubic.select_small_cubic: 用于提取小立方体数据的函数。
        torch.utils.data.DataLoader: 用于创建数据加载器的类。
    """
    gt_all = gt[total_indices] - 1
    y_train = gt[train_indices] - 1
    y_test = gt[test_indices] - 1
    all_data = extract_small_cubic.select_small_cubic(
        TOTAL_SIZE,
        total_indices,
        whole_data,
        PATCH_LENGTH,
        padded_data,
        INPUT_DIMENSION,
    )
    train_data = extract_small_cubic.select_small_cubic(
        TRAIN_SIZE,
        train_indices,
        whole_data,
        PATCH_LENGTH,
        padded_data,
        INPUT_DIMENSION,
    )
    test_data = extract_small_cubic.select_small_cubic(
        TEST_SIZE, test_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION
    )
    x_train = train_data.reshape(
        train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION
    )
    x_test_all = test_data.reshape(
        test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION
    )
    x_val = x_test_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]
    x_test = x_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]
    x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, y1_tensor_train)
    x1_tensor_valida = torch.from_numpy(x_val).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_valida = torch.from_numpy(y_val).type(torch.FloatTensor)
    torch_dataset_valida = Data.TensorDataset(x1_tensor_valida, y1_tensor_valida)
    x1_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test, y1_tensor_test)
    all_data.reshape(
        all_data.shape[0], all_data.shape[1], all_data.shape[2], INPUT_DIMENSION
    )
    all_tensor_data = torch.from_numpy(all_data).type(torch.FloatTensor).unsqueeze(1)
    all_tensor_data_label = torch.from_numpy(gt_all).type(torch.FloatTensor)
    torch_dataset_all = Data.TensorDataset(all_tensor_data, all_tensor_data_label)
    train_iter = Data.DataLoader(
        dataset=torch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0
    )
    valiada_iter = Data.DataLoader(
        dataset=torch_dataset_valida, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_iter = Data.DataLoader(
        dataset=torch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0
    )
    all_iter = Data.DataLoader(
        dataset=torch_dataset_all, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_iter, valiada_iter, test_iter, all_iter


def generate_png(all_iter, net, gt_hsi, Dataset, device, total_indices):
    pred_test = []
    for X, y in all_iter:
        X = X.to(device)
        net.eval()
        pred_test.extend(np.array(net(X).cpu().argmax(axis=1)))
    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)
    for i in range(len(gt)):
        if gt[i] == 0:
            gt[i] = 17
            x_label[i] = 16
    gt = gt[:] - 1
    x_label[total_indices] = pred_test
    x = np.ravel(x_label)
    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)
    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    timestamp = time.strftime("-%y-%m-%d-%H.%M")
    path = f"{PWD}/" + net.name
    print(path)
    classification_map(
        y_re,
        gt_hsi,
        300,
        path + "/classification_maps/" + Dataset + "_" + net.name + timestamp + ".png",
    )
    classification_map(
        gt_re,
        gt_hsi,
        300,
        path + "/classification_maps/" + Dataset + timestamp + "_gt.png",
    )
    print("------Get classification maps successful-------")
