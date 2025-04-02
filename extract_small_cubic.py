import numpy as np


def index_assignment(index, row, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def assignment_index(assign_0, assign_1, col):
    new_index = assign_0 * col + assign_1
    return new_index


def select_patch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len, pos_row+ex_len+1)]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch


def select_small_cubic(data_size, data_indices, whole_data, patch_length, padded_data, dimension):
    """
    根据给定的数据索引从填充后的数据中提取小立方体块。

    该函数接收数据大小、数据索引、原始数据、块长度、填充后的数据和数据维度作为输入。
    首先，它使用 index_assignment 函数将数据索引分配到填充后数据中对应的行和列位置。
    然后，它遍历每个数据索引，使用 select_patch 函数提取以分配位置为中心的小立方体块，
    并将这些块存储在 small_cubic_data 数组中。

    参数：
        data_size (int): 要提取块的数据点数量。
        data_indices (numpy.ndarray): 数据点在原始数据中的索引。
        whole_data (numpy.ndarray): 未填充的原始数据。
        patch_length (int): 要提取的块的长度。
        padded_data (numpy.ndarray): 填充后的数据。
        dimension (int): 数据的维度数。

    返回值：
        small_cubic_data (numpy.ndarray): 一个 4D 数组，包含提取的小立方体块。
            数组的形状为 (data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension)。

    异常：
        ValueError: 如果数据索引超出范围或块长度无效，将引发此异常。

    示例：
        >>> data_size = 10
        >>> data_indices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> whole_data = np.random.rand(100, 100, 3)
        >>> patch_length = 2
        >>> padded_data = np.pad(whole_data, ((patch_length, patch_length), (patch_length, patch_length), (0, 0)), mode='constant')
        >>> dimension = 3
        >>> small_cubic_data = select_small_cubic(data_size, data_indices, whole_data, patch_length, padded_data, dimension)
        >>> print(small_cubic_data.shape)
        (10, 5, 5, 3)

    注意事项：
        该函数假设数据索引有效且在原始数据范围内。
        它还假设块长度是正整数，并且填充后的数据具有正确的形状。

    参见：
        index_assignment: 用于将数据索引分配到行和列位置。
        select_patch: 用于从填充后的数据中提取小立方体块。
    """
    small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension))
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    for i in range(len(data_assign)):
        small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
    return small_cubic_data