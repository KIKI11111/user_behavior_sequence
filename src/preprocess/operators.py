
import numpy as np
import random
from collections import Counter
from src.constants import Constants


def random_cut_sequence(sequence: dict, min_m_size =16, min_w_size = 16):
    """
    随机将序列切分为[0:M] , [M,M+W]的表征窗口和重建窗口， 分别作为表征的输入和预训练的label
    :param sequence:
    :param min_m_size:
    :param min_w_size:
    :return:
    """
    keys = list(sequence.keys())
    sequence_length = len(sequence[keys[0]])
    m = random.randint(max(sequence_length//2, min_m_size), sequence_length - min_w_size + 1)
    m_seq = {
        k: v[:, m+1] for k, v in sequence.items()
    }
    w_seq = {
        f"w_{k}": v[m+1:] for k, v in sequence.items()
    }

    return {**m_seq, **w_seq}


def apiid_seq2distribution(apiid, val_size):
    """
    将apiid的序列转化为分布
    loss是分布的似然函数，可以直接使用nn.CrossEntropyloss()
    :param apiid:
    :param val_size:
    :return:
    """
    distribution = np.zero((val_size,), dtype=float)
    apiid_cnt = Counter(apiid)
    for k, v in apiid_cnt.items():
        distribution[k] = v/len(apiid)
    return distribution


def interval_seq2distribution(interval):
    """
    将间隔转化为连续的分布
    假设： sqrt(interval) ~ N(alpha, sigma)
    :param interval:
    :return:
    """
    sqrt_interval = np.sqrt(np.array(interval))
    return np.mean(sqrt_interval), np.std(sqrt_interval)


def apiid2order(apiid_list):
    apiid2ids = {}
    order_list = []
    for item in apiid_list:
        apiid = len(apiid2ids)
        if apiid2ids.get(item) is not None:
            apiid = apiid2ids.get(item)
        else:
            apiid2ids[item] = apiid
        order_list.append(apiid)
    return order_list


def bucket_data(interval):
    """
    间隔分桶函数
    :param interval:
    :return:
    """
    bins = np.array(Constants.interval_bucket_bins)
    bins = np.array(bins)
    interval = np.array(interval)

    # 分桶索引
    indices = np.searchsorted(bins, interval, side='left')
    indices[interval == bins[0]] = 0

    return list(indices)
