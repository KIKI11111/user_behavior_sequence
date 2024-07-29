import numpy as np
from src.preprocess.operators import *
from src.constants import Constants
import torch
from torch import nn


def sequence_preprocess_map_func(row, seq_max_length):
    """
    序列预处理函数， 传入dataset中map方法中
    完成以下功能：
        1.将apiid转为顺序的id
        2.计算segment(interval需要后续计算label使用，不在这一步离散化)
    :param row:
    :param seq_max_length:
    :return:
    """
    apiid = row['apiid']
    if isinstance(apiid, str):
        apiid = apiid.split(',')
    apiid_list = apiid[:seq_max_length]
    apiid_order_list = apiid2order(apiid_list)
    timestamp = row['timestamp']
    if isinstance(timestamp, str):
        timestamp = timestamp.split(',')

    zipped = zip(apiid_list, timestamp)
    sorted_zip = sorted(zipped, key=lambda x: x[1])
    apiid_list, timestamp = zip(*sorted_zip)

    interval = np.array(timestamp[:seq_max_length], dtype=int)
    interval_list = list(np.array(interval[1:]) - np.array(interval[:-1])) + [0]
    segment_list = list(np.cumsum([1 if item > Constants.segment_interval else 0 for item in interval_list]))
    interval_list = [0 if item > Constants.segment_interval else item for item in interval_list]

    return {
        'apiid': apiid_order_list,
        'interval': interval_list,
        'segment': segment_list
    }


def sequence_preprocess_batched_map_func(batch, seq_max_length):
    """
    用于序列预处理的函数，按batch传入dataset中的map方法中
    sequence_preprocess_map_func 的batch调用
    input: batch = {
            'apiid': [
                '1,2,3,4,5,6',
                '7,8,9,10,11'
            ],
            'timestamp': [
                '1,4,7,12,16,20',
                '2,5,8,12,18'
            ]
        }
    output:
        {
            'apiid': [
                [1,2,3,4],
                []
            ],
            'interval':[
                [],
                []
            ],
            'segment':[
                [],
                []
            ]
        }
    """
    new_batch = [sequence_preprocess_map_func({'apiid':apiid, 'timestamp': timestamp}, seq_max_length) for apiid, timestamp in zip(batch['apiid'], batch['timestamp'])]
    new_batch = {
        'apiid': [row['apiid'] for row in new_batch],
        'interval': [row['interval'] for row in new_batch],
        'segment': [row['segment'] for row in new_batch]
    }

    return new_batch


def interface_sequence_collect_fn(batch, min_m_size, min_w_size, seq_max_length):
    """
    将序列计算表征和label， 计算好快呀送入batch中的特征
    完成以下功能：
        1.将sequence_length大小的输入序列随机切分为m和w两个子序列，sequence_length = len(m) + len(w)
        2.将interval离散化
        3.添加<mask>求mask
        4.将w序列的apiid转为分布
        5.添加cls
        6.包装为tensor
    input:
        dataset({
        features: ['entity_value', 'timestamp', 'apiid', 'cnt', 'interval', 'segment']
        num_rows:5000
        })
    output:
        dict {
        'input_apiid_sequence' : tensor(shape(batch, seq_max_length, seq_max_length), dtype=torch.int)
        'input_interval_sequence' : tensor(shape(batch, seq_max_length, seq_max_length), dtype=torch.int)
        'input_segment' : tensor(shape(batch, seq_max_length, seq_max_length), dtype=torch.int)
        'attention_mask' : tensor(shape(batch, seq_max_length), dtype=torch.bool)
        'label_apiid_dist' : tensor(shape(batch, seq_max_length), dtype=torch.float32)
        'label_interval_mu' : tensor(shape(batch,), dtype=torch.float32)
        'label_interval_sigma' : tensor(shape(batch,), dtype=torch.float32)
        }
    :param batch:
    :param min_m_size:
    :param min_w_size:
    :param seq_max_length:
    :return:
    """
    special_tokens = {
        '<cls>': (seq_max_length, Constants.interval_bucket_num, seq_max_length),
        '<mask>': (seq_max_length + 1, Constants.interval_bucket_num + 1, seq_max_length + 1)
    }

    input_apiid_sequence = []
    input_interval_sequence = []
    input_segment = []
    attention_masks = []
    label_apiid_dist = []
    label_interval_dist = []

    batch = [random_cut_sequence(row, min_m_size, min_w_size) for row in batch]

    for row in batch:
        w_apiid_dist = apiid_seq2distribution(row['w_apiid'], seq_max_length)
        w_interval_dist = apiid_seq2distribution(bucket_data(row['w_interval']), Constants.interval_bucket_num)

        m_interval = bucket_data[row['interval']]
        m_apiid = row['apiid']
        m_segment = row['segment']
        if len(m_apiid) < seq_max_length:
            m_apiid = m_apiid + [special_tokens['<mask>'][0] for _ in range(seq_max_length - len(m_apiid))]
            m_interval = m_interval + [special_tokens['<mask>'][1] for _ in range(seq_max_length - len(m_interval))]
            m_segment = m_segment + [special_tokens['<mask>'][2] for _ in range(seq_max_length - len(m_segment))]

        m_apiid = [special_tokens['<cls>'][0]] + m_apiid
        m_interval = [special_tokens['<cls>'][0]] + m_interval
        m_segment = [special_tokens['<cls>'][0]] + m_segment

        seq_mask = [item == special_tokens['<mask>'][0] for item in m_apiid]

        input_apiid_sequence.append(m_apiid)
        input_interval_sequence.append(m_interval)
        input_segment.append(m_segment)
        attention_masks.append(seq_mask)
        label_apiid_dist.append(w_apiid_dist)
        label_interval_dist.append(w_interval_dist)

        output = {
            'input_apiid_sequence': torch.tensor(input_apiid_sequence, dtype=torch.long),
            'input_interval_sequence': torch.tensor(input_interval_sequence, dtype=torch.long),
            'input_segment': torch.tensor(input_segment, dtype=torch.long),
            'attention_masks': torch.tensor(attention_masks, dtype=torch.bool),
            'label_apiid_dist': torch.tensor(label_apiid_dist, dtype=torch.float32),
            'label_interval_dist': torch.tensor(label_interval_dist, dtype=torch.float32)
        }
        return output






