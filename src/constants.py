
class Constants:
    # 序列分割
    segment_interval = 1800
    # interval离散化分桶函数
    interval_bucket_bins = [i for i in range(100, 600, 10)] + [i for i in range(600, 1800, 50)]
    interval_bucket_num = len(interval_bucket_bins) + 1