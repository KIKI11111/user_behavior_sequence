import numpy as np
import pandas as pd
import pickle
from datasets import Dataset
from src.preprocess import transforms

if __name__ == '__main__':
    data = pd.read_pickle('../data/data_demo.pkl')
    print(data.head())
    data.rename(columns={'uid': 'entity_value', 'pid': 'apiid'}, inplace=True)
    data.to_csv('../data/data_demo.csv')
    train_df = data.iloc[:1000, :]
    trainset = Dataset.from_pandas(train_df)

    val_df = data.iloc[1000:, :]
    valset = Dataset.from_pandas(val_df)

    print(trainset)

    trainset = trainset.map(lambda batch: transforms.sequence_preprocess_batched_map_func(batch, 512), batched=True, batch_size=50)
    valset = valset.map(lambda batch: transforms.sequence_preprocess_batched_map_func(batch, 512), batched=True, batch_size=50)
    print(trainset)
    print(trainset[0]['apiid'])
    print(trainset[0]['interval'])
    print(trainset[0]['segment'])
