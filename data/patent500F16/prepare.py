import os
import requests
import tiktoken
import numpy as np
import pandas as pd



DESCRIPTION_ONLY = True

# 500F16 csv file path
path_csv_file = '/content/gdrive/MyDrive/Colab Notebooks/500_F16.csv'

# read csv file and drop 2 first lines
patents_table = pd.read_csv(path_csv_file, names=['id', 'title', 'abstracts', 'description_text'])
patents_table.drop([0, 1], axis=0, inplace=True)

# get all columns with all non Nan values
ids = patents_table['id'].to_numpy()
titles = patents_table['title'].to_numpy()
abstracts = patents_table['abstracts']
descriptions = patents_table['description_text']
descriptions = descriptions[descriptions.notna()].to_numpy()
abstracts = abstracts.to_numpy()

# flag to use just descriptions
if DESCRIPTION_ONLY:
    n = len(descriptions)
    train_data = descriptions[:int(n * 0.9)]
    val_data = descriptions[int(n * 0.9):]


    # encode with tiktoken gpt2 bpe
    tokenizer = tiktoken.get_encoding("gpt2")
    train_ids = tokenizer.encode_ordinary(train_data)
    val_ids = tokenizer.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.txt'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))






