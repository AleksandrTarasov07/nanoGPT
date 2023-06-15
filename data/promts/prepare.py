import os
import tiktoken
import numpy as np
import pandas as pd
import json

# all examples file path
file_path = 'all_examples.csv'

# download data
data = pd.read_csv(file_path)

# take just 10 prompt
initial_data = data[data['task_id'] == 10].reset_index()

# init tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# group data
input = []
target = []
for i in range(len(initial_data)):
    input += [initial_data['prompt'][i] + '\n' + initial_data['input'][i]]
    target += [initial_data['output'][i]]

input = [tokenizer.encode(input[i]) for i in range(len(input))]
target = [tokenizer.encode(target[i]) for i in range(len(input))]

for i in range(len(input)):
    input[i] += [50256]
    target[i] += [50256]

input_lens = [len(input[i]) for i in range(len(input))]
target_lens = [len(target[i]) for i in range(len(input))]

# sort data by the size of target
df = pd.DataFrame(np.array(target_lens), columns=['1'])

sorted_indexes = np.array(df.index)

# create dict with all data in order min --> max
train_data = {}
val_data = {}
for i in range(len(sorted_indexes)):
    if i < int(0.1 * len(input)):
        val_data.update({str(i): {'input': input[sorted_indexes[i]],
                               'target': target[sorted_indexes[i]]}})
    else:
        train_data.update({i: {'input': input[sorted_indexes[i]],
                     'target': target[sorted_indexes[i]]}})




print(f"input has {np.sum(input_lens):,} tokens")
print(f"targer has {np.sum(target_lens):,} tokens")


# export data to json files

with open("train_data.json", "w") as f:
    json.dump(train_data, f)

with open("val_data.json", "w") as f:
    json.dump(val_data, f)
