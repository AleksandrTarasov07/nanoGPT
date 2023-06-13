import os
import requests
import tiktoken
import numpy as np
import pandas as pd




# all examples file path
file_path = '/content/gdrive/MyDrive/Colab Notebooks/all_examples.xlsx - Sheet1.csv'


data = pd.read_csv(file_path)
initial_data = data[data['task_id'] == 10].reset_index()

input = []
target = []
for i in range(len(initial_data)):
    input += [initial_data['prompt'][i] + '\n' + initial_data['input'][i]]
    target += [initial_data['output'][i]]


# encode with tiktoken gpt2 bpe and added <|pad|> like a special token
tokenizer_base = tiktoken.get_encoding("gpt2")
tokenizer = tiktoken.Encoding(
    # If you're changing the set of special tokens, make sure to use a different name
    # It should be clear from the name what behaviour to expect.
    name="gpt2_pad",
    pat_str=tokenizer_base._pat_str,
    mergeable_ranks=tokenizer_base._mergeable_ranks,
    special_tokens={
        **tokenizer_base._special_tokens,
        "<|pad|>": 50257
    }
)
input = [tokenizer.encode(input[i]) for i in range(100)]
target = [tokenizer.encode(target[i]) for i in range(100)]

input_lens = [len(input[i]) for i in range(100)]
target_lens = [len(target[i]) for i in range(100)]

max_len_input = np.max(input_lens)
max_len_target = np.max(target_lens)


for i in range(100):
    for j in range(input_lens[i], np.max([max_len_input, max_len_target])):
        input[i] += [50257]
    for j in range(target_lens[i], np.max([max_len_input, max_len_target])):
        target[i] += [50257]

input = np.array(input)
target = np.array(target)

train_input_ids = input[:int(len(input) * 0.9)]
train_target_ids = target[:int(len(target) * 0.9)]

val_input_ids = input[int(len(input) * 0.9):]
val_target_ids = target[int(len(target) * 0.9):]


print(f"train input has {len(train_input_ids) * train_input_ids.shape[1]:,} tokens")
print(f"train target has {len(train_target_ids) * train_target_ids.shape[1]:,} tokens")
print(f"val input has {len(val_input_ids) * val_input_ids.shape[1]:,} tokens")
print(f"val target has {len(val_target_ids) * val_target_ids.shape[1]:,} tokens")

# export to bin files
train_input_ids.tofile(os.path.join(os.path.dirname(__file__), 'train_input.bin'))
train_target_ids.tofile(os.path.join(os.path.dirname(__file__), 'train_target.bin'))
val_input_ids.tofile(os.path.join(os.path.dirname(__file__), 'val_input.bin'))
val_target_ids.tofile(os.path.join(os.path.dirname(__file__), 'val_target.bin'))
