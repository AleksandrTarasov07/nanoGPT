"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""
import json
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchmetrics import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bert import BERTScore
from torch.nn import functional as F
import tiktoken
from datetime import datetime

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 5
log_interval = 1
eval_iters = 5
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'gpt2' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'prompts_gpt2_small_finetune'
wandb_run_name = 'ft-' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
# data
dataset = 'promts'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 1 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 3e-5 # max learning rate
max_iters = 20 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = False # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# device = 'cpu'
dtype = 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

# flag for conditional learning (fine-tuning)
conditional_learning = True

# temperature
temperature = 1.0

#
top_k = 10

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# # -----------------------------------------------------------------------------



# tokenizer

tokenizer = tiktoken.get_encoding("gpt2")


# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    assert gradient_accumulation_steps % torch.cuda.device_count() == 0
    gradient_accumulation_steps //= torch.cuda.device_count()
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('/content/nanoGPT/data', dataset)

if not conditional_learning:
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

else:
    # train_input = np.memmap(os.path.join(data_dir, 'train_input.bin'), dtype=np.uint64, mode='r')
    # train_target = np.memmap(os.path.join(data_dir, 'train_target.bin'), dtype=np.uint64, mode='r')
    # val_input = np.memmap(os.path.join(data_dir, 'val_input.bin'), dtype=np.uint64, mode='r')
    # val_target = np.memmap(os.path.join(data_dir, 'train_target.bin'), dtype=np.uint64, mode='r')

    with open('data/promts/train_data.json') as f:
        train_data = json.load(f)

    with open('data/promts/val_data.json') as f:
        val_data = json.load(f)

# for k, v in train_data.items():
#     print(f"train type key {type(k)}, key  {k}")
#
# for k, v in val_data.items():
#     print(f"val type key {type(k)}, key  {k}")

def get_batch(split, displaying=False):

    if not conditional_learning:
        data = train_data if split == 'train' else val_data
        if not displaying:
            ix = torch.randint(len(data) - block_size, (batch_size,))
        else:
            ix = torch.arange(3, 4)

        x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
        y_seq = tokenizer.decode(y[0].numpy())

    else:
        data = train_data if split == "train" else val_data

        if not displaying:
            ix = torch.randint(len(data) - batch_size - 1, (1,))
        else:
            ix = torch.arange(3, 4)
        # print(split)
        target_len = np.max([len(data[str(i)]['target']) for i in range(ix, ix + batch_size)])
        input_len = np.max([len(data[str(i)]['input']) for i in range(ix, ix + batch_size)])

        for i in range(ix, ix + batch_size):

            if len(data[str(i)]['target']) < target_len:
                for j in range(len(data[i]['target']), target_len):
                    data[str(i)]['target'] += [50256]

            if len(data[str(i)]['input']) < input_len:
                for j in range(len(data[i]['input']), input_len):
                    data[str(i)]['input'] += [50256]

        x = torch.Tensor(np.array([data[str(i)]['input'] for i in range(ix, ix + batch_size)])).int()
        y = torch.Tensor(np.array([data[str(i)]['target'] for i in range(ix, ix + batch_size)])).int()
        y_seq = tokenizer.decode(y[0].numpy())

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y, y_seq


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value

if block_size > model.config.block_size:
    model.augmentation_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss_and_metrics():

    bleu_score = BLEUScore(n_gram=3)
    rouge_score = ROUGEScore()
    out_loss = {}
    out_perp = {}
    out_bleu = {}
    out_rouge1 = {}
    out_rouge2 = {}
    out_rougeL = {}


    model.eval()
    for split in ['train', 'val']:

        losses = torch.zeros(eval_iters)
        perps = torch.zeros(eval_iters)
        bleu = torch.zeros(eval_iters)
        rouge1 = torch.zeros(eval_iters)
        rouge2 = torch.zeros(eval_iters)
        rougeL = torch.zeros(eval_iters)


        for k in range(eval_iters):
            if not conditional_learning:
                X, Y, Y_seq = get_batch(split)
                with ctx:

                    logits, loss = model(X, Y)

                losses[k] = loss.item()
                perps[k] = torch.exp(loss).item()

                X_seq = logits.argmax(dim=-1)[0].cpu().numpy()
                # print(len(X_seq))
                X_seq = tokenizer.decode(X_seq)

            else:
                X, Y, Y_seq = get_batch(split)
                with torch.no_grad():
                    with ctx:
                        X_seq = model.generate(X, X.shape[1])

                X_seq = X_seq[0].cpu().numpy()
                X_seq = tokenizer.decode(X_seq)

            bleu[k] = bleu_score(X_seq, Y_seq)

            rouge_curr = rouge_score(X_seq, Y_seq)
            rouge1[k] = rouge_curr['rouge1_fmeasure']
            rouge2[k] = rouge_curr['rouge2_fmeasure']
            rougeL[k] = rouge_curr['rougeL_fmeasure']

        out_loss[split] = losses.mean()
        out_perp[split] = perps.mean()
        out_bleu[split] = bleu.mean()

        out_rouge1[split] = rouge1.mean()
        out_rouge2[split] = rouge2.mean()
        out_rougeL[split] = rougeL.mean()

    X1, Y1, Y_seq_display = get_batch(split, displaying=True)

    if not conditional_learning:
        with ctx:
            logits, _ = model(X1, Y1)
        X_seq_display = logits.argmax(dim=-1)[0].cpu().numpy()
        X_seq_display = tokenizer.decode(X_seq_display)

    else:
        with torch.no_grad():
            with ctx:
                X_seq1 = model.generate(X1, X1.shape[1])

            X_seq1 = X_seq1[0].cpu().numpy()
            X_seq_display = tokenizer.decode(X_seq1)
    output = X_seq_display
    target = Y_seq_display
    model.train()

    return out_loss, out_perp, out_bleu, out_rouge1, out_rouge2, out_rougeL, output, target

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
# print(model.lm_head)
print("*********************GL HF !!!*********************")
X, Y, Y_seq = get_batch('train') # fetch the very first batch
logits, _ = model(X)
# print(f'logits {logits}\nloss {_}')
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
output = []
target = []
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num != 0 and iter_num % eval_interval == 0 and master_process:
        losses, perps, bleus, rouges1, rouges2, rougesL, output_curr, target_curr = estimate_loss_and_metrics()
        if not conditional_learning:
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} \
              \ntrain perplexity {perps['train']:.4f}, val perplexity {perps['val']:.4f}")
        else:
            print(f"step {iter_num}: \ntrain bleu {bleus['train']:.4f}, val bleu {bleus['val']:.4f}, \
            train rouge1 f_mesure {rouges1['train']}")
        output += [output_curr]
        target += [target_curr]
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "train/perplexity": perps['train'],
                "val/perplexity": perps['val'],
                "train/bleu": bleus['train'],
                "val/bleu": bleus['val'],
                "train/rouge1_fmesure": rouges1['train'],
                "val/rouge1_fmesure": rouges1['val'],
                "train/rouge2_fmesure": rouges2['train'],
                "val/rouge2_fmesure": rouges2['val'],
                "train/rougeL_fmesure": rougesL['train'],
                "val/rougeL_fmesure": rougesL['val'],
                "va"
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage

            })
            '''
            TODO: 
            "train/bert_f1": bert_f1['train'],
                "train/bert_precision": bert_precision['train'],
                "train/bert_recall": bert_recall['train'],
                "val/bert_f1": bert_f1['val'],
                "val/bert_precision": bert_precision['val'],
                "val/bert_recall": bert_recall['val'],
            '''
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        if not conditional_learning:
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

        else:
            with ctx:
                X_gen, logits = model.generate(X, X.shape[1], loss=True)
                length = Y.shape[1]
                # print(f"x gen shape  {X_gen[:, -length:].shape}, y shape {Y.shape}, \n{X_gen}\n{Y}")

                loss = torch.nn.functional.cross_entropy(logits, Y.int()) / gradient_accumulation_steps

        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, Y_seq = get_batch('train')
        print(f"input: {tokenizer.decode(X[0].cpu().numpy())},\ntarget: {Y_seq}")
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
pd.DataFrame(np.array([target, output]).reshape(2, -1), index=['target', 'output']).to_excel(dataset + '_' + init_from + '.xlsx')

if ddp:
    destroy_process_group()


