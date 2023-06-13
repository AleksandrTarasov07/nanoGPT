import time
from datetime import datetime

out_dir = 'out-prompts'
eval_interval = 5
eval_iters = 20
wandb_log = False # feel free to turn on
wandb_project = 'prompts_gpt2_small_finetune'
# wandb_run_name = 'ft-' + str(time.time())
wandb_run_name = 'ft-' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

dataset = 'promts'
init_from = 'gpt2' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 20

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False

conditional_learning = True