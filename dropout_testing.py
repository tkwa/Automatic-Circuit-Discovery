#!/usr/bin/env python
# coding: utf-8

# %%


try:
    import google.colab

    IN_COLAB = True
    print("Running as a Colab notebook")

    import subprocess # to install graphviz dependencies
    command = ['apt-get', 'install', 'graphviz-dev']
    subprocess.run(command, check=True)

    import os # make images folder
    os.mkdir("ims/")

    from IPython import get_ipython
    ipython = get_ipython()

    ipython.run_line_magic( # install ACDC
        "pip",
        "install git+https://github.com/ArthurConmy/Automatic-Circuit-Discovery.git@2cc2d6d71416bddd3a88f287ffccfc0863ac8ddc",
    )

except Exception as e:
    IN_COLAB = False
    print("Running as a outside of colab")

    import numpy # crucial to not get cursed error
    import plotly

    plotly.io.renderers.default = "colab"  # added by Arthur so running as a .py notebook with #%% generates .ipynb notebooks that display in colab
    # disable this option when developing rather than generating notebook outputs

    import os # make images folder
    if not os.path.exists("ims/"):
        os.mkdir("ims/")

    from IPython import get_ipython

    ipython = get_ipython()
    if ipython is not None:
        print("Running as a notebook")
        ipython.run_line_magic("load_ext", "autoreload")  # type: ignore
        ipython.run_line_magic("autoreload", "2")  # type: ignore
    else:
        print("Running as a script")


# %%:


import wandb
import IPython
from IPython.display import Image, display
import torch
import gc
from tqdm import tqdm
import networkx as nx
import os
import torch
from torch import Tensor
import huggingface_hub
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from tqdm import tqdm
import yaml
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from jaxtyping import Float, Int, Bool

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import transformer_lens
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformer import (
    HookedTransformer,
)
DEVICE="cuda"


# %%:


num_examples = 100
# things = get_all_ioi_things(
#     num_examples=num_examples, device=DEVICE, metric_name=args.metric
# )

from acdc.ioi.utils import get_gpt2_small
tl_model = get_gpt2_small(device=DEVICE)


# ### Testing how to do dropout...

# %%


import transformers
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model_gpt2_small = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE)


# %%


model_gpt2_small.config.attn_pdrop


# %%


# definitely dropout here in normal gpt2 small
model_gpt2_small.transformer.h[0]


# %%


type(tl_model.blocks[0])


# %%


transformer_lens.components.TransformerBlock


# In[ ]:





# # Implementing dropout using hooks
# 
# Plan from Callum:
# - Use permanent hooks
# - Write a hook for each kind of dropout: attention, resid, MLP
# - Dropout hooks have a higher level
# - Add a command line argument for dropout
# - Change self.update_cur_metric
# 
# TODO now:
# - Figure out the attention, resid, and MLP dropout hooks

# %%


tl_model = HookedTransformer.from_pretrained('gpt2')


# %%


def dropout_hook(input: Float[Tensor, "batch ..."], hook: HookPoint,
                   p: float = 0.1) -> Float[Tensor, "batch ..."]:
    """
    Same function used in torch implementation of t.nn.Dropout
    """
    return torch.nn.functional.dropout(input, p)

def dropout_name_fn(name: str) -> bool:
    ret = "pattern" in name or "mlp_out" in name or "attn_out" in name or name == "blocks.0.hook_resid_pre"
    # print(ret, name)
    return ret


# %%


# def tl_logit_diff(seed=1234):
torch.manual_seed(1234)
input_tl = "hello my name is"
tl_model.reset_hooks()
logits_tl_eval = tl_model(input_tl)
tl_model.add_hook(dropout_name_fn, dropout_hook)
logits_tl_train = tl_model(input_tl)
# logits_tl_train.shape
logit_diff_tl = logits_tl_train - logits_tl_eval
(logit_diff_tl**2).mean()


# %%
# There should be a difference between train and eval logits for HF model

# def hf_logit_diff(seed=1234):
torch.manual_seed(1234)
tokenizer.add_bos_token = True
input = tokenizer.encode("hello my name is", return_tensors="pt", add_special_tokens=True).to(DEVICE)
model_gpt2_small.train()
logits_train = model_gpt2_small(input).logits.cuda()
logits_train = logits_train - logits_train.mean(dim=2).unsqueeze(-1)

model_gpt2_small.eval()
logits_eval = model_gpt2_small(input).logits.cuda()
logits_eval = logits_eval - logits_eval.mean(dim=2).unsqueeze(-1)
logit_diff = logits_train - logits_eval
# print(logit_diff.shape)
(logit_diff**2).mean()


# %%


# Confirm that logits are very similar without dropout
# normalize mean of HF model
tl_vs_hf = logits_tl_eval - logits_eval

(tl_vs_hf**2).mean()


# %%


# Are logits same with dropout? It could just be random seed...
# normalize mean of HF model
tl_vs_hf_train = logits_tl_train - logits_train

(tl_vs_hf_train**2).mean()


# %%

from collections import OrderedDict
# Apply a hook on model_gpt2_small to save layer 0 resid_pre to a global variable
# Apply a hook on tl_model to save layer 0 resid_pre to a global variable
def hf_l0_hook(self, input, output):
    global hf_l0
    (result,) = input
    # result = output
    hf_l0 = result.to(DEVICE)

def tl_l0_hook(input, hook):
    global tl_l0
    tl_l0 = input

hf_hook_point = model_gpt2_small.transformer.h[0]
hf_hook_handle = hf_hook_point.register_forward_hook(hf_l0_hook)

model_gpt2_small.train()
torch.manual_seed(12345)
model_gpt2_small(input)
hf_hook_point._forward_hooks = OrderedDict() # remove hook
hf_hook_handle.remove()

tl_model.reset_hooks()
tl_model.add_hook(dropout_name_fn, dropout_hook) # Add this hook before the observation hook
tl_model.add_hook(name="blocks.0.hook_resid_pre", hook=tl_l0_hook) # blocks.0.hook_resid_pre
torch.manual_seed(12345)
tl_model(input)

tl_l0.shape, hf_l0.shape

print(f"{tl_l0=},\n {hf_l0=}, \n {tl_l0 - hf_l0=}")
print(f"Zero elements in tl_l0: {(tl_l0 == 0).sum()}")
print(f"Zero elements in hf_l0: {(hf_l0 == 0).sum()}")
print(f"Mean square difference between HF and TL layer 0 resid_pre: {(tl_l0 - hf_l0).pow(2).mean()}")

# %%

# Do two dropout modules even have the same output given same input and same seed? Yes. So why do we have a discrepancy?
dropout1 = torch.nn.Dropout(p=0.1).to(DEVICE).train()
dropout2 = torch.nn.Dropout(p=0.1).to("cpu").train()
torch.backends.cudnn.deterministic = True  
input1 = torch.randn(1, 10)
torch.manual_seed(12345)
out1 = dropout1(input1)
torch.manual_seed(12345)
out2 = dropout1(input1)

print(out1 - out2)
print((out1 == out2).all())
# %%
# Now test the actual first modules...
input_ids = input
token_embed = model_gpt2_small.transformer.wte(input)
input_shape = input_ids.size()
input_ids = input_ids.view(-1, input_shape[-1])
batch_size = input_ids.shape[0]
past_length = 0
past_key_values = tuple([None] * len(model_gpt2_small.transformer.h))
position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device='cpu')
position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
position_embed = model_gpt2_small.transformer.wpe(position_ids)
hidden_states = (token_embed + position_embed).to(DEVICE)

print(hidden_states.shape)
hidden_states - hf_l0

# %%


generated_text_samples = model_gpt2_small.generate(input)

for i, beam in enumerate(generated_text_samples):
    print(f"{i}: {tokenizer.decode(beam, skip_special_tokens=True)}")
    print()


# %%


tl_model


# %%


model_gpt2_small.transformer


# %%


type(model_gpt2_small.transformer.h[0])


# %%


transformers.models.gpt2.modeling_gpt2.GPT2Block


# %%


model_gpt2_small.transformer.h[0].train()


# %%


tl_model.eval()


# In[ ]:




transformer_lens.components.TransformerBlock
transformers.GPT2Model
torch.nn.Dropout