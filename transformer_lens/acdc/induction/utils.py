import wandb
import os
from collections import defaultdict
import pickle
import torch
import huggingface_hub
import datetime
from typing import Dict
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import warnings
import networkx as nx
from transformer_lens.acdc.utils import (
    make_nd_dict,
    TorchIndex,
    Edge, 
    EdgeType,
)  # these introduce several important classes !!!

DEVICE = "cuda:0"
SEQ_LEN = 300
NUM_EXAMPLES = 40
MODEL_ID = "attention_only_2"
PRINT_CIRCUITS = True
ACTUALLY_RUN = True
SLOW_EXPERIMENTS = True
EVAL_DEVICE = "cuda:0"
MAX_MEMORY = 20000000000
# BATCH_SIZE = 2000
USING_WANDB = True
MONOTONE_METRIC = "maximize"
START_TIME = datetime.datetime.now().strftime("%a-%d%b_%H%M%S")
PROJECT_NAME = f"induction_arthur"

# get the dataset from HF
validation_fname = huggingface_hub.hf_hub_download(
    repo_id="ArthurConmy/redwood_attn_2l", filename="validation_data.pt"
)
validation_data = torch.load(validation_fname)

good_induction_candidates_fname = huggingface_hub.hf_hub_download(
    repo_id="ArthurConmy/redwood_attn_2l", filename="good_induction_candidates.pt"
)
good_induction_candidates = torch.load(good_induction_candidates_fname)

mask_repeat_candidates_fname = huggingface_hub.hf_hub_download(
    repo_id="ArthurConmy/redwood_attn_2l", filename="mask_repeat_candidates.pkl"
)
mask_repeat_candidates = torch.load(mask_repeat_candidates_fname)
mask_repeat_candidates.requires_grad = False
mask_repeat_candidates = mask_repeat_candidates[:NUM_EXAMPLES, :SEQ_LEN]


def shuffle_tensor(tens):
    """Shuffle tensor along first dimension"""
    torch.random.manual_seed(42)
    return tens[torch.randperm(tens.shape[0])]


toks_int_values = validation_data[:NUM_EXAMPLES, :SEQ_LEN].to(DEVICE).long()
toks_int_values_other = (
    shuffle_tensor(validation_data[:NUM_EXAMPLES, :SEQ_LEN]).to(DEVICE).long()
)
good_induction_candidates = mask_repeat_candidates[:NUM_EXAMPLES, :SEQ_LEN].to(DEVICE)
labels = validation_data[:NUM_EXAMPLES, 1 : SEQ_LEN + 1].to(DEVICE).long()


def kl_divergence(
    logits: torch.Tensor,
    base_model_probs: torch.Tensor,
    using_wandb,
):
    """Compute KL divergence between base_model_probs and probs"""
    probs = F.softmax(logits, dim=-1)

    assert probs.min() >= 0.0
    assert probs.max() <= 1.0

    kl_div = (base_model_probs.cpu() * (base_model_probs.cpu().log() - probs.cpu().log())).sum(dim=-1)

    assert kl_div.shape == mask_repeat_candidates.shape, (
        kl_div.shape,
        mask_repeat_candidates.shape,
    )
    kl_div = kl_div * mask_repeat_candidates.cpu().long()

    answer = (kl_div.sum() / mask_repeat_candidates.cpu().int().sum().item()).item()

    if using_wandb:
        wandb.log({"metric": answer})

    return answer