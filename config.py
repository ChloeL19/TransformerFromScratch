import einops
from fancy_einsum import einsum
from dataclasses import dataclass
from transformer_lens import HookedTransformer
import torch as t
import torch.nn as nn
import numpy as np
import math
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
import tqdm.auto as tqdm

@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12