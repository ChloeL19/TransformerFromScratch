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
from config import Config

cfg = Config()
print(cfg)

class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual):
        # residual: [batch, position, d_model]
        # output: [batch, position, d_model]
        residual_mean = residual.mean(dim=-1, keepdim=True)
        residual_std = (residual.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps).sqrt()

        residual = (residual - residual_mean) / residual_std
        return residual * self.w + self.b
                    
class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        # output: [batch, position, d_model]
        return self.W_E[tokens]
    
class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        # output: [batch, position, d_model]
        batchsize, seqlength = tokens.shape
        return einops.repeat(self.W_pos[:seqlength], "seq model_d -> batch seq model_d", batch=batchsize)

class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))

        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))

        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device="cuda"))

    def forward(self, normalized_resid_pre: t.Tensor):
        # normalized_resid_pre: [batch, position, d_model]
        # output: [batch, position, d_model]

        # calculate the Q,K,V embeddings
        Q = einsum("batch position d_model, n_heads d_model d_head -> batch position n_heads d_head",
                   normalized_resid_pre, self.W_Q) + self.b_Q
        K = einsum("batch position d_model, n_heads d_model d_head -> batch position n_heads d_head",
                   normalized_resid_pre, self.W_K) + self.b_K
        V = einsum("batch position d_model, n_heads d_model d_head -> batch position n_heads d_head",
                   normalized_resid_pre, self.W_V) + self.b_V

        # compute the attention scores, which is the QK^T matrix
        attn_scores = einsum("batch positionq n_heads d_head, batch positionk n_heads d_head -> batch positionq positionk", 
                             Q, K)
        # scale and apply causal mask
        scaled_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head ** 0.5)

        # compute the attention probabilities
        attn_probs = t.softmax(scaled_masked, dim=-1)

        # take the weighted average of the value vectors
        z = einsum("batch pq pk, batch pq n_heads d_head -> batch pq n_heads d_head", attn_probs, V)

        # linearly map z to the proper output dimension
        out = einsum("batch pq n_heads d_head, n_heads d_head d_model -> batch pq d_model", z, self.W_O) + self.b_O
        

        if self.cfg.debug == True:
          print("number of heads: {}".format(self.cfg.n_heads))
          print("model_d : {}".format(self.cfg.d_model))
          print("Shape of Q embeddings: {}".format(Q.shape))
          print("Shape of K embeddings: {}".format(K.shape))
          print("Shape of V embeddings: {}".format(V.shape))
          print("Attn scores shape: {}".format(attn_scores.shape))
          print("Attn probabilities: {}".format(attn_probs.shape))
          print("z shape: {}".format(z.shape))

        return out

    def apply_causal_mask(self, attn_scores: t.Tensor):
        # attn_scores: [batch, n_heads, query_pos, key_pos]
        # output: [batch, n_heads, query_pos, key_pos]
        mask = t.triu(t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores
    
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        nn.init.normal_(self.W_out, std=self.cfg.init_range)
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))

    def forward(self, normalized_resid_mid):
        # normalized_resid_mid: [batch, position, d_model]
        # output: [batch, position, d_model]
        h1 = einsum("batch pos d_model, d_model d_mlp -> batch pos d_mlp",
                    normalized_resid_mid, self.W_in) + self.b_in
        h2 = gelu_new(h1)
        out = einsum("batch pos d_mlp, d_mlp d_model -> batch pos d_model", h2, self.W_out) + self.b_out
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid_pre):
        # resid_pre [batch, position, d_model]
        resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post
    
class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))

    def forward(self, normalized_resid_final):
        # normalized_resid_final: [batch, position, d_model]
        # output: [batch, position, d_vocab]
        pass
    
class DemoTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens):
        # tokens [batch, position]
        residual = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        logits = self.unembed(self.ln_final(residual))
        return logits
    
    def inference_w_gpt2_weights(self):
        #TODO: implement this
        pass