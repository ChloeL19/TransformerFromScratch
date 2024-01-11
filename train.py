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

import datasets
import transformers
import plotly.express as px

from model import *
from config import Config

batch_size = 8
num_epochs = 1
max_steps = 1000
log_every = 10
lr = 1e-3
weight_decay = 1e-2
model_cfg = Config(debug=False, d_model=256, n_heads=4, d_head=64, d_mlp=1024, n_layers=2, n_ctx=256, d_vocab=reference_gpt2.cfg.d_vocab)

dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train")
print(dataset)
print(dataset[0]['text'][:100])
tokens_dataset = tokenize_and_concatenate(dataset, reference_gpt2.tokenizer, streaming=False, max_length=model_cfg.n_ctx, column_name="text", add_bos_token=True, num_proc=4)
data_loader = t.utils.data.DataLoader(tokens_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

model = DemoTransformer(model_cfg)
model.cuda()

optimizer = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

losses = []
print("Number of batches:", len(data_loader))

for epoch in range(num_epochs):
    
    for c, batch in tqdm.tqdm(enumerate(data_loader)):
        
        # Get batch of tokens, and run your model on them
        tokens = batch['tokens'].cuda()
        logits = model(tokens)
        # Get the avg cross entropy loss of your predictions
        loss = -get_log_probs(logits, tokens).mean()
        # Backprop on the loss (so our parameters store gradients)
        loss.backward()
        # Update the values of your parameters, using the Adam method
        optimizer.step()
        # Reset gradients to zero (since grads accumulate with each .backward() call)
        optimizer.zero_grad()

        losses.append(loss.item())
        if c % log_every == 0:
            print(f"Step: {c}, Loss: {loss.item():.4f}")
        if c > max_steps:
            break

px.line(y=losses, x=np.arange(len(losses))*(model_cfg.n_ctx * batch_size), labels={"y":"Loss", "x":"Tokens"}, title="Training curve for my tiny demo model!")
