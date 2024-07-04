# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python [conda env:diplomka]
#     language: python
#     name: conda-env-diplomka-py
# ---

# %% [markdown]
# ## Imports

# %%
from transformers import AutoTokenizer, AutoModel
import torch
from transformers import AutoTokenizer, AutoModel

# %%
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

# %%
dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(dna)["input_ids"]


# %%
inputs

# %%
tokenizer.decode(inputs)

# %% [markdown]
# > need to load the model from specific commit which solves the issue https://huggingface.co/zhihan1996/DNABERT-2-117M/commit/6617c7e3829423fddd80ba03c7c7dc4f8aab4d19 I've been having otherwise -> revision can be ommited if the PR is accepted

# %%
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, revision='6617c7e')

# %%
hidden_states = model(inputs)[0] # [1, sequence_length, 768]

# %%
from datasets import load_dataset

# %%
dataset = load_dataset('simecek/Human_DNA_v0')

# %%
dataset
