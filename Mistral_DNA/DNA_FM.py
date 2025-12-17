 # conda activate torch_gpu
import os
import pickle
import pandas as pd
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
from datasets import load_dataset

# First load the dataset
# Got the dataset from https://github.com/raphaelmourad/LLM-for-genomics-training
savedir = "/mnt/data/projects/.immune/Personal/DNA-Language-Model/Mistral_DNA/"
os.chdir(savedir)
DNA_text = pd.read_csv(os.path.join(savedir,"data/genome_sequences/hg38/sequences_hg38_200b_verysmall.csv.gz"))
DNA_joined = "".join(DNA_text['text'].tolist())
# tokens = DNA_joined.encode("utf-8") # raw bytes
DNA_int = {'A':0, 'C':1, 'G':2, 'T':3}
tokens = [DNA_int[ch] for ch in DNA_joined] # convert to a list of integers in range for convenience

#region Tokenization
### Combining the two character into one
class BPEtokenizer:
  def __init__(self, ids):
    self.ids = ids

  def get_stats(self):
      counts = {}
      for pair in zip(self.ids, self.ids[1:]): # Pythonic way to iterate consecutive elements
          counts[pair] = counts.get(pair, 0) + 1
      return counts

  def merge(self, idx):
    stats = self.get_stats() # calling the stats of the tokens
    pair = max(stats, key=stats.get) # extracting the maximum tokens
    new_text = []
    i = 0
    while i < len(self.ids):
      if i < len(self.ids) -1 and self.ids[i] == pair[0] and self.ids[i+1] == pair[1]:
        new_text.append(idx)
        i+=2
      else:
        new_text.append(self.ids[i])
        i+=1
    return new_text

### Since we are running again for model training we do not need to train the Tokenization
vocab_size = 10 # the desired final vocabulary size
num_merges = vocab_size - 4 # since it is only A,T,G,C
ids = list(tokens) # copy so we don't destroy the original list
bpe = BPEtokenizer(ids)

merges = {}
for i in range(num_merges):
    print(i)
    idx = 4 + i
    bpe.ids = bpe.merge(idx)

print(f"Tokenization done: {bpe.ids[0:20]}")
print(f"Tokenization saved: {os.path.join(savedir,"data/genome_sequences/hg38/token_encoded.csv")}")
pd.DataFrame(bpe.ids).to_csv(os.path.join(savedir,"data/genome_sequences/hg38/ids_encoded.csv"), index = False, header = False)
print(f"compression ratio: {len(tokens) / len(ids):.2f}X")
# 3721774 4093
# compression ratio: 5.37X


# ### Loading the tokens
ids = pd.read_csv(os.join.path(savedir, "data/genome_sequences/hg38/ids_encoded.csv"), header = False)
ids = ids[0].tolist()
