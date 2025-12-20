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
import inspect

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
vocab_size = 5 # the desired final vocabulary size
num_merges = vocab_size - 4 # since it is only A,T,G,C
ids = list(tokens) # copy so we don't destroy the original list
bpe = BPEtokenizer(ids)

# merges = {}
# for i in range(num_merges):
#     print(i)
#     idx = 4 + i
#     bpe.ids = bpe.merge(idx)

# print(f"Tokenization done: {bpe.ids[0:20]}")
# # pd.DataFrame(bpe.ids).to_csv(os.path.join(savedir,"data/genome_sequences/hg38/ids_encoded.csv"), index = False, header = False)
# print(f"compression ratio: {len(tokens) / len(ids):.2f}X")
# 3721774 4093
# compression ratio: 5.37X


# ### Loading the tokens
ids = pd.read_csv(os.path.join(savedir, "data/genome_sequences/hg38/ids_encoded_2.csv"), header = None)
ids = ids[0].tolist()

#region GPT2 like DNA model
ids_t = torch.tensor(ids)
# I have tried a configuration of 2**n makes it more efficients
@dataclass
class DNAGPTconfig:
    block_size: int = 1024 ## it is the token size
    n_layer: int = 16
    embd_size: int = 512
    n_head: int = 16
    vocab_size: int = (ids_t.max() + 1)
    batch_size: int = 16
    master_process: bool = True
    dropout: float = 0.1

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.embd_size, 3 * config.embd_size) # 3 dimension as it is divided into q,k,v
        self.c_proj = nn.Linear(config.embd_size, config.embd_size)
        self.n_head = config.n_head
        self.embd_size = config.embd_size
        # self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B,T,C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embd_size, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        # wei = q @ k.transpose(-2,-1)
        # wei = wei * C**-0.5
        # wei = F.softmax(wei, dim = -1) 
        # wei = self.dropout(wei)
        # wei = wei @ v
        # Instead of running all of them, we can use flash attention at once
        wei = F.scaled_dot_product_attention(q,k,v, is_causal = True) # flash attention
        # combine all of them
        wei = wei.transpose(1,2).contiguous().view(B,T,C)
        # wei = self.dropout(wei)
        wei = self.c_proj(wei)
        return wei

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.Linear(config.embd_size, 4 * config.embd_size)
        self.nln = nn.GELU(approximate = "tanh")
        self.ln2 = nn.Linear(4 * config.embd_size, config.embd_size)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.ln1(x)
        x = self.nln(x)
        x = self.dropout(self.ln2(x))
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_f1 = nn.LayerNorm(config.embd_size)
        self.self_attn = CausalSelfAttention(config)
        self.ln_f2 = nn.LayerNorm(config.embd_size)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.self_attn(self.ln_f1(x))
        x = x + self.mlp(self.ln_f2(x))
        return x

class DNAGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformers = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embd_size),
            wpe = nn.Embedding(config.block_size, config.embd_size),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_norm = nn.LayerNorm(config.embd_size)
            )
        )
        self.lm_head = nn.Linear(config.embd_size, config.vocab_size, bias = False)
        
        # weight sharing scheme
        self.transformers.wte.weight = self.lm_head.weight

    
    def forward(self, idx, targets = None):
        B,T = idx.shape
        tok = self.transformers.wte(idx)
        pos = self.transformers.wpe(torch.arange(T, dtype = torch.long, device = idx.device))
        x = tok + pos ## require both the position and token
        for block in self.transformers.h:
            x=block(x) ## Since it will go through all the layers of the transformers
        x=self.transformers.ln_norm(x)
        logits=self.lm_head(x)
        loss = None

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, label_smoothing=0.05) # neg log likelihood

        return logits,loss
      
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
          {'params': decay_params, 'weight_decay': weight_decay},
          {'params': nodecay_params, 'weight_decay': 0.0}
          ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if self.config.master_process:
          print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
          print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
          
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if self.config.master_process:
          print(f"using fused AdamW: {use_fused}")
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        
        return optimizer

## Now for training you surely need to have GPU or cuda
# device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
    device = "mps" ### This is for MACs

print("device name",device)

## This will pass all the weights and bias from gpt2 to our model
model = DNAGPT(DNAGPTconfig())
print(model.to(device))

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        self.epoch = 1
        # state
        self.current_position = 0

    def nextbatch(self, ids):
        B = self.B
        T = self.T
        ids = torch.tensor(ids)
        ix = torch.randint(0, len(ids) - B*T - 1, (1,))
        buf = ids[ix : ix + B*T + 1]
        # buf = ids[self.current_position : self.current_position+B*T+1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)
        # self.current_position += B * T + 1
        # print(self.current_position)
        # if loading the last batch is greater than the lenght
        # if (self.current_position + (B * T + 1) > len(ids)):
        #   self.epoch += 1
        #   print(f'epoch: {self.epoch}')
        #   self.current_position = 0
        x=x.to(device) ## putting it on GPU
        y=y.to(device)
        return x,y

# Create a torch optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)


# Data Loader
train_loader = DataLoaderLite(DNAGPTconfig.batch_size,DNAGPTconfig.block_size)

split = int(0.9 * len(ids))
train_ids = ids[:split]
val_ids   = ids[split:]

model.train()
for steps in range(1000):
    x,y = train_loader.nextbatch(train_ids)
    logits,loss = model(x,y)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    if steps % 100 == 0:
      print(f'Training loss at every 100 steps: {loss.item()}')
print(f'Final Training loss: {loss.item()}')

# max_steps = 1000
# accum_steps = 4
# optimizer.zero_grad(set_to_none=True)

# scaler = torch.cuda.amp.GradScaler()
# optimizer.zero_grad(set_to_none=True)

# for step in range(max_steps):
#     x, y = train_loader.nextbatch(train_ids)

#     with torch.cuda.amp.autocast():
#         logits, loss = model(x, y)
#         loss = loss / accum_steps

#     scaler.scale(loss).backward()

#     if (step + 1) % accum_steps == 0:
#         scaler.unscale_(optimizer)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         scaler.step(optimizer)
#         scaler.update()
#         scheduler.step()
#         optimizer.zero_grad(set_to_none=True)

#     if step % 100 == 0:
#         print(f"step {step} | loss {loss.item() * accum_steps:.4f}")


val_loader = DataLoaderLite(DNAGPTconfig.batch_size,DNAGPTconfig.block_size)

@torch.no_grad() ## So it should not update any weights
def val_step():
    model.eval()
    losses=[]
    for _ in range(10):
        x, y = val_loader.nextbatch(val_ids)
        logistics, lossing = model(x, targets = None)
        B, T, C = logistics.shape
        # print(B,T,C)
        logits = logistics.view(B*T,C)
        targets = y.view(B*T)
        loss = F.cross_entropy(logits, targets) # neg log likelihood
        losses.append(loss.item())
        print(f'Validation loss: {loss.item()}')
    return sum(losses) / len(losses)

print(f'Final Validation loss: {val_step()}')