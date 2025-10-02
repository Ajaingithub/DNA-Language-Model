# Following this notebook https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/nucleotide_transformer_dna_sequence_modelling.ipynb#scrollTo=8XJthJun4mVw 
# conda create -n nt_finetune_v2 python=3.10
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install transformers datasets huggingface_hub accelerate
# pip install biopython
# pip install bitsandbytes
# pip install xformers
# pip install --upgrade torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# pip install numpy==1.26.4 scipy==1.13.1 Keep the numpy and scipy with this version
# Imports
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
# 1. AutoTokenizer
#     What it does: This class is a factory that automatically loads the correct tokenizer associated with a specific pre-trained model checkpoint (e.g., an NT model on the Hugging Face Hub).
#     Why you use it: You don't need to know the exact tokenizer class (like BertTokenizer or DNATokenizer). You just provide the model's name or path, and AutoTokenizer ensures you get the exact tokenization logic (like the 6-mer split, k-mer vocabulary, and special tokens) that the original model was trained with. This is crucial for data consistency.
# 2. AutoModelForSequenceClassification
#     What it does: Similar to AutoTokenizer, this is a factory that automatically loads the weights of a pre-trained Transformer model (like NT) and adds a sequence classification head on top.
#     Why you use it: This is the specific architecture for fine-tuning. It loads the massive, frozen Transformer layers and attaches a small, learnable classification layer (or "head") with the appropriate number of output classes for your specific task (e.g., 2 classes for benign/pathogenic, or 5 classes for tissue types).
# 3. TrainingArguments
#     What it does: This class is used to define all the hyperparameters and configuration settings for your fine-tuning run.
#     Why you use it: You instantiate this class with arguments like:
#         output_dir: Where to save the checkpoints.
#         learning_rate: The specific rate for optimization.
#         per_device_train_batch_size: The batch size you asked about (e.g., 512).
#         num_train_epochs: How many times to loop over the data.
#         logging_steps: How often to print training progress.
#         ...and many other settings (like weight decay, evaluation strategy, etc.).
# 4. Trainer
#     What it does: The Trainer class is a high-level training API that orchestrates the entire fine-tuning process. It ties together the model, data, and hyperparameters.
#     Why you use it: You initialize the Trainer by passing it four key ingredients:
#         The model (from AutoModelForSequenceClassification).
#         The training arguments (from TrainingArguments).
#         Your training and evaluation datasets.
#         A custom function to compute metrics (like accuracy or F1-score).
#     Execution: Once initialized, you simply call trainer.train(), and it handles the entire loop: moving data to the GPU, calculating loss, backpropagation, gradient updates, and saving checkpoints.
import torch
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from accelerate.test_utils.testing import get_backend
device, _, _ = get_backend()

# The nucleotide transformer will be fine-tuned on two classification tasks: promoter and enhancer 
# types classification. The AutoModelForSequenceClassification module automatically loads the model
# and adds a simple classification head on top of the final embeddings.

# Promoter prediction is a **sequence classification** problem, in which the DNA sequence is predicted to be either a promoter or not.
# A promoter is a region of DNA where transcription of a gene is initiated. Promoters are a vital component of expression 
# vectors because they control the binding of RNA polymerase to DNA. RNA polymerase transcribes DNA to mRNA which is ultimately 
# translated into a functional protein
# This task was introduced in [DeePromoter](https://www.frontiersin.org/articles/10.3389/fgene.2019.00286/full), where a set of 
# TATA and non-TATA promoters was gathered. A negative sequence was generated from each promoter, by randomly sampling subsets of 
# the sequence, to guarantee that some obvious motifs were present both in the positive and negative dataset.

num_labels_promoter = 2
# Load the model
model = AutoModelForSequenceClassification.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref", num_labels=num_labels_promoter)
model = model.to(device)


from datasets import load_dataset, Dataset

# Load the promoter dataset from the InstaDeep Hugging Face ressources
dataset_name = "default"
train_dataset_promoter = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks",
        dataset_name,
        split="train",
        streaming= False,
    )
test_dataset_promoter = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks",
        dataset_name,
        split="test",
        streaming= False,
    )