# Following this notebook https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/nucleotide_transformer_dna_sequence_modelling.ipynb#scrollTo=8XJthJun4mVw 
# conda create -n nt_finetune_v2 python=3.10
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install transformers datasets huggingface_hub accelerate
# pip install biopython
# pip install bitsandbytes
# pip install xformers
# pip install --upgrade torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# pip install numpy==1.26.4 scipy==1.13.1 Keep the numpy and scipy with this version
# conda activate nt_finetune_v2
# Imports
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
# 1. AutoTokenizer
#     What it does: This class is a factory that automatically loads the correct tokenizer associated with a specific pre-trained model 
# checkpoint (e.g., an NT model on the Hugging Face Hub).
#     Why you use it: You don't need to know the exact tokenizer class (like BertTokenizer or DNATokenizer). You just provide the model's 
# name or path, and AutoTokenizer ensures you get the exact tokenization logic (like the 6-mer split, k-mer vocabulary, and special tokens) 
# that the original model was trained with. This is crucial for data consistency.
# 2. AutoModelForSequenceClassification
#     What it does: Similar to AutoTokenizer, this is a factory that automatically loads the weights of a pre-trained Transformer model 
# (like NT) and adds a sequence classification head on top.
#     Why you use it: This is the specific architecture for fine-tuning. It loads the massive, frozen Transformer layers and attaches a small, 
# learnable classification layer (or "head") with the appropriate number of output classes for your specific task (e.g., 2 classes for 
# benign/pathogenic, or 5 classes for tissue types).
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
#     What it does: The Trainer class is a high-level training API that orchestrates the entire fine-tuning process. It ties together the model,
#  data, and hyperparameters.
#     Why you use it: You initialize the Trainer by passing it four key ingredients:
#         The model (from AutoModelForSequenceClassification).
#         The training arguments (from TrainingArguments).
#         Your training and evaluation datasets.
#         A custom function to compute metrics (like accuracy or F1-score).
#     Execution: Once initialized, you simply call trainer.train(), and it handles the entire loop: moving data to the GPU, 
# calculating loss, backpropagation, gradient updates, and saving checkpoints.
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

# Promoter prediction is a sequence classification problem, in which the DNA sequence is predicted to be either a promoter or not.
# This task was introduced in [DeePromoter](https://www.frontiersin.org/articles/10.3389/fgene.2019.00286/full), where a set of 
# TATA and non-TATA promoters was gathered. A negative sequence was generated from each promoter, by randomly sampling subsets of 
# the sequence, to guarantee that some obvious motifs were present both in the positive and negative dataset.

num_labels_promoter = 2
# Load the model
model = AutoModelForSequenceClassification.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref", num_labels=num_labels_promoter)
# Some weights of EsmForSequenceClassification were not initialized from the model checkpoint at 
# InstaDeepAI/nucleotide-transformer-500m-human-ref and are newly initialized: 
# ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
# Frozen Weights: The vast majority of the 500M parameters (the entire Transformer backbone) were loaded successfully from the 
# checkpoint and are ready to be used.
# Newly Initialized Weights: The listed weights (classifier.dense.bias, etc.) belong to the newly added Classification Head. 
# The original pre-trained model (nucleotide-transformer-500m-human-ref) did not have this head because it was trained only 
# for the masked language modeling task.
# Initialization: These new weights are initialized with random values. They have no prior knowledge and must be learned 
# during your fine-tuning run.
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
model = model.to(device)

# This model architecture, displayed as an EsmForSequenceClassification object, is a Transformer model adapted 
# for a classification task. While the naming (Esm refers to Evolutionary Scale Modeling) originally comes from protein 
# language modeling, the structure here is a standard, large-scale BERT-style encoder applied to sequences—in your case, 
# DNA sequences (the Nucleotide Transformer).
# The architecture is split into three main parts: Embeddings, the Encoder (EsmModel), and the Classification Head.

EsmForSequenceClassification(
  (esm): EsmModel(
    (embeddings): EsmEmbeddings(
      (word_embeddings): Embedding(4105, 1280, padding_idx=1) # 6 mer sequence where 4105 = 4^6 = 4096 + special token, 1280 embedding dimensions
      (dropout): Dropout(p=0.0, inplace=False)
      (position_embeddings): Embedding(1002, 1280, padding_idx=1) # The maximum sequence length is 1000 token + start/end token
    )
    (encoder): EsmEncoder(
      (layer): ModuleList(
        (0-23): 24 x EsmLayer( # 24 identical Transformer layers stacked consecutively.
          (attention): EsmAttention( # This allows the model to dynamically weigh the importance of all other tokens in the sequence when processing a specific token
            (self): EsmSelfAttention(
              (query): Linear(in_features=1280, out_features=1280, bias=True)
              (key): Linear(in_features=1280, out_features=1280, bias=True)
              (value): Linear(in_features=1280, out_features=1280, bias=True)
            )
            (output): EsmSelfOutput(
              (dense): Linear(in_features=1280, out_features=1280, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (LayerNorm): LayerNorm((1280,), eps=1e-12, elementwise_affine=True)
          )
          (intermediate): EsmIntermediate( # Feed-Forward Network GELU. This layer applies a non-linear transformation that expands the dimension from 1280 to 5120 (a common ratio of 4:1 in Transformer models) and then compresses it back down.
            (dense): Linear(in_features=1280, out_features=5120, bias=True)
          )
          (output): EsmOutput( ## Then after learning by non-linear transforming reduce to 1280 from 5120.
            (dense): Linear(in_features=5120, out_features=1280, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (LayerNorm): LayerNorm((1280,), eps=1e-12, elementwise_affine=True)
        )
      )
      (emb_layer_norm_after): LayerNorm((1280,), eps=1e-12, elementwise_affine=True)
    )
    (contact_head): EsmContactPredictionHead( # This is a residual head from the original protein-focused ESM model used for tasks like contact prediction. For your DNA sequence classification task, this is likely unused or ignored during fine-tuning.
      (regression): Linear(in_features=480, out_features=1, bias=True)
      (activation): Sigmoid()
    )
  )
  (classifier): EsmClassificationHead(
    (dense): Linear(in_features=1280, out_features=1280, bias=True)
    (dropout): Dropout(p=0.0, inplace=False)
    (out_proj): Linear(in_features=1280, out_features=2, bias=True)
  )
  # Input: It takes the final hidden state (the 1280-dimensional vector) corresponding to the special [CLS] token (the first token) from the last encoder 
  # layer, which summarizes the entire sequence.
  #   classifier.dense: Linear(in_features=1280, out_features=1280)
  #   A simple fully connected layer for transformation, often accompanied by an activation function (like GELU, which is implicit).
  # 
  # classifier.out_proj (Output Projection): Linear(in_features=1280, out_features=2)
  # This is the final layer that maps the internal representation (1280 dimensions) to the number of classes in your fine-tuning task.
  # 2: This matches the num_labels=num_labels_promoter you provided, indicating you are performing a binary classification task 
  # (e.g., promoter vs. non-promoter).
)

# 2. Pre-Trained Weights (Backbone)
# This is the point that requires clarification:

#     The common practice (and the default in Hugging Face Trainer): The pre-trained backbone weights are NOT typically frozen. They are included in the training process, but they are often trained with a very small learning rate (a process called differential learning rates).
#     Why train them? Allowing a small change lets the model gently shift its learned understanding of the general "language of DNA" to better fit the specific nuances of your downstream task (e.g., promoter sequences).

# If you had truly frozen the backbone weights, only the classifier head would be updated. While that is one valid fine-tuning strategy (called linear probing), the most common and often highest-performing strategy is to update the entire model with a low learning rate.
# In conclusion: You are 100% correct that the architecture is the same and the new head weights are learned from scratch. You are largely correct that the pre-trained weights are protected, either by explicit freezing or, more commonly, by a much lower learning rate.

# from datasets import load_dataset, Dataset

# # Load the promoter dataset from the InstaDeep Hugging Face ressources
# dataset_name = "promoter_all"
# train_dataset_promoter = load_dataset(
#         "InstaDeepAI/nucleotide_transformer_downstream_tasks",
#         dataset_name,
#         split="train",
#         streaming= False,
#     )
# test_dataset_promoter = load_dataset(
#         "InstaDeepAI/nucleotide_transformer_downstream_tasks",
#         dataset_name,
#         split="test",
#         streaming= False,
#     )

from datasets import load_dataset

# Set the dataset_name to None or 'default' since the prompt is about the config name
# but 'promoter_all' is actually a value in the 'task' column.
dataset_config_name = None 

# Load the full downstream tasks dataset
full_train_dataset = load_dataset(
    "InstaDeepAI/nucleotide_transformer_downstream_tasks",
    dataset_config_name,  # Pass None to use the default configuration
    split="train",
    streaming=False,
)

# Filter the dataset for the specific task "promoter_all"
# Filter the training split
train_dataset_enhancers = full_train_dataset.filter(
    lambda example: example["task"] == "enhancers_types"
)

# Load the full test split
full_test_dataset = load_dataset(
    "InstaDeepAI/nucleotide_transformer_downstream_tasks",
    None,  # Use 'None' for the default configuration
    split="test",
    streaming=False,
)

# Filter the test split
test_dataset_enhancers = full_test_dataset.filter(
    lambda example: example["task"] == "enhancers_types"
)


# >>> train_dataset_promoter
# Dataset({
#     features: ['sequence', 'name', 'label', 'task'],
#     num_rows: 461850
# })
# Input: The sequence column provides the raw DNA text.
# Target: The label column provides the 0/1 ground truth for classification.
# Efficiency: The MemoryMappedTable ensures that despite the large size (461,850 rows), the training process can 
# be run efficiently without consuming excessive memory.

# Get training data
train_sequences_promoter = train_dataset_promoter['sequence']
train_labels_promoter = train_dataset_promoter['label']

# Split the dataset into a training and a validation dataset
train_sequences_promoter, validation_sequences_promoter, train_labels_promoter, validation_labels_promoter = train_test_split(train_sequences_promoter,
                                                                              train_labels_promoter, test_size=0.05, random_state=42)

# Get test data
test_sequences_promoter = test_dataset_promoter['sequence']
test_labels_promoter = test_dataset_promoter['label']

# Let us have a look at the data. If we extract the last sequence of the dataset, we see that it is indeed a promoter, 
# as its label is 1. Furthermore, we can also see that it is a TATA promoter, as the TATA motif is present at the 
# 221th nucleotide of the sequence!
#  This position corresponds to the known −25 to −35 region where the TATA box is expected to be found in a TATA-containing promoter.
# Since the sequence is 500 nucleotide from TSS + 250 and -250. So from -250 - 25 to -35 = 221 

idx_sequence = -1
sequence, label = train_sequences_promoter[idx_sequence], train_labels_promoter[idx_sequence]
print(f"The DNA sequence is {sequence}.")
print(f"Its associated label is label {label}.")

idx_TATA = sequence.find("TATA")
print(f"This promoter is a TATA promoter, as the TATA motif is present at the {idx_TATA}th nucleotide.")
# This promoter is a TATA promoter, as the TATA motif is present at the 58th nucleotide.

# All inputs to neural nets must be numerical. The process of converting strings into numerical indices suitable for a 
# neural net is called tokenization.

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
# AutoTokenizer.from_pretrained(...): This command downloads and loads the pre-trained tokenizer associated with the 
# specified model, "InstaDeepAI/nucleotide-transformer-500m-human-ref".
# Purpose: The tokenizer converts raw DNA sequences (strings of 'A', 'C', 'G', 'T') into numerical input IDs and 
# attention masks that the transformer model can understand. For the Nucleotide Transformer, this typically involves 
# k-mer tokenization (breaking the sequence into overlapping or non-overlapping short nucleotide chunks). i.e. into 6 mers
# So the sequence which is currently 300 bp converted into 50 tokens.

# Promoter dataset
# Dataset.from_dict(...): This converts standard Python dictionaries into highly optimized Hugging Face Dataset objects.
ds_train_promoter = Dataset.from_dict({"data": train_sequences_promoter,'labels':train_labels_promoter})
ds_validation_promoter = Dataset.from_dict({"data": validation_sequences_promoter,'labels':validation_labels_promoter})
ds_test_promoter = Dataset.from_dict({"data": test_sequences_promoter,'labels':test_labels_promoter})
# ds_train_promoter in here the sequence length is 300 nucleotide

def tokenize_function(examples):
    outputs = tokenizer(examples["data"])
    return outputs

# Creating tokenized promoter dataset
tokenized_datasets_train_promoter = ds_train_promoter.map(
    tokenize_function,
    batched=True, # It tells the mapping process to send multiple examples at once to tokenize_function, which allows the tokenizer to leverage its vectorized and optimized code.
    remove_columns=["data"], ## Remove the sequences since now we have save the tokens so no longer required 
)
tokenized_datasets_validation_promoter = ds_validation_promoter.map(
    tokenize_function,
    batched=True,
    remove_columns=["data"],
)
tokenized_datasets_test_promoter = ds_test_promoter.map(
    tokenize_function,
    batched=True,
    remove_columns=["data"],
)

## Fine-tuning and evaluation
# The hyper-parameters introduced here are different from the ones used in the paper since we are training the whole 
# model. Further hyper-parameters search will surely improve the performance on the task!. We initialize our 
# TrainingArguments. These control the various training hyperparameters, and will be passed to our Trainer.

batch_size = 8
model_name='nucleotide-transformer'
args_promoter = TrainingArguments(
    f"{model_name}-finetuned-NucleotideTransformer",
    remove_unused_columns=False,
    eval_strategy="steps",
    save_strategy="steps",
    learning_rate=1e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps= 1,
    per_device_eval_batch_size= 64,
    num_train_epochs= 2,
    logging_steps= 100,
    load_best_model_at_end=True,  # Keep the best model according to the evaluation
    metric_for_best_model="f1_score",
    label_names=["labels"],
    dataloader_drop_last=True,
    max_steps= 1000
)

# Next, we define the metric we will use to evaluate our models and write a compute_metrics function. 
# We can load this from the scikit-learn library.
# Define the metric for the evaluation using the f1 score
def compute_metrics_f1_score(eval_pred):
    """Computes F1 score for binary classification"""
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    references = eval_pred.label_ids
    r={'f1_score': f1_score(references, predictions)}
    return r

trainer = Trainer(
    model.to(device),
    args_promoter,
    train_dataset= tokenized_datasets_train_promoter,
    eval_dataset= tokenized_datasets_validation_promoter,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_f1_score,
)

# Hyperparameter	Value	Explanation
# learning_rate	1e-5 (or 1×10−5)	This is the most crucial optimization hyperparameter. It determines the step size the 
# optimizer takes when updating the model's weights based on the loss gradient. A small value (like 10−5) is typical when 
# fine-tuning a pre-trained model to ensure the large pre-trained weights are adjusted gradually without destroying the 
# learned representations.
# per_device_train_batch_size	8	The number of samples processed on a single GPU before calculating a gradient step.
# gradient_accumulation_steps	1	The number of steps the gradients are accumulated over before a single parameter 
# update is performed. A value of 1 means a parameter update happens after every batch. Increasing this value can 
# simulate a larger effective batch size when limited by GPU memory (Effective Batch Size = 8×1=8).
# num_train_epochs	2	The number of complete passes over the entire training dataset. For fine-tuning, a small number 
# of epochs is often sufficient.
# max_steps	1000	Overrides num_train_epochs. The training process will stop after reaching 1000 total optimization 
# steps, regardless of how many epochs that takes. This provides precise control over the training duration.
# dataloader_drop_last	TRUE	If the total number of training samples is not perfectly divisible by the batch size, 
# the last (smaller) batch is discarded. This is usually done to ensure consistent batch sizes, which can be important 
# for distributed training or some model architectures.

# We can now finetune our model by just calling the train method:
train_results = trainer.train()
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [15:46<00:00,  1.06it/s]

curve_evaluation_f1_score =[[a['step'],a['eval_f1_score']] for a in trainer.state.log_history if 'eval_f1_score' in a.keys()]
eval_f1_score = [c[1] for c in curve_evaluation_f1_score]
steps = [c[0] for c in curve_evaluation_f1_score]

pd.DataFrame(curve_evaluation_f1_score)
# step eval_f1_score
# 100  0.896191
# 200  0.902549
# 300  0.922675
# 400  0.926667
# 500  0.930654
# 600  0.929023
# 700  0.930103
# 800  0.937952
# 900  0.937936
# 1000  0.939582

plt.plot(steps, eval_f1_score, 'b', label='Validation F1 score')
plt.title('Validation F1 score for promoter prediction')
plt.xlabel('Number of training steps performed')
plt.ylabel('Validation F1 score')
plt.legend()
plt.show()
plt.savefig("/mnt/data/projects/.immune/Munich/DNA-Language-Model/code/promotor_model_Eval.pdf")

# F1 score on the test dataset
# Compute the F1 score on the test dataset :
print(f"F1 score on the test dataset: {trainer.predict(tokenized_datasets_test_promoter).metrics['test_f1_score']}")
# 100%|█████████████████████████████████████████████████████████| 92/92 [01:23<00:00,  1.11it/s]
# F1 score on the test dataset: 0.9333564734467199