### For Enhancer
## Previously performed the Nucleotide Transformer on Promoter_FineTune.py
# conda activate nt_finetune_v2
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
num_labels_enhancers_types = 3
# Load the model
model = AutoModelForSequenceClassification.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref", num_labels=num_labels_enhancers_types)
model = model.to(device)

EsmForSequenceClassification(
  (esm): EsmModel(
    (embeddings): EsmEmbeddings(
      (word_embeddings): Embedding(4105, 1280, padding_idx=1)
      (dropout): Dropout(p=0.0, inplace=False)
      (position_embeddings): Embedding(1002, 1280, padding_idx=1)
    )
    (encoder): EsmEncoder(
      (layer): ModuleList(
        (0-23): 24 x EsmLayer(
          (attention): EsmAttention(
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
          (intermediate): EsmIntermediate(
            (dense): Linear(in_features=1280, out_features=5120, bias=True)
          )
          (output): EsmOutput(
            (dense): Linear(in_features=5120, out_features=1280, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (LayerNorm): LayerNorm((1280,), eps=1e-12, elementwise_affine=True)
        )
      )
      (emb_layer_norm_after): LayerNorm((1280,), eps=1e-12, elementwise_affine=True)
    )
    (contact_head): EsmContactPredictionHead(
      (regression): Linear(in_features=480, out_features=1, bias=True)
      (activation): Sigmoid()
    )
  )
  (classifier): EsmClassificationHead(
    (dense): Linear(in_features=1280, out_features=1280, bias=True)
    (dropout): Dropout(p=0.0, inplace=False)
    (out_proj): Linear(in_features=1280, out_features=3, bias=True)
  )
)

from datasets import load_dataset, Dataset

# Load the enhancers dataset from the InstaDeep Hugging Face ressources
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
    lambda example: example["task"] == "promoter_all"
)

# Load the full test split
full_test_dataset = load_dataset(
    "InstaDeepAI/nucleotide_transformer_downstream_tasks",
    None,  # Use 'None' for the default configuration
    split="test",
    streaming=False,
)

# Filter the test split
test_dataset_promoter = full_test_dataset.filter(
    lambda example: example["task"] == "promoter_all"
)

#  Get training data
train_sequences_enhancers = train_dataset_enhancers['sequence']
train_labels_enhancers = train_dataset_enhancers['label']

# Split the dataset into a training and a validation dataset
train_sequences_enhancers, validation_sequences_enhancers, train_labels_enhancers, validation_labels_enhancers = train_test_split(train_sequences_enhancers,
                                                                              train_labels_enhancers, test_size=0.10, random_state=42)

# Get test data
test_sequences_enhancers = test_dataset_enhancers['sequence']
test_labels_enhancers = test_dataset_enhancers['label']

# Tokenizing the datasets
# Enhancer dataset
ds_train_enhancers = Dataset.from_dict({"data": train_sequences_enhancers,'labels':train_labels_enhancers})
ds_validation_enhancers = Dataset.from_dict({"data": validation_sequences_enhancers,'labels':validation_labels_enhancers})
ds_test_enhancers = Dataset.from_dict({"data": test_sequences_enhancers,'labels':test_labels_enhancers})

# Creating tokenized enhancer dataset
tokenized_datasets_train_enhancers = ds_train_enhancers.map(
    tokenize_function,
    batched=True,
    remove_columns=["data"],
)
# Map: 100%|█████████████████████████████████████████████████████████████████████████████| 13471/13471 [00:03<00:00, 3628.05 examples/s]
tokenized_datasets_validation_enhancers = ds_validation_enhancers.map(
    tokenize_function,
    batched=True,
    remove_columns=["data"],
)
# Map: 100%|█████████████████████████████████████████████████████████████████████████████| 1497/1497 [00:00<00:00, 3820.00 examples/s]
tokenized_datasets_test_enhancers = ds_test_enhancers.map(
    tokenize_function,
    batched=True,
    remove_columns=["data"],
)
# Map: 100%|█████████████████████████████████████████████████████████████████████████████| 400/400 [00:00<00:00, 3701.10 examples/s]
# Fine Tuning 
batch_size = 8
model_name='nucleotide-transformer_enhancer'
args_enhancers = TrainingArguments(
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
    metric_for_best_model="mcc_score", # The mcc_score on the evaluation dataset used to select the best model
    label_names=["labels"],
    dataloader_drop_last=True,
    max_steps= 1000
)

# Define the metric for the evaluation
def compute_metrics_mcc(eval_pred):
    """Computes Matthews correlation coefficient (MCC score) for binary classification"""
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    references = eval_pred.label_ids
    r={'mcc_score': matthews_corrcoef(references, predictions)}
    return r

trainer = Trainer(
    model,
    args_enhancers,
    train_dataset= tokenized_datasets_train_enhancers,
    eval_dataset= tokenized_datasets_validation_enhancers,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_mcc,
)

train_results = trainer.train()
# {'loss': 0.4563, 'grad_norm': 5.6018500328063965, 'learning_rate': 9.01e-06, 'epoch': 0.06}
# {'eval_loss': 0.3868066966533661, 'eval_mcc_score': 0.7759834926549631, 'eval_runtime': 17.1198, 'eval_samples_per_second': 87.442, 'eval_steps_per_second': 1.402, 'epoch': 0.06}
# {'loss': 0.3052, 'grad_norm': 39.665775299072266, 'learning_rate': 8.010000000000001e-06, 'epoch': 0.12}
# {'eval_loss': 0.5572882294654846, 'eval_mcc_score': 0.764205115195118, 'eval_runtime': 16.7554, 'eval_samples_per_second': 89.344, 'eval_steps_per_second': 1.432, 'epoch': 0.12}
# {'loss': 0.291, 'grad_norm': 8.146517753601074, 'learning_rate': 7.01e-06, 'epoch': 0.18}
# {'eval_loss': 1.2418009042739868, 'eval_mcc_score': 0.6293870789682456, 'eval_runtime': 17.0434, 'eval_samples_per_second': 87.835, 'eval_steps_per_second': 1.408, 'epoch': 0.18}
# {'loss': 0.3154, 'grad_norm': 0.07464942336082458, 'learning_rate': 6.01e-06, 'epoch': 0.24}
# {'eval_loss': 0.5305999517440796, 'eval_mcc_score': 0.763446707764568, 'eval_runtime': 16.9902, 'eval_samples_per_second': 88.109, 'eval_steps_per_second': 1.413, 'epoch': 0.24}
# {'loss': 0.3011, 'grad_norm': 0.9514262080192566, 'learning_rate': 5.01e-06, 'epoch': 0.3}
# {'eval_loss': 0.6188954710960388, 'eval_mcc_score': 0.7431953237517085, 'eval_runtime': 16.9998, 'eval_samples_per_second': 88.06, 'eval_steps_per_second': 1.412, 'epoch': 0.3}
# {'loss': 0.4663, 'grad_norm': 22.48845100402832, 'learning_rate': 4.0100000000000006e-06, 'epoch': 0.36}
# {'eval_loss': 0.38267526030540466, 'eval_mcc_score': 0.7808210195037593, 'eval_runtime': 17.1377, 'eval_samples_per_second': 87.351, 'eval_steps_per_second': 1.4, 'epoch': 0.36}
# {'loss': 0.3995, 'grad_norm': 5.422210693359375, 'learning_rate': 3.01e-06, 'epoch': 0.42}
# {'eval_loss': 0.3436007797718048, 'eval_mcc_score': 0.8071010964685599, 'eval_runtime': 17.1113, 'eval_samples_per_second': 87.486, 'eval_steps_per_second': 1.403, 'epoch': 0.42}
# {'loss': 0.3925, 'grad_norm': 17.821670532226562, 'learning_rate': 2.0100000000000002e-06, 'epoch': 0.48}
# {'eval_loss': 0.3323294222354889, 'eval_mcc_score': 0.7958085547363439, 'eval_runtime': 17.0612, 'eval_samples_per_second': 87.743, 'eval_steps_per_second': 1.407, 'epoch': 0.48}
# {'loss': 0.3776, 'grad_norm': 23.42879867553711, 'learning_rate': 1.01e-06, 'epoch': 0.53}
# {'eval_loss': 0.3194162845611572, 'eval_mcc_score': 0.8051601717119992, 'eval_runtime': 16.9478, 'eval_samples_per_second': 88.33, 'eval_steps_per_second': 1.416, 'epoch': 0.53}
# {'loss': 0.3982, 'grad_norm': 31.059839248657227, 'learning_rate': 1e-08, 'epoch': 0.59}
# {'eval_loss': 0.3337206542491913, 'eval_mcc_score': 0.7988266747123874, 'eval_runtime': 17.0225, 'eval_samples_per_second': 87.943, 'eval_steps_per_second': 1.41, 'epoch': 0.59}

# 100%|█████████████████████████████████████████████████████| 1000/1000 [09:58<00:00,  2.42it/s]

curve_evaluation_mcc_score=[[a['step'],a['eval_mcc_score']] for a in trainer.state.log_history if 'eval_mcc_score' in a.keys()]
eval_mcc_score = [c[1] for c in curve_evaluation_mcc_score]
steps = [c[0] for c in curve_evaluation_mcc_score]

plt.figure()   
plt.plot(steps, eval_mcc_score, 'b', label='Validation MCC score')
plt.title('Validation MCC score for enhancer prediction')
plt.xlabel('Number of training steps performed')
plt.ylabel('Validation MCC score')
plt.legend()
plt.savefig("enhancer_validation_MCC_score.pdf")
plt.show()