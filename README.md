# DNA Language Model: Nucleotide Transformers

Pretraining with GPT2, Mistral AI architecture and Fine-tuning DNA language models (Nucleotide Transformers) for genomic prediction tasks including promoter/enhancer identification, variant effect prediction, and more.

## Overview

This project applies transformer-based language models pre-trained on genomic sequences to various downstream prediction tasks, leveraging DNA sequence representations for biological insights.

## Key Applications

### 1. **Promoter & Enhancer Region Identification**
Identifies regulatory DNA elements using sequence context.

### 2. **Variant Effect Prediction**
Predicts the functional impact of genomic variants on:
- Gene expression
- Protein function
- Disease association

### 3. **GWAS (Genome-Wide Association Studies)**
Associates genetic variants with traits and diseases.

### 4. **Transcription Factor Binding Prediction**
Identifies DNA motifs and binding sites for transcription factors.

### 5. **Epigenetic Marks Prediction**
Predicts histone modifications and chromatin states.

### 6. **Chromatin Profile Prediction**
Models chromatin accessibility and 3D structure from sequence alone.

## Methodology

- **Base Model**: Nucleotide Transformers (pre-trained on genomic sequences)
- **Fine-tuning Approach**: Task-specific adaptation with frozen/unfrozen layers
- **Evaluation Metrics**: AUC, F1-score, correlation for classification/regression tasks


## Installation

```bash
conda create -n torch_gpu python=3.10
conda activate dna-lm
pip install torch transformers huggingface-hub biopython pandas numpy
```

## Quick Start

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained Nucleotide Transformer
model_name = "InstaDeep/nucleotide-transformer-v2-500m-human-ref"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# Tokenize sequence
sequence = "ACGTACGTACGTACGT"
inputs = tokenizer(sequence, return_tensors="pt", padding=True)

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
```

## Fine-tuning Example

```bash
# Promoter prediction fine-tuning
python scripts/finetune.py \
    --model_name "InstaDeep/nucleotide-transformer-v2-500m-human-ref" \
    --task "promoter_detection" \
    --data_path "data/promoter_sequences.fasta" \
    --output_dir "models/promoter_model" \
    --epochs 5 \
    --batch_size 8
```

## Key References

- **Nucleotide Transformers**: [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v1)
- **DNABERT**: [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.02.20.431900v1)
- **Enformer**: [Nature Methods](https://www.nature.com/articles/s41592-021-01252-x)

## Contributing

When adding new prediction tasks:
1. Document the biological rationale
2. Provide benchmark datasets
3. Include baseline performance metrics
4. Test on held-out validation sets

## Contact

For questions about specific genomic tasks or model applications, refer to task-specific scripts.
