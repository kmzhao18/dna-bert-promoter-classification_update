Fine-Tuning DNABERT for Promoter (TATA-Box) Classification



This project demonstrates how to fine-tune a pretrained DNABERT model for DNA sequence classification using synthetic promoter data. The notebook walks through dataset generation, preprocessing, model fine-tuning, and evaluation using the Hugging Face ecosystem.

The model used in this notebook is available via the Hugging Face model hub and is trained with the PyTorch backend.



📌 Project Overview

The goal of this notebook is to:

Generate synthetic DNA sequences

Insert a canonical TATA-box motif (TATAAA) into positive samples

Fine-tune a pretrained DNABERT model for binary classification:

1 → Promoter (contains TATA-box)

0 → Non-promoter

Evaluate classification accuracy on a validation set

This is a minimal, educational example of applying transformer models to genomics.



🧪 Dataset

The dataset is synthetically generated:

1,000 random DNA sequences

Sequence length: 100 nucleotides

50% contain a randomly inserted TATAAA motif

80/20 train-validation split (stratified)

Example DNA sequence:

ACGTGCTAGCTAGTACGATCGATCG...


🤖 Model

Pretrained model used:

zhihan1996/DNA_bert_6

Loaded via:

from transformers import AutoTokenizer, AutoModelForSequenceClassification

Configuration:

num_labels=2

Max sequence length: 128

Fine-tuned for 3 epochs

Mixed precision (FP16) enabled if GPU available



⚙️ Training Configuration

Key training parameters:

Parameter	Value
Batch size	8
Epochs	3
Optimizer	Default Hugging Face Trainer
Logging steps	20
Evaluation	After training

Training is handled via:

from transformers import TrainingArguments, Trainer


📊 Evaluation

After training:

Predictions are generated on validation set

Accuracy is computed using sklearn.metrics.accuracy_score

Example output:

Validation Accuracy: ~0.9+

(Exact performance varies due to random initialization and synthetic data.)



📦 Requirements

Install dependencies:

pip install transformers datasets peft accelerate biopython scikit-learn matplotlib torch


🖥 Hardware Support

The notebook automatically detects GPU:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Works on CPU (slower)

Optimized for CUDA-enabled GPU



📁 Output

Fine-tuned model is saved to:

./dna_promoter_model

You can later reload it with:

model = AutoModelForSequenceClassification.from_pretrained("./dna_promoter_model")



🚀 How to Run

Clone this repository

Open the notebook:

Fine_tune_DNA_Bert.ipynb

Run all cells sequentially



Compare against CNN/LSTM baselines

