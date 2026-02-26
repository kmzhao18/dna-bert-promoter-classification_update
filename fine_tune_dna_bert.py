# -*- coding: utf-8 -*-
"""Fine_tune_DNA_Bert.ipynb

"""

# Install required packages (quiet mode)
!pip install -q transformers datasets peft accelerate biopython scikit-learn matplotlib

import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split

# innitate GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# create traning set example

def generate_random_dna(length=100):
    return "".join(random.choices(["A","C","G","T"], k=length))

def insert_tata(sequence):
    """
    Insert canonical TATA-box motif randomly
    """
    pos = random.randint(20, 60)
    return sequence[:pos] + "TATAAA" + sequence[pos+6:]

sequences = []
labels = []

for _ in range(1000):
    seq = generate_random_dna(100)

    if random.random() > 0.5:
        seq = insert_tata(seq)
        labels.append(1)  # promoter
    else:
        labels.append(0)  # non-promoter

    sequences.append(seq)

print("Example sequence:", sequences[0])
print("Label:", labels[0])

train_seqs, val_seqs, train_labels, val_labels = train_test_split(
    sequences,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# load DNABert
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "zhihan1996/DNA_bert_6"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
).to(device)

# build pytorch dataset
from torch.utils.data import Dataset

class DNADataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.sequences[idx],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx])
        }

train_dataset = DNADataset(train_seqs, train_labels)
val_dataset = DNADataset(val_seqs, val_labels)

# fine tune model
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./dna_promoter_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_steps=20,
    fp16=torch.cuda.is_available(),
    do_train=True,
    do_eval=True,
    report_to=[]   # disable external logging safely
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

metrics = trainer.evaluate()
print(metrics)

# evaluate accuracy
import numpy as np
from sklearn.metrics import accuracy_score

predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=1)

print("Validation Accuracy:", accuracy_score(val_labels, preds))



