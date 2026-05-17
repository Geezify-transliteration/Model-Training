# TPU-Optimized Training Pipeline for Amharic Transliteration

This guide adapts your notebook for:

- Cloud TPU VM (v5e / v6e)
- PyTorch/XLA TPU training
- Automatic checkpoint saving
- Google Cloud Storage (GCS) integration
- Recovery after TPU preemption
- Faster training on TPU

---

# 1. Recommended TPU Workflow

Instead of storing files directly on the TPU VM:

```text
GitHub -> source code
Google Cloud Storage -> datasets + checkpoints
TPU VM -> temporary compute
```

This is important because spot TPUs can disappear.

---

# 2. Create a GCS Bucket

From your LOCAL machine:

```bash
gcloud storage buckets create gs://YOUR_BUCKET_NAME
```

Example:

```bash
gcloud storage buckets create gs://geezify-translit-bucket
```

---

# 3. Upload Dataset to GCS

Example:

```bash
gcloud storage cp amharic_clean_merged.csv gs://geezify-translit-bucket/data/
```

---

# 4. SSH Into TPU VM

```bash
gcloud compute tpus tpu-vm ssh my-v6e --zone=europe-west4-a
```

---

# 5. Create Working Directory

Inside TPU VM:

```bash
mkdir -p ~/projects/transliteration
cd ~/projects/transliteration
```

---

# 6. Create Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# 7. Install TPU-Compatible Packages

```bash
pip install -U pip
```

Install PyTorch/XLA TPU stack:

```bash
pip install torch~=2.7.0 torch_xla[tpu]~=2.7.0
```

Install training libraries:

```bash
pip install transformers datasets evaluate sacrebleu jiwer sentencepiece accelerate pandas scikit-learn gcsfs
```

---

# 8. Download Dataset From GCS

Inside TPU VM:

```bash
gcloud storage cp gs://geezify-translit-bucket/data/amharic_clean_merged.csv .
```

---

# 9. TPU-Optimized Training Script

Create a file called:

```text
train_tpu.py
```

Use this code:

```python
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import evaluate
from datasets import Dataset, DatasetDict

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# TPU/XLA
import torch_xla
import torch_xla.core.xla_model as xm

# =========================
# CONFIG
# =========================

CLEAN_CSV_PATH = "data/merged_normalized.csv"

MODEL_CHECKPOINT = "google/mt5-small"

OUTPUT_DIR = "./outputs"

GCS_CHECKPOINT_DIR = "gs://geezify-translit-bucket/checkpoints"

TEST_SIZE = 0.1
RANDOM_STATE = 42

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128

BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3

USE_SUBSET = False
SUBSET_ROWS = 1000

# =========================
# LOAD DATA
# =========================

print("Loading dataset...")

assert os.path.exists(CLEAN_CSV_PATH)

df = pd.read_csv(CLEAN_CSV_PATH)

if USE_SUBSET:
    df = df.sample(min(SUBSET_ROWS, len(df)), random_state=42)

# Expected format:
# romanized_text,amharic_text

required_columns = {"romanized_text", "amharic_text"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Dataset must contain columns: {required_columns}"
    )

# Remove missing values

df = df.dropna(subset=["romanized_text", "amharic_text"])

# Convert to string

df["romanized_text"] = df["romanized_text"].astype(str)
df["amharic_text"] = df["amharic_text"].astype(str)

# Remove empty rows

df = df[
    (df["romanized_text"].str.strip() != "")
    & (df["amharic_text"].str.strip() != "")
]

# Fixed transliteration direction: Romanized -> Amharic

df["source"] = df["romanized_text"]
df["target"] = df["amharic_text"]

train_ready_df = df[["source", "target"]].copy()

print(f"Dataset size after cleaning: {len(train_ready_df)}")

train_df, test_df = train_test_split(
    train_ready_df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

raw_datasets = DatasetDict(
    {
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df),
    }
)

# =========================
# TOKENIZER + MODEL
# =========================

print("Loading tokenizer/model...")

# use_fast=False helps some multilingual tokenizers

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_CHECKPOINT,
    use_fast=False,
)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

# =========================
# PREPROCESSING
# =========================


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["source"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )

    labels = tokenizer(
        text_target=examples["target"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


print("Tokenizing dataset...")

# num_proc speeds up preprocessing on TPU VM CPUs

tokenized_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=raw_datasets["train"].column_names,
)

# =========================
# METRICS
# =========================

metric = evaluate.load("sacrebleu")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels



def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(
        preds,
        skip_special_tokens=True,
    )

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True,
    )

    decoded_preds, decoded_labels = postprocess_text(
        decoded_preds,
        decoded_labels,
    )

    result = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
    )

    return {"bleu": result["score"]}

# =========================
# DATA COLLATOR
# =========================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)

# =========================
# TRAINING ARGS
# =========================

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    weight_decay=WEIGHT_DECAY,
    num_train_epochs=NUM_EPOCHS,

    predict_with_generate=True,

    logging_steps=20,

    evaluation_strategy="steps",
    eval_steps=200,

    save_strategy="steps",
    save_steps=200,

    save_total_limit=3,

    load_best_model_at_end=True,

    bf16=True,

    report_to="none",

    dataloader_num_workers=4,

    remove_unused_columns=True,

    push_to_hub=False,
)

# =========================
# TRAINER
# =========================

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# =========================
# TRAIN
# =========================

print("Starting TPU training...")

trainer.train()

# =========================
# SAVE FINAL MODEL
# =========================

print("Saving model...")

trainer.save_model("final_model")
tokenizer.save_pretrained("final_model")

# =========================
# UPLOAD CHECKPOINTS TO GCS
# =========================

print("Uploading checkpoints to GCS...")

os.system(f"gcloud storage cp -r {OUTPUT_DIR} {GCS_CHECKPOINT_DIR}")

os.system(
    f"gcloud storage cp -r final_model gs://geezify-translit-bucket/final_model"
)

print("Training complete.")
```

---

# 10. Start Training

Inside TPU VM:

```bash
python train_tpu.py
```

---

# 11. Resume Training After TPU Preemption

If TPU dies:

1. recreate TPU VM
2. SSH again
3. re-clone repo
4. re-download dataset
5. restore checkpoint

Example:

```bash
gcloud storage cp -r gs://geezify-translit-bucket/checkpoints ./outputs
```

Then:

```python
trainer.train(resume_from_checkpoint=True)
```

---

# 12. Recommended Improvements

After this works, consider:

- gradient accumulation
- larger mt5 models
- sentencepiece vocabulary adaptation
- mixed direction training
- curriculum learning
- wandb logging
- distributed TPU slices

---

# 13. Important TPU Notes

TPU best practices:

- save checkpoints often
- use GCS for persistence
- avoid storing important work only on TPU VM
- use bf16 on TPU
- keep batch sizes powers of 2 when possible
- test on small subsets first

---

# 14. Recommended Development Workflow

Use:

```text
VSCode Remote SSH -> TPU VM
```

This gives:

- notebook support
- file explorer
- integrated terminal
- remote Python execution
- direct editing on TPU VM

This is the cleanest TPU research workflow.

