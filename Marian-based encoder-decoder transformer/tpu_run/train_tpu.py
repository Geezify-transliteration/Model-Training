#!/usr/bin/env python3
"""
TPU-only transliteration training (mT5 seq2seq). No Colab / google.colab.

Usage (after ./install_tpu_deps.sh once):
  export PJRT_DEVICE=TPU
  python3 train_tpu.py --data_csv /path/to/merged_normalized.csv

Or: ./run_all.sh --data_csv ./data/merged_normalized.csv

Put your CSV on the TPU VM (scp, gcsfuse, or download). Required columns:
  romanized_text, amharic_text
"""

from __future__ import annotations

import argparse
import math
import os
import sys

# PJRT before torch is imported downstream
os.environ.setdefault("PJRT_DEVICE", "TPU")

try:
    import torch_xla.core.xla_model as xm
except ImportError as exc:
    sys.exit(
        "torch_xla is not installed. On a Cloud TPU VM run:\n"
        "  ./install_tpu_deps.sh\n"
        "See https://cloud.google.com/tpu/docs/run-calculation-pytorch\n"
        f"Original error: {exc}"
    )

import numpy as np
import pandas as pd
import evaluate
import jiwer
from sklearn.model_selection import train_test_split

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


_GCS_SCOPES = ("https://www.googleapis.com/auth/devstorage.full_control",)


def _gc_credentials():
    try:
        from google.colab import auth as _colab_auth  # type: ignore

        _colab_auth.authenticate_user()
    except ImportError:
        pass
    import google.auth

    credentials, project = google.auth.default(scopes=_GCS_SCOPES)
    return credentials, project


def _gcs_fs():
    import gcsfs

    credentials, project = _gc_credentials()
    return gcsfs.GCSFileSystem(project=project, token=credentials)


def ensure_gcs_bucket_exists(bucket: str, location: str) -> None:
    from google.cloud import storage

    credentials, project = _gc_credentials()
    client = storage.Client(project=project, credentials=credentials)
    b = client.bucket(bucket)
    if b.exists():
        print(f"GCS bucket gs://{bucket} found.")
        return
    client.create_bucket(bucket, location=location)
    print(f"Created gs://{bucket} (location={location}).")


def resolve_gcs_output_dir(bucket: str, location: str) -> str:
    ensure_gcs_bucket_exists(bucket, location)
    fs = _gcs_fs()
    models_prefix = f"{bucket}/models"
    final_marker = f"{models_prefix}/final/config.json"

    def has_hf_checkpoint(prefix: str) -> bool:
        return bool(fs.glob(f"{prefix}/checkpoint-*/trainer_state.json"))

    if not fs.exists(final_marker):
        out = f"gs://{models_prefix}/checkpoints"
        print(f"OUTPUT_DIR (no published final yet): {out}")
        return out

    last_k = 0
    for k in range(1, 128):
        if has_hf_checkpoint(f"{models_prefix}/checkpoints{k}"):
            last_k = k
    if last_k:
        out = f"gs://{models_prefix}/checkpoints{last_k}"
        print(f"OUTPUT_DIR (resume numbered run after final): {out}")
        return out
    out = f"gs://{models_prefix}/checkpoints1"
    print(f"OUTPUT_DIR (new numbered run; models/final exists): {out}")
    return out


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def _transliteration_metrics(refs_flat, hyps):
    if not hyps:
        return {"cer": 0.0, "word_accuracy": 1.0, "mean_char_edit_distance": 0.0}
    cer = jiwer.cer(refs_flat, hyps)
    wer = jiwer.wer(refs_flat, hyps)
    word_accuracy = 1.0 - wer
    char_edits = []
    for r, h in zip(refs_flat, hyps):
        m = jiwer.compute_measures(r, h)
        char_edits.append(m["substitutions"] + m["deletions"] + m["insertions"])
    mean_char_edit = float(np.mean(char_edits))
    return {
        "cer": float(cer),
        "word_accuracy": float(word_accuracy),
        "mean_char_edit_distance": mean_char_edit,
    }


def build_compute_metrics(tokenizer_ref, bleu_metric):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.clip(preds, 0, tokenizer_ref.vocab_size - 1)
        decoded_preds = tokenizer_ref.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer_ref.pad_token_id)
        decoded_labels = tokenizer_ref.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = bleu_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
        )
        ref_flat = [row[0] for row in decoded_labels]
        tm = _transliteration_metrics(ref_flat, decoded_preds)
        return {
            "bleu": result["score"],
            "cer": tm["cer"],
            "word_accuracy": tm["word_accuracy"],
            "mean_char_edit_distance": tm["mean_char_edit_distance"],
        }

    return compute_metrics


def _print_xla_tpu_devices() -> None:
    pjrt = os.environ.get("PJRT_DEVICE", "")
    if pjrt and pjrt.upper() != "TPU":
        print(
            f"Warning: PJRT_DEVICE={pjrt!r}; expected TPU. Try: export PJRT_DEVICE=TPU",
            file=sys.stderr,
        )
    try:
        devices = xm.get_xla_supported_devices("TPU")
    except RuntimeError as exc:
        err = str(exc).lower()
        if "topology" in err or "global tpu" in err:
            sys.exit(
                "PyTorch/XLA could not read the TPU topology.\n\n"
                "Checks:\n"
                "  - export PJRT_DEVICE=TPU   (ensure you did not set PJRT_DEVICE=CPU)\n"
                "  - unset XRT_TPU_CONFIG    (legacy XRT env vars conflict with PJRT)\n"
                "  - pip show libtpu          (re-run repair_tpu_torch_abi.sh if missing)\n\n"
                "Multi-host TPU slices (multiple VMs; hostname like ...-w-0, ...-w-1, ...):\n"
                "  Start the same command on every worker together, not only on worker 0.\n"
                "  Example:\n"
                '    gcloud compute tpus tpu-vm ssh TPU_NAME --zone ZONE --worker=all \\\n'
                '      --command="PJRT_DEVICE=TPU bash -lc \'cd /path/to/tpu_run && '
                "python3 train_tpu.py --data_csv ./data/merged_normalized.csv\'\"\n"
                "  https://cloud.google.com/tpu/docs/pytorch-pods\n\n"
                f"Original error: {exc}"
            )
        raise
    print("XLA TPU devices:", devices)


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="TPU transliteration training (mT5).")
    p.add_argument(
        "--data_csv",
        required=True,
        help="Path to merged_normalized.csv (romanized_text, amharic_text).",
    )
    p.add_argument(
        "--output_dir",
        default=os.path.join(here, "outputs", "checkpoints"),
        help="Local checkpoint dir, or gs://... if --gcs_bucket is set (see resolve logic).",
    )
    p.add_argument(
        "--gcs_bucket",
        default=None,
        help="If set, checkpoints go under gs://BUCKET/models/... (same layout as notebook).",
    )
    p.add_argument("--gcs_location", default="US")
    p.add_argument("--model_name", default="google/mt5-small")
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_input_length", type=int, default=128)
    p.add_argument("--max_target_length", type=int, default=128)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument(
        "--subset_rows",
        type=int,
        default=30_000,
        help="If >0, train on first N rows only (smoke test). 0 = full dataset.",
    )
    p.add_argument("--batch_size", type=int, default=1, help="Per TPU core.")
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_checkpoint_saves", type=int, default=15)
    p.add_argument("--no_resume", action="store_true")
    p.add_argument(
        "--final_model_dir",
        default=None,
        help="Where to save final model (default: <here>/outputs/final_model).",
    )
    p.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Lower memory; slower. Off by default on TPU for mt5-small + micro-batch.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _print_xla_tpu_devices()
    here = os.path.dirname(os.path.abspath(__file__))
    final_dir = args.final_model_dir or os.path.join(here, "outputs", "final_model")

    if not os.path.isfile(args.data_csv):
        sys.exit(f"CSV not found: {args.data_csv}")

    if args.gcs_bucket:
        output_dir = resolve_gcs_output_dir(args.gcs_bucket, args.gcs_location)
    else:
        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"OUTPUT_DIR (local): {output_dir}")

    print("Loading dataset...")
    df = pd.read_csv(args.data_csv)
    if args.subset_rows and args.subset_rows > 0:
        df = df.head(min(args.subset_rows, len(df))).copy()
        print(f"Subset: {len(df)} rows")

    required = {"romanized_text", "amharic_text"}
    if not required.issubset(df.columns):
        sys.exit(f"CSV must contain columns {required}")

    df = df.dropna(subset=["romanized_text", "amharic_text"])
    df["romanized_text"] = df["romanized_text"].astype(str)
    df["amharic_text"] = df["amharic_text"].astype(str)
    df = df[
        (df["romanized_text"].str.strip() != "")
        & (df["amharic_text"].str.strip() != "")
    ]
    df["source"] = df["romanized_text"]
    df["target"] = df["amharic_text"]
    train_ready = df[["source", "target"]].copy()
    print(f"Rows after cleaning: {len(train_ready)}")

    train_df, test_df = train_test_split(
        train_ready,
        test_size=args.test_size,
        random_state=args.seed,
    )
    raw_datasets = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "test": Dataset.from_pandas(test_df),
        }
    )

    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    else:
        model.config.use_cache = True

    max_in = args.max_input_length
    max_tgt = args.max_target_length

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["source"],
            max_length=max_in,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=examples["target"],
            max_length=max_tgt,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["length"] = [len(ids) for ids in model_inputs["input_ids"]]
        return model_inputs

    print("Tokenizing...")
    tokenized = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=0,
        remove_columns=raw_datasets["train"].column_names,
    )

    bleu_metric = evaluate.load("sacrebleu")
    compute_metrics = build_compute_metrics(tokenizer, bleu_metric)

    _base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=None,
    )

    def data_collator(features):
        stripped = [{k: v for k, v in row.items() if k != "length"} for row in features]
        return _base_collator(stripped)

    n_train = len(tokenized["train"])
    steps_per_epoch = max(1, math.ceil(n_train / (args.batch_size * args.grad_accum)))
    total_steps = max(1, steps_per_epoch * args.num_epochs)
    base = max(1, math.ceil(total_steps / args.num_checkpoint_saves))
    eval_steps = max(1, base // 2)
    save_steps = max(1, base * 2)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_accumulation_steps=1,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_epochs,
        predict_with_generate=False,
        generation_max_length=max_tgt,
        logging_steps=20,
        logging_nan_inf_filter=False,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=args.num_checkpoint_saves + 2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_bleu",
        greater_is_better=True,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        group_by_length=True,
        length_column_name="length",
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    resume = not args.no_resume
    if not str(output_dir).startswith("gs://"):
        os.makedirs(output_dir, exist_ok=True)
    last_ckpt = get_last_checkpoint(output_dir)
    if resume and last_ckpt is None:
        print("No checkpoint-* under OUTPUT_DIR — starting from scratch.")
    resume_arg = last_ckpt if (resume and last_ckpt) else None
    print(f"resume_from_checkpoint={resume_arg!r}")

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_arg)

    os.makedirs(final_dir, exist_ok=True)
    print(f"Saving final model to {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Done.")


if __name__ == "__main__":
    main()
