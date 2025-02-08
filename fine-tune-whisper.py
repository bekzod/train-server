#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import torch
import re
import datetime
import logging
from transformers import (
    pipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperFeatureExtractor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
# ------------------- CHANGED: We now import wandb and WandbCallback instead of TensorBoard --------------------
import wandb
from transformers.integrations import WandbCallback
# -------------------------------------------------------------------------------------------------------------

from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets, Audio
import librosa
import soundfile as sf
import evaluate
import getpass
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from peft import LoraConfig, get_peft_model

# Configure logging to file and console
logger = logging.getLogger("transformers")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("training.log")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info("Starting training script...")

model_name_or_path = "openai/whisper-large-v3"
language = "Uzbek"
language_abbr = "uz"
task = "transcribe"
org = "bekzod123"
trained_adapter_name = "whisper-turbo-llm-lingo-adapters"
trained_model_name = "whisper-turbo-llm-lingo"
trained_adapter_repo = org + "/" + trained_adapter_name
trained_model_repo = org + '/' + trained_model_name

logger.info("Stage: Loading and preparing additional datasets")

def load_and_prepare_datasets(datasets_info):
    combined_dataset = DatasetDict()
    for dataset_info in datasets_info:
        dataset_name = dataset_info["name"]
        audio_col = dataset_info["audio_col"]
        text_col = dataset_info["text_col"]
        subset = dataset_info.get("subset")
        revision = dataset_info.get("revision", "main")
        limit = dataset_info.get("limit")  # optional
        custom_filter = dataset_info.get("filter_fn")  # optional filter function

        print(f"Loading dataset: {dataset_name} | subset: {subset} | revision: {revision}")

        # Load dataset
        if subset:
            loaded_dataset = load_dataset(
                dataset_name,
                subset,
                revision=revision,
                trust_remote_code=True
            )
        else:
            loaded_dataset = load_dataset(
                dataset_name,
                revision=revision,
                trust_remote_code=True
            )

        # Identify train / test splits
        if isinstance(loaded_dataset, Dataset):
            ds_train = loaded_dataset
            ds_test = None
        else:
            ds_train = loaded_dataset.get("train")
            ds_test = loaded_dataset.get("validate")
            if loaded_dataset.get("validation"):
                ds_train = concatenate_datasets([ds_train, loaded_dataset["validation"]])
            if loaded_dataset.get("other"):
                ds_train = concatenate_datasets([ds_train, loaded_dataset["other"]])
            if ds_test and loaded_dataset.get("test"):
                ds_test = concatenate_datasets([ds_test, loaded_dataset["test"]])
            else:
                ds_test = loaded_dataset.get("test") or ds_test

        if ds_train is None:
            available_splits = list(loaded_dataset.keys())
            if len(available_splits) == 1:
                single_split_name = available_splits[0]
                ds_train = loaded_dataset[single_split_name]
                ds_test = None
            else:
                raise ValueError(
                    f"No 'train' split found for {dataset_name}. Available splits: {available_splits}"
                )

        if ds_test is None:
            split_result = ds_train.train_test_split(test_size=0.1, seed=42)
            ds_train = split_result["train"]
            ds_test = split_result["test"]

        # Apply custom filter if provided
        if custom_filter is not None:
            ds_train = ds_train.filter(custom_filter)
            ds_test = ds_test.filter(custom_filter)

        # Rename columns to standard 'audio' and 'text'
        rename_map = {}
        if audio_col in ds_train.column_names:
            rename_map[audio_col] = "audio"
        if text_col in ds_train.column_names:
            if text_col != "text":
                if "text" in ds_train.column_names:
                    ds_train = ds_train.remove_columns(["text"])
                if "text" in ds_test.column_names:
                    ds_test = ds_test.remove_columns(["text"])
            rename_map[text_col] = "text"

        ds_train = ds_train.rename_columns(rename_map)
        ds_test = ds_test.rename_columns(rename_map)

        # Keep only "audio" and "text" columns
        columns_to_keep = ["audio", "text"]
        ds_train = ds_train.remove_columns([col for col in ds_train.column_names if col not in columns_to_keep])
        ds_test = ds_test.remove_columns([col for col in ds_test.column_names if col not in columns_to_keep])

        # Cast "audio" column to Audio feature with a sampling rate of 16000
        if "audio" in ds_train.column_names:
            ds_train = ds_train.cast_column("audio", Audio(sampling_rate=16000))
        if "audio" in ds_test.column_names:
            ds_test = ds_test.cast_column("audio", Audio(sampling_rate=16000))

        # NOTE: The previous version applied a map() to trim the text here.
        # We now handle text trimming inside prepare_dataset so we only call map() once later.

        # Apply limit if specified
        if limit is not None:
            ds_train = ds_train.select(range(min(limit, len(ds_train))))
            ds_test = ds_test.select(range(min(int(limit * 0.2), len(ds_test))))

        # Concatenate into combined dataset
        if "train" not in combined_dataset:
            combined_dataset["train"] = ds_train
        else:
            combined_dataset["train"] = concatenate_datasets([combined_dataset["train"], ds_train])

        if "validation" not in combined_dataset:
            combined_dataset["validation"] = ds_test
        else:
            combined_dataset["validation"] = concatenate_datasets([combined_dataset["validation"], ds_test])

    return combined_dataset

datasets_info = [
    {
        "name": "DavronSherbaev/uzbekvoice-filtered",
        "audio_col": "path",
        "text_col": "sentence",
        "filter_fn": lambda ex: (
            ex.get("reported_reasons") is None and
            ex.get("downvotes_count", 0) == 0 and
            ex.get("reported_count", 0) == 0 and
            ex.get("client_id") not in [
                "56ac8e86-b8c9-4879-a342-0eeb94f686fc",
                "3d3fca02-6a07-41e2-9af4-60886ea60300",
                "231d3776-2dbe-4a42-a535-c67943427e3f",
                "e2716f95-70b5-4832-b903-eef2343591a4",
                "2a815774-e953-4031-931a-8a28052e5cf9",
                "d6fd3dc4-a55d-4a80-9bbf-b713325d05be",
                "10b29e87-bf01-4b16-bead-a044076f849b",
                "e3412d51-f079-4167-b3f9-311a976443ce"
            ]
        )
    },
    {
        "name": "Beehzod/dataset_for_STT_TTSmodels",
        "audio_col": "audio",
        "text_col": "transcription",
        "revision": "refs/convert/parquet"
    },
    {
        "name": "google/fleurs",
        "subset": "uz_uz",
        "audio_col": "audio",
        "text_col": "transcription",
    },
    {
        "name": "bekzod123/uzbek_voice",
        "audio_col": "audio",
        "text_col": "text",
        "revision": "refs/convert/parquet",
        "filter_fn": lambda ex: (ex.get("is_correct") == True)
    },
    {
        "name": "bekzod123/uzbek_voice_2",
        "audio_col": "audio",
        "text_col": "sentence"
    },
    # {
    #     "name": "mozilla-foundation/common_voice_17_0",
    #     "subset": "uz",
    #     "audio_col": "audio",
    #     "text_col": "sentence",
    # },
]

dataset = load_and_prepare_datasets(datasets_info)
logger.info("Combined dataset loaded.")

logger.info("Stage: Preparing feature extractor, tokenizer and processor")
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
tokenizer.pad_token = tokenizer.eos_token
processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)

logger.info("Sample training data: %s", dataset["train"][0])

def prepare_dataset(batch):
    # Instead of a separate map call for text trimming, trim the text here.
    # Handle both single string or list of strings.
    if isinstance(batch["text"], str):
        batch["text"] = batch["text"].strip()
    else:
        batch["text"] = [txt.strip() for txt in batch["text"]]

    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch

logger.info("Stage: Mapping dataset for training")
# A single map() call now prepares both the features and the trimmed text.
dataset = dataset.map(
    prepare_dataset,
    remove_columns=dataset.column_names["train"],
    num_proc=1
)
logger.info("Processed training and validation datasets.")

logger.info("Stage: Setting up data collator and evaluation metrics")
use_bf16 = True

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_dtype = torch.bfloat16 if use_bf16 else torch.float16
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        batch = {k: v.to(input_dtype) for k, v in batch.items()}

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove BOS token if all labels start with it
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer_val = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer_val}

logger.info("Stage: Loading pre-trained Whisper model")
model = WhisperForConditionalGeneration.from_pretrained(
    model_name_or_path,
    load_in_8bit=False,
    torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.pad_token_id

logger.info("Model dtype: %s", model.dtype)
model.config.forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(
    language=language, task=task
)
model.config.suppress_tokens = []
model.generation_config.forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(
    language=language, task=task
)
model.generation_config.suppress_tokens = []

logger.info("Stage: Applying LoRA")
config = LoraConfig(
    r=32,
    lora_alpha=8,
    use_rslora=True,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
    modules_to_save=["model.embed_tokens"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

logger.info("Stage: Setting training configuration")
training_args = Seq2SeqTrainingArguments(
    output_dir=trained_model_name,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    weight_decay=0,
    warmup_ratio=0.1,
    num_train_epochs=2,
    optim="adamw_torch",
    eval_strategy="steps",
    fp16=not use_bf16,
    bf16=use_bf16,
    generation_max_length=128,
    save_steps=800,
    eval_steps=400,
    save_total_limit=5,
    logging_steps=200,
    remove_unused_columns=False,
    label_names=["labels"],
    predict_with_generate=True,
    do_eval=True,
    lr_scheduler_type="constant",
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    metric_for_best_model="wer",
    greater_is_better=False,
    report_to="wandb",
    run_name="my-whisper-run",
)

# Optionally configure W&B environment before trainer:
wandb.init(project="my-whisper-project")

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=5),
        WandbCallback(),
    ],
)
model.config.use_cache = False

logger.info("Stage: Starting training")
trainer.train()

# Push the LoRA adapter to the hub with error handling
today = datetime.date.today().strftime("%Y-%m-%d")
adapter_to_push = f"{trained_adapter_repo}-{today}"

try:
    logger.info("Pushing adapter to the hub: %s", adapter_to_push)
    model.push_to_hub(adapter_to_push, private=True)
    logger.info("Successfully pushed adapter to the hub.")
except Exception as e:
    logger.error("Failed to push adapter to the hub: %s", str(e))

# Save final model, tokenizer, and processor locally
final_dir = trained_model_name + "_final"
try:
    logger.info("Saving final model locally in directory: %s", final_dir)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    logger.info("Successfully saved final model locally.")
except Exception as e:
    logger.error("Failed to save the final model locally: %s", str(e))

# Reload base model, load adapter, merge and push the final merged model
try:
    logger.info("Reloading base model for adapter merging.")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path,
        load_in_8bit=False,
        device_map="auto"
    )
    # Load the adapter from the hub
    merged_model = get_peft_model(base_model, config)  # Ensure the model structure is consistent
    merged_model = merged_model.from_pretrained(merged_model, adapter_to_push)
    merged_model = merged_model.merge_and_unload()
    logger.info("Merged model dtype after merging: %s", merged_model.dtype)
except Exception as e:
    logger.error("Failed during adapter merge: %s", str(e))
    merged_model = None

if merged_model is not None:
    try:
        logger.info("Pushing the final merged model to the hub: %s", trained_model_repo)
        merged_model.push_to_hub(trained_model_repo, safe_serialization=True)
        processor.push_to_hub(trained_model_repo, private=True)
        logger.info("Successfully pushed the final merged model and processor to the hub.")
    except Exception as e:
        logger.error("Failed to push the final merged model to the hub: %s", str(e))
