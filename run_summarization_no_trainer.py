#!/usr/bin/env python
# coding=utf-8
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import random
import sys

import datasets
import nltk
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset, load_metric
from filelock import FileLock
from torch.nn import MSELoss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.utils.versions import require_version

import wandb
from quant.configuration_bart_quant import BartConfig as QBartConfig
from quant.modeling_bart_quant import BartForConditionalGeneration as QBart

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}

distill_mappings = {1: {0: 5},
                    2: {0: 0, 1: 5},
                    3: {0: 0, 1: 2, 2: 5},
                    4: {0: 0, 1: 2, 2: 3, 3: 5},
                    5: {0: 0, 1: 1, 2: 3, 3: 4, 4: 5},
                    6: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
                    }
distill_mappings_new = {1: {0: 0}}
NUMS = [str(i) for i in range(6)]


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument("--dataset_name",
                        type=str,
                        default=None,
                        help="The name of the dataset to use (via the datasets library).", )
    parser.add_argument("--dataset_config_name",
                        type=str,
                        default=None,
                        help="The configuration name of the dataset to use (via the datasets library).", )
    parser.add_argument("--train_file",
                        type=str,
                        default=None,
                        help="A csv or a json file containing the training data.")
    parser.add_argument("--validation_file",
                        type=str,
                        default=None,
                        help="A csv or a json file containing the validation data.")
    parser.add_argument("--ignore_pad_token_for_loss",
                        type=bool,
                        default=True,
                        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.", )
    parser.add_argument("--max_source_length",
                        type=int,
                        default=1024,
                        help="The maximum total input sequence length after "
                             "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--source_prefix",
                        type=str,
                        default=None,
                        help="A prefix to add before every source text " "(useful for T5 models).", )
    parser.add_argument("--preprocessing_num_workers",
                        type=int,
                        default=None,
                        help="The number of processes to use for the preprocessing.", )
    parser.add_argument("--overwrite_cache",
                        type=bool,
                        default=None,
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--max_target_length",
                        type=int,
                        default=128,
                        help="The maximum total sequence length for target text after "
                             "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
                             "during ``evaluate`` and ``predict``.", )
    parser.add_argument("--val_max_target_length",
                        type=int,
                        default=None,
                        help="The maximum total sequence length for validation "
                             "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
                             "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
                             "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.", )
    parser.add_argument("--pad_to_max_length",
                        action="store_true",
                        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.", )
    parser.add_argument("--model_name_or_path",
                        type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models.",
                        required=True, )
    parser.add_argument("--config_name",
                        type=str,
                        default=None,
                        help="Pretrained config name or path if not the same as model_name", )
    parser.add_argument("--tokenizer_name",
                        type=str,
                        default=None,
                        help="Pretrained tokenizer name or path if not the same as model_name", )
    parser.add_argument("--text_column",
                        type=str,
                        default=None,
                        help="The name of the column in the datasets containing the full texts (for summarization).", )
    parser.add_argument("--summary_column",
                        type=str,
                        default=None,
                        help="The name of the column in the datasets containing the summaries (for summarization).", )
    parser.add_argument("--use_slow_tokenizer",
                        action="store_true",
                        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).", )
    parser.add_argument("--per_device_train_batch_size",
                        type=int,
                        default=8,
                        help="Batch size (per device) for the training dataloader.", )
    parser.add_argument("--per_device_eval_batch_size",
                        type=int,
                        default=4,
                        help="Batch size (per device) for the evaluation dataloader.", )
    parser.add_argument("--learning_rate",
                        type=float,
                        default=3e-5,
                        help="Initial learning rate (after the potential warmup period) to use.", )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps",
                        type=int,
                        default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.", )
    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--lr_scheduler_type",
                        type=SchedulerType,
                        default="linear",
                        help="The scheduler type to use.",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                 "constant_with_warmup"], )
    parser.add_argument("--warmup_ratio",
                        type=float,
                        default=0.05,
                        help="warmup in the lr scheduler.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=28, help="A seed for reproducible training.")
    parser.add_argument("--model_type",
                        type=str,
                        default=None,
                        help="Model type to use if training from scratch.",
                        choices=MODEL_TYPES, )

    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        help="The models directory.")
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        help="The models directory.")
    parser.add_argument('--pred_distill',
                        action='store_true',
                        help="Whether to distil with task layer")
    parser.add_argument('--intermediate_distill',
                        action='store_true',
                        help="Whether to distil with intermediate layers")

    parser.add_argument("--weight_bits",
                        default=8,
                        type=int,
                        choices=[2, 8, 16],
                        help="Quantization bits for weight.")
    parser.add_argument("--input_bits",
                        default=8,
                        type=int,
                        help="Quantization bits for activation.")
    parser.add_argument("--clip_val",
                        default=2.5,
                        type=float,
                        help="Initial clip value.")
    parser.add_argument("--length_penalty",
                        default=1.0,
                        type=float,
                        help="model config param lengthy_penalty.")
    parser.add_argument("--max_length",
                        default=128,
                        type=int,
                        help=(
                            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"" sequences shorter will be padded if `--pad_to_max_lengh` is passed."),
                        )
    parser.add_argument("--min_length",
                        default=12,
                        type=int,
                        help="model config param min_length.")
    parser.add_argument("--num_beams",
                        default=4,
                        type=int,
                        help="Number of beams to use for evaluation. This argument will be "
                             "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.", )

    parser.add_argument('--do_train',
                        action='store_true',
                        help="Whether to do train and evaluation")
    parser.add_argument('--do_test',
                        action='store_true',
                        help="Whether to do test")
    parser.add_argument('--test_teacher',
                        action='store_true',
                        help="Whether to test teacher")
    parser.add_argument('--distill_encoder',
                        default=6,
                        type=int,
                        help="Number of encoder layers after distillation")
    parser.add_argument('--distill_decoder',
                        default=6,
                        type=int,
                        help="Number of decoder layers after distillation")
    parser.add_argument('--log_steps', default=20)
    parser.add_argument('--local_rank', default=0)
    parser.add_argument('--weighted', action='store_true')
    parser.add_argument('--new_distill_map', action='store_true')

    args = parser.parse_args()

    # Sanity checks
    if args.new_distill_map:
        assert args.distill_decoder == 1
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    args.output_dir = f'./output_{args.dataset_name}/{args.weight_bits}_{args.input_bits}_{args.distill_encoder}_{args.distill_decoder}_{args.num_train_epochs}_{args.learning_rate}_fp16'
    if args.new_distill_map:
        args.output_dir += '_new'
    if (not args.pred_distill) and (not args.intermediate_distill):
        args.output_dir += '_nodis'

    if args.student_model is None:
        args.student_model = args.model_name_or_path
    if args.teacher_model is None:
        args.teacher_model = args.model_name_or_path

    if args.dataset_name == "xsum":
        args.length_penalty = 1.0
        args.max_length = 62
        args.min_length = 11
        args.num_beams = 6
    elif args.dataset_name == "cnn_dailymail":
        args.length_penalty = 2.0
        args.max_length = 142
        args.min_length = 56
        args.num_beams = 4
    else:
        assert False, f'args error: dataset name {args.dataset_name}'
    if args.weighted:
        args.task_weight = 1
        args.logits_weight = 0.8
        args.hid_weight = 3
        args.output_dir += '_weighted'
    else:
        args.task_weight = 1
        args.logits_weight = 1
        args.hid_weight = 1
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    return args


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output_dir, "training.log")),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    is_master = accelerator.is_local_main_process
    if is_master:
        logger.info(accelerator.state)
        logger.warning(args)
        task, run = args.output_dir.split('/')[1:]
        wandb.init(project=task, name=run, config=args)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    distill_enc_mapping = distill_mappings[args.distill_encoder]
    distill_dec_mapping = distill_mappings[args.distill_decoder] if not args.new_distill_map else distill_mappings_new[
        args.distill_decoder]
    maps = {'enc': distill_enc_mapping, 'dec': distill_dec_mapping}

    if args.model_name_or_path:
        teacher_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )

        student_config = QBartConfig.from_pretrained(args.teacher_model,
                                                     quantize_act=True,
                                                     weight_bits=args.weight_bits,
                                                     input_bits=args.input_bits,
                                                     clip_val=args.clip_val,
                                                     decoder_layers=args.distill_decoder,
                                                     encoder_layers=args.distill_encoder)
        student_model = QBart(student_config)

        dst_dict = student_model.state_dict()  # Initilized student model state dict, needs loading weights
        src_dict = teacher_model.state_dict()  # Pretrained teacher model state dict, whose weights will be loaded

        for key in dst_dict.keys():
            if ("encoder" in key or "decoder" in key) and key[
                21] in NUMS:  # Determine if the key belongs to a encoder/decoder layer,
                # which starts with sth like model.decoder.layers.1

                m = maps[key[6:9]]  # Determin if it is an encoder or decoder, and get the layer mapping
                old_idx = int(key[21])  # The layer index of the student model that needs loading
                new_idx = str(m[old_idx])  # The layer index of the teacher model that should be loaded
                mapped_key = key[:21] + new_idx + key[22:]  # Get the full teacher layer key
                if mapped_key in src_dict.keys():  # Exclude the cases
                    # which does not exist in the teacher model
                    dst_dict[key] = src_dict[mapped_key]  # Load the weights of the layer
            else:
                if key in src_dict.keys():  # Load the weights of non-encoder/decoder layers
                    dst_dict[key] = src_dict[key]

        student_model.load_state_dict(dst_dict, strict=False)  # Pass the dict to the student model

    else:
        raise ValueError(
            "You did not provide a pre-trained teacher_model."
        )

    teacher_model.resize_token_embeddings(len(tokenizer))
    student_model.resize_token_embeddings(len(tokenizer))

    if teacher_model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(args.dataset_name, None)
    if args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
        num_proc=10
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=teacher_model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.    
    student_model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        student_model, optimizer, train_dataloader, eval_dataloader, test_dataloader
    )
    teacher_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * args.max_train_steps),
        num_training_steps=args.max_train_steps,
    )

    # Metric
    metric = load_metric("rouge")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # print(distill_enc_mapping, distill_dec_mapping)

    logger.info(f"  student encoder layers = {student_config.encoder_layers}")
    logger.info(f"  student decoder layers = {student_config.decoder_layers}")
    logger.info(
        f"  student encoder layers {list(distill_enc_mapping.keys())} is mapped with teacher encoder layers {list(distill_enc_mapping.values())}")
    logger.info(
        f"  student decoder layers {list(distill_dec_mapping.keys())} is mapped with teacher decoder layers {list(distill_dec_mapping.values())}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, mininterval=2)
    completed_steps = 0
    loss_mse = MSELoss()
    assert teacher_model.training == False
    prev = 0.0

    gen_kwargs = {
        "length_penalty": args.length_penalty,
        "max_length": args.max_length,
        "min_length": args.min_length,
        "num_beams": args.num_beams,
    }

    if args.do_train:
        for epoch in range(args.num_train_epochs):
            student_model.train()

            log_total_loss = 0.0
            log_task_loss = 0.0
            log_logits_loss = 0.0
            log_enc_att_loss = 0.0
            log_dec_att_loss = 0.0
            log_crs_att_loss = 0.0
            log_enc_hid_loss = 0.0
            log_enc_hid_last_loss = 0.0
            log_dec_hid_loss = 0.0

            for step, batch in enumerate(train_dataloader):
                task_loss = 0.0
                logits_loss = 0.0
                enc_att_loss = 0.0
                dec_att_loss = 0.0
                crs_att_loss = 0.0
                enc_hid_loss = 0.0
                enc_hid_last_loss = 0.0
                dec_hid_loss = 0.0

                student_outputs = student_model(**batch, output_attentions=True, output_hidden_states=True)
                task_loss = student_outputs.loss

                with torch.no_grad():
                    teacher_outputs = teacher_model(**batch, output_attentions=True, output_hidden_states=True)
                if args.pred_distill:
                    logits_loss = loss_mse(student_outputs.logits, teacher_outputs.logits)

                if args.intermediate_distill:
                    for i, student_att in enumerate(student_outputs.encoder_attentions):
                        mapped_idx = distill_enc_mapping[i]
                        teacher_att = teacher_outputs.encoder_attentions[mapped_idx]
                        enc_att_loss += loss_mse(student_att, teacher_att)

                    for student_hs, teacher_hs in zip(student_outputs.encoder_last_hidden_state,
                                                      teacher_outputs.encoder_last_hidden_state):
                        enc_hid_last_loss += loss_mse(student_hs, teacher_hs)

                    for i, student_hs in enumerate(student_outputs.encoder_hidden_states):
                        if i == 0:
                            mapped_idx = 0
                        else:
                            mapped_idx = distill_enc_mapping[i - 1] + 1

                        teacher_hs = teacher_outputs.encoder_hidden_states[mapped_idx]
                        enc_hid_loss += loss_mse(student_hs, teacher_hs)

                    for i, student_att in enumerate(student_outputs.cross_attentions):
                        mapped_idx = distill_dec_mapping[i]
                        teacher_att = teacher_outputs.cross_attentions[mapped_idx]
                        crs_att_loss += loss_mse(student_att, teacher_att)

                    for i, student_att in enumerate(student_outputs.decoder_attentions):
                        mapped_idx = distill_dec_mapping[i]
                        teacher_att = teacher_outputs.decoder_attentions[mapped_idx]
                        dec_att_loss += loss_mse(student_att, teacher_att)

                    for i, student_hs in enumerate(student_outputs.decoder_hidden_states):
                        if i == 0:
                            mapped_idx = 0
                        else:
                            mapped_idx = distill_dec_mapping[i - 1] + 1

                        teacher_hs = teacher_outputs.decoder_hidden_states[mapped_idx]
                        dec_hid_loss += loss_mse(student_hs, teacher_hs)

                total_loss = args.task_weight * task_loss + \
                             args.logits_weight * logits_loss + \
                             args.hid_weight * (
                                         enc_att_loss + dec_att_loss + crs_att_loss + enc_hid_loss + enc_hid_last_loss + dec_hid_loss)

                accelerator.backward(total_loss / args.gradient_accumulation_steps)

                log_total_loss += total_loss.item()
                log_task_loss += task_loss.item()
                if args.pred_distill:
                    log_logits_loss += logits_loss.item()
                if args.intermediate_distill:
                    log_enc_att_loss += enc_att_loss.item()
                    log_dec_att_loss += dec_att_loss.item()
                    log_crs_att_loss += crs_att_loss.item()
                    log_enc_hid_loss += enc_hid_loss.item()
                    log_enc_hid_last_loss += enc_hid_last_loss.item()
                    log_dec_hid_loss += dec_hid_loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:

                    optimizer.step()
                    lr_scheduler.step()

                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                    cur_step = (epoch * len(train_dataloader) + step) // args.gradient_accumulation_steps
                    log_accu_steps = args.log_steps * args.gradient_accumulation_steps
                    if (step + 1) % log_accu_steps == 0 and is_master:
                        wandb.log({'train/lr': lr_scheduler.get_last_lr()[0], 'train/step': cur_step,
                                   'train/loss': log_total_loss / log_accu_steps,
                                   'train/task_loss': log_task_loss / log_accu_steps,
                                   'train/logits_loss': log_logits_loss / log_accu_steps,
                                   "train/enc_att_loss": log_enc_att_loss / log_accu_steps,
                                   "train/dec_att_loss": log_dec_att_loss / log_accu_steps,
                                   "train/crs_att_loss": log_crs_att_loss / args.gradient_accumulation_steps,
                                   "train/enc_hid_loss": log_enc_hid_loss / log_accu_steps,
                                   "train/enc_hid_last_loss": log_enc_hid_last_loss / log_accu_steps,
                                   "train/dec_hid_loss": log_dec_hid_loss / log_accu_steps})

                        log_total_loss = 0.0
                        log_task_loss = 0.0
                        log_logits_loss = 0.0
                        log_enc_att_loss = 0.0
                        log_dec_att_loss = 0.0
                        log_crs_att_loss = 0.0
                        log_enc_hid_loss = 0.0
                        log_enc_hid_last_loss = 0.0
                        log_dec_hid_loss = 0.0

                if completed_steps >= args.max_train_steps:
                    break

            for step, batch in enumerate(tqdm(eval_dataloader)):
                with torch.no_grad():
                    generated_tokens = accelerator.unwrap_model(student_model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        **gen_kwargs,
                    )

                    generated_tokens = accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                    )
                    labels = batch["labels"]
                    if not args.pad_to_max_length:
                        # If we did not pad to max length, we need to pad the labels too
                        labels = accelerator.pad_across_processes(batch["labels"], dim=1,
                                                                  pad_index=tokenizer.pad_token_id)

                    generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                    labels = accelerator.gather(labels).cpu().numpy()

                    if args.ignore_pad_token_for_loss:
                        # Replace -100 in the labels as we can't decode them.
                        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                    if isinstance(generated_tokens, tuple):
                        generated_tokens = generated_tokens[0]
                    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                    metric.add_batch(predictions=decoded_preds, references=decoded_labels)

            result = metric.compute(use_stemmer=True)
            # Extract a few results from ROUGE
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
            result = {'eval/' + k: round(v, 4) for k, v in result.items()}
            if is_master:
                wandb.log(result)
            res_rougeL = result['eval/rougeLsum']

            logger.info(f"evaluation result: {result} ")

            if args.output_dir is not None and res_rougeL > prev:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(student_model)
                unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                prev = res_rougeL

                # load best model and evaluate on testset
    if args.do_test:
        try:
            student_model.to('cpu')
            teacher_model.to('cpu')
            del student_model
            del teacher_model
            del student_outputs
            del teacher_outputs
            for i in batch:
                del i
        except Exception as e:
            logger.warning(f'Error in deletion: {e}')
        if not args.test_teacher:
            best_model_config = QBartConfig.from_pretrained(args.output_dir,
                                                            quantize_act=True,
                                                            weight_bits=args.weight_bits,
                                                            input_bits=args.input_bits,
                                                            clip_val=args.clip_val,
                                                            decoder_layers=args.distill_decoder,
                                                            encoder_layers=args.distill_encoder)
            best_model = QBart(best_model_config)
            best_model.load_state_dict(
                torch.load(os.path.join(args.output_dir + "/", "pytorch_model.bin"), map_location='cpu'))

        if args.test_teacher:
            best_model = teacher_model
            logger.info(f"testing teacher model from {args.teacher_model} ")

        best_model = accelerator.prepare(best_model)
        best_model.eval()

        for step, batch in enumerate(tqdm(test_dataloader)):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(best_model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]
                if not args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        result = metric.compute(use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result = {'test/' + k: round(v, 4) for k, v in result.items()}
        if is_master:
            wandb.log(result)
        logger.info(f"test result: {result}")


if __name__ == "__main__":
    main()
