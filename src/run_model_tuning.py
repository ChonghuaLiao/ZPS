#!/usr/bin/env python
# coding=utf-8
# Copyright BigScience, The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
Reproduce the main evaluation in `Multitask Prompted Training Enables Zero-Shot Task Generalization` using PyTorch.

This script is heavily adapted from https://github.com/huggingface/transformers/blob/7533d30acd975027e83a548e4c38e06fa335291b/examples/pytorch/multiple-choice/run_swag_no_trainer.py
"""

import argparse
import logging
import math
import os
import random
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
import json
import codecs
import datetime

import datasets
import torch
from datasets import load_dataset, load_metric, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import Counter
import numpy as np

import transformers
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    default_data_collator,
)
from transformers.file_utils import PaddingStrategy
from transformers import Adafactor

# from promptsource.templates_test import DatasetTemplates
from templates import DatasetTemplates
from torch.nn import CrossEntropyLoss
from torch import nn

logger = logging.getLogger(__name__)


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

def read_split_list(file_name):
    test_task_list = []
    with codecs.open(file_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.replace('\n', '')
            task_tuple = line.split('/')
            if len(task_tuple) == 2:
                test_task_list.append(task_tuple)
            else:
                test_task_list.append((task_tuple[0], None))

    return test_task_list


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce main evaluation in T0.")
    parser.add_argument("--max_length", type=int, default=1024,
                        help=("The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
                              " sequences shorter will be padded if `--pad_to_max_lengh` is passed."), )
    parser.add_argument("--target_max_length", type=int, default=256,
                        help="Target max length. Sequences longer than this will be truncated." )
    parser.add_argument("--pad_to_max_length", action="store_true",
                        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.", )
    parser.add_argument("--model_name_or_path", type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models. The list of T0 variants can be found on `https://huggingface.co/bigscience/T0_3B`",
                        required=True, )
    parser.add_argument("--config_name", type=str, default=None, help="Pretrained config name or path if not the same as model_name", )
    parser.add_argument("--template_dir", type=str, default='../templates_test', help="模版文件的位置", )
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name", )
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).", )
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Batch size (per device) for the evaluation dataloader.", )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--debug", action="store_true",
                        help="Activate debug mode and run training only with a subset of data.", )
    parser.add_argument("--ga_dev_distribution", type=str, choices=['uniform', 'ratio'], default='uniform', help="ga_dev的分布", )
    parser.add_argument("--parallelize", action="store_true",
                        help=(
                            "If passed, will call `model.parallelize` which splits the model on all GPUs available when applicable (model parallelism). "
                            "Note that this feature is still experimental in HF Transformers."),
                        )
    parser.add_argument("--test_split", type=str, help='测试任务名单')
    parser.add_argument("--dataset_type", type=str, choices=['ga', 'pt', 'all'])

    # prompt tuning 的参数
    parser.add_argument("--lr", type=float, default=5e-5)   # PT 原文使用0.3
    parser.add_argument("--warmup_ratio", type=float, default=0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help='单卡的batch size')
    parser.add_argument("--num_training_steps", type=int, default=1000)
    parser.add_argument("--eval_period", type=int, default=50, help='训练时eval的时间间隔(用来挑ckpt)')
    # parser.add_argument("--optimization", type=str, default="adamw")
    # parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    parser.add_argument("--only_train_single_template", action="store_true")
    parser.add_argument("--only_eval_single_template", action="store_true")

    args = parser.parse_args()

    return args


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
            sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
            maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
            different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
            Note that it's very NOT recommended to use fp16 to do any time of inference with T0 as the predictions will vastly differ from the predictions using fp32.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    # 这里得到的应该是一个batch的feature
    def __call__(self, features):
        num_choices = len(features[0]["input_ids"])  # input_ids： exam_num * num_choice * seq_len
        # 输入之前是一个样本的多个选项一个dict，flatten后是[ [{exam1_opt1}, {exam1_opt2}], []  ]
        flattened_features = [
            [
                {
                    k: v[i]
                    for k, v in feature.items()
                    if k != "targets"
                }
                for i in range(num_choices)
            ]
            for feature in features
        ]
        # 把所有样本拍成一个list
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Pad the labels because it's not padded automatically
        max_label_length = max([len(elem["labels"]) for elem in flattened_features])   # 当前batch所有label的最大长度

        # padding 后的label
        batch["labels"] = [
            l + [self.tokenizer.pad_token_id] * (max_label_length - len(l))
            for l in [elem["labels"] for elem in flattened_features]
        ]
        batch["labels_attention_mask"] = [
            m + [0] * (max_label_length - len(m))
            for m in [elem["labels_attention_mask"] for elem in flattened_features]
        ]

        # Convert to tensors
        batch = {
            k: torch.tensor(v)
            for k, v in batch.items()
        }

        batch["targets"] = torch.tensor([f.pop("targets") for f in features])
        return batch


@dataclass
class DataCollatorForMultipleChoiceTraining:
    """
    用于训练数据
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    # 这里得到的应该是一个batch的feature
    def __call__(self, features):
        # num_choices = len(features[0]["input_ids"])  # input_ids： exam_num * num_choice * seq_len
        # 输入之前是一个样本的多个选项一个dict，flatten后是[ [{exam1_opt1}, {exam1_opt2}], []  ]
        # flattened_features = [
        #     [
        #         {
        #             k: v[i]
        #             for k, v in feature.items()
        #             if k != "targets"
        #         }
        #         for i in range(num_choices)
        #     ]
        #     for feature in features
        # ]
        # print(f'features: {features}')
        flattened_features = [
            {
                k: v
                for k, v in feature.items()
                if k != "targets"
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Pad the labels because it's not padded automatically
        max_label_length = max([len(elem["labels"]) for elem in flattened_features])  # 当前batch所有label的最大长度

        # padding 后的label
        batch["labels"] = [
            l + [self.tokenizer.pad_token_id] * (max_label_length - len(l))
            for l in [elem["labels"] for elem in flattened_features]
        ]
        # 屏蔽pad位
        for idx, label_ids in enumerate(batch["labels"]):
            label_ids = [ids if ids != 0 else -100 for ids in label_ids]
            batch["labels"][idx] = label_ids

        batch["labels_attention_mask"] = [
            m + [0] * (max_label_length - len(m))
            for m in [elem["labels_attention_mask"] for elem in flattened_features]
        ]

        # Convert to tensors
        batch = {
            k: torch.tensor(v)
            for k, v in batch.items()
        }

        batch["targets"] = torch.tensor([f.pop("targets") for f in features])
        return batch


def build_pt_dataset(dataset_name, dataset_config_name, raw_datasets, distribution='uniform'):
    task_name = f'{dataset_name}/{dataset_config_name}' if dataset_config_name else f'{dataset_name}'
    dataset_distribution = {'anli/r1': {0: 334, 2: 333, 1: 333},
                            'anli/r2': {0: 334, 1: 333, 2: 333},
                            'anli/r3': {0: 402, 1: 402, 2: 396},
                            'super_glue/cb': {1: 28, 0: 23, 2: 5},
                            'super_glue/rte': {0: 146, 1: 131},
                            'super_glue/wsc.fixed': {0: 66, 1: 38},
                            'winogrande/winogrande_xl': {'2': 639, '1': 628},
                            'super_glue/copa': {0: 55, 1: 45},
                            'hellaswag': {'2': 2584, '0': 2515, '1': 2485, '3': 2458},
                            'super_glue/wic': {0: 319, 1: 319}
                            }

    label_key = 'label'
    if dataset_name in ['winogrande']:
        label_key = 'answer'
    filtered_dataset = raw_datasets

    # 对anli数据集和winogrande的特别处理
    if dataset_name == 'anli':
        print(f'len of raw_dataset: {filtered_dataset}')
        filtered_dataset = filtered_dataset.filter(lambda x: len(x['reason']) > 0)
        print(f'len of filtered_dataset: {filtered_dataset}')
        if dataset_config_name == 'r1':
            filtered_dataset = filtered_dataset.select(range(900))
        elif dataset_config_name == 'r2':
            filtered_dataset = filtered_dataset.select(range(1500))
        elif dataset_config_name == 'r3':
            index = list(range(len(filtered_dataset)))
            index = index[8000:]
            filtered_dataset = filtered_dataset.select(index)
    if dataset_name == 'winogrande':
        filtered_dataset = load_dataset('winogrande', 'winogrande_debiased', split='train')

    label_list = filtered_dataset[label_key]
    label_type_set = set(label_list)
    print(f'label_type_set: {label_type_set}')
    ga_dataset_list = []
    # 把每个类别的样本分类存储
    for label_type in label_type_set:
        single_label_dataset = filtered_dataset.filter(lambda x: x[label_key] == label_type)
        single_label_dataset = single_label_dataset.shuffle(seed=42)

        if distribution == 'ratio':
            example_num_per_label = math.ceil(
                dataset_distribution[task_name][label_type] / sum(dataset_distribution[task_name].values()) * 32)
        else:
            example_num_per_label = math.ceil(32 / len(label_type_set))
        ga_dataset_list.append(single_label_dataset.select(
            range(min(example_num_per_label, len(single_label_dataset)))))

    # 每个class的样本train和eval各分一半
    train_dataset_list = [ga_dataset.select(range(len(ga_dataset) // 2)) for ga_dataset in ga_dataset_list]
    dev_dataset_list = [ga_dataset.select(range(len(ga_dataset)//2, len(ga_dataset))) for ga_dataset in ga_dataset_list]

    combined_train_dataset = concatenate_datasets(train_dataset_list)
    combined_dev_dataset = concatenate_datasets(dev_dataset_list)

    print(f'combined_train_dataset: {combined_train_dataset}')
    print(f'labels: {combined_train_dataset[label_key]}')
    print(f'combined_dev_dataset: {combined_dev_dataset}')
    print(f'labels: {combined_dev_dataset[label_key]}')
    return combined_train_dataset, combined_dev_dataset


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO, )
    logger.info(accelerator.state)

    test_task_list = read_split_list(args.test_split)
    logger.info(f'训练任务: {test_task_list}')

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Handle the output directory creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    elif args.model_name_or_path:
        # T0 model
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "Either `args.config_name` or `args.model_name_or_path` should be provided."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    if args.parallelize:
        assert torch.cuda.is_available(), "You need at least 1 GPU to call `parallelize` (even though if there is only 1 GPU, there won't be any model parallelism)."
        model.parallelize()
    else:
        model.to(device)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    dev_dataset_dict = {}
    train_dataset_list = []
    # 读数据集
    for dataset_name, dataset_config_name in test_task_list:
        # In distributed evaluation, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.
        if args.dataset_type == 'ga' or args.dataset_type == 'pt':
            # 注意anli和winogrand使用测试集，因为其dev/test是adversarial的
            if dataset_name == "anli":
                raw_datasets = load_dataset(dataset_name, split=f'train_{dataset_config_name}')
            elif dataset_name == 'winogrande':
                raw_datasets = load_dataset(dataset_name, dataset_config_name, split="train")
            elif dataset_name == 'hellaswag':
                raw_datasets = load_dataset('../T0_dataset/hellaswag', split="train")
            else:
                # dataset_name: super glue, dataset_config_name: rte
                raw_datasets = load_dataset(dataset_name, dataset_config_name, split="train")
        else:
            if dataset_name == "anli":
                # 读anli数据集（不是p3，是原始数据集）
                raw_datasets = load_dataset(dataset_name, split=f'dev_{dataset_config_name}')
            elif dataset_name == 'hellaswag':
                raw_datasets = load_dataset('../T0_dataset/hellaswag', split="validation")
            else:
                # dataset_name: super glue, dataset_config_name: rte
                raw_datasets = load_dataset(dataset_name, dataset_config_name, split="validation")

        logger.info(f'raw dataset for {dataset_name}_{dataset_config_name}: {raw_datasets}')

        column_names = raw_datasets.column_names
        logger.info(f'column name: {column_names}')

        def preprocess_eval_function(examples):
            bs = len(examples[column_names[0]])  # map的回调函数一次处理一个batch数据(跟训练的batch不一样)

            input_texts = []
            target_texts = []
            answer_choices_texts = []
            for i in range(bs):
                # 样本的每个字段组合起来
                ex = {
                    k: examples[k][i]
                    for k in column_names
                }
                # print(f'debug: ex: {ex}')
                input, target = template.apply(ex)

                ex_answer_choices = template.get_answer_choices_list(ex)
                assert target in ex_answer_choices
                input_texts.append(input)
                target_texts.append(target)
                answer_choices_texts.append(ex_answer_choices)  # 列表的每个元素是该样本的选项列表

            tokenized_inputs = tokenizer(
                input_texts,
                padding=padding,
                max_length=args.max_length,
                truncation=True,
                add_special_tokens=False,
            )
            # input只编码一次, 选项每个都要编码
            # exam * num_choice * label_len
            tokenized_targets = [
                tokenizer(
                    ans_choi,
                    padding=True,
                    max_length=args.target_max_length,
                    truncation=True,
                )
                for ans_choi in answer_choices_texts
            ]

            # k 是input ids/attention_mask等
            # 这里的意思就是是把input_ids等复制选项个数的次数
            # input_ids: [ [exam1, exam1], [exam2, exam2], ... ]
            features = {
                k: [
                    [elem for _ in range(len(tokenized_targets[idx]["input_ids"]))]
                    for idx, elem in enumerate(v)
                ]
                for k, v in tokenized_inputs.items()
            }

            # label的ids： exam_num * label_len
            features["labels"] = [
                tokenized_targets[idx]["input_ids"]
                for idx in range(bs)
            ]

            # label的attention mask
            features["labels_attention_mask"] = [
                tokenized_targets[idx]["attention_mask"]
                for idx in range(bs)
            ]

            # targets 是答案文本
            features["targets"] = [
                answer_choices_texts[idx].index(t)
                for idx, t in enumerate(target_texts)
            ]
            return features

        def preprocess_train_function(examples):
            bs = len(examples[column_names[0]])

            input_texts = []
            target_texts = []
            answer_choices_texts = []
            for i in range(bs):
                # 样本的每个字段组合起来
                ex = {
                    k: examples[k][i]
                    for k in column_names
                }
                # print(f'debug: ex: {ex}')
                input, target = template.apply(ex)

                ex_answer_choices = template.get_answer_choices_list(ex)
                assert target in ex_answer_choices
                input_texts.append(input)
                target_texts.append(target)
                answer_choices_texts.append(ex_answer_choices)

            tokenized_inputs = tokenizer(
                input_texts,
                padding=padding,
                max_length=args.max_length,
                truncation=True,
                add_special_tokens=False,
            )
            # input只编码一次, 选项每个都要编码
            tokenized_targets = tokenizer(
                    target_texts,
                    padding=True,
                    max_length=args.target_max_length,
                    truncation=True,
                )

            # k 是input ids/attention_mask等
            # 这里的意思就是是把input_ids等复制选项个数的次数
            features = tokenized_inputs

            # label的ids
            features["labels"] = tokenized_targets["input_ids"]

            # label的attention mask
            features["labels_attention_mask"] = tokenized_targets["attention_mask"]

            # targets 是答案的id
            features["targets"] = [
                answer_choices_texts[idx].index(t)
                for idx, t in enumerate(target_texts)
            ]
            return features

        # Get the prompt to apply and the possible targets.
        logger.info(f'use template_dir: {args.template_dir}')

        prompts = DatasetTemplates(
            f"{dataset_name}" if dataset_config_name is None else f"{dataset_name}/{dataset_config_name}",
            template_dir=args.template_dir)

        # key是uuid
        # template_list = prompts.name_to_id_mapping.keys()
        template_list = prompts.templates.keys()
        logger.info(f'{dataset_name}的模板列表：{template_list}')

        for template_id in template_list:
            template = prompts.templates[template_id]
            template_name = template.name

            logger.info(f'{template.metadata.original_task}, type: {type(template.metadata.original_task)}')
            if template.metadata.original_task is not True:
                logger.info(f'跳过{template_name}, 因为不是原始任务形式')
                continue

            # 过滤copa样本, 一些prompt只适用于部分样本
            filtered_dataset = None
            if dataset_config_name == 'copa':
                if template_name in ["\u2026What could happen next, C1 or C2?", "\u2026As a result, C1 or C2?"]:
                    filtered_dataset = raw_datasets.filter(lambda example: example['question'] == 'effect')
                if template_name in ["\u2026which may be caused by", "\u2026why? C1 or C2"]:
                    filtered_dataset = raw_datasets.filter(lambda example: example['question'] == 'cause')

            # ga的32 dev在这里过滤，不然filter完可能少于32
            if not filtered_dataset:
                filtered_dataset = raw_datasets
            single_train, single_dev = build_pt_dataset(dataset_name, dataset_config_name, filtered_dataset,
                                                        args.ga_dev_distribution)
            # label_key = 'label'
            # if dataset_name in ['winogrande']:
            #     label_key = 'answer'

            # print(f'filtered_dataset: {filtered_dataset}')
            # print(f'label distribution: {Counter(filtered_dataset[label_key])}')

            print(f'使用数据 {dataset_name}_{dataset_config_name}_{template_name}')

            with accelerator.main_process_first():
                single_train_processed_dataset = single_train.map(
                    preprocess_train_function, batched=True, remove_columns=column_names,
                    load_from_cache_file=False)
                single_dev_processed_dataset = single_dev.map(
                    preprocess_eval_function, batched=True, remove_columns=column_names,
                    load_from_cache_file=False)

                train_dataset_list.append(single_train_processed_dataset)

                unified_task_name = dataset_name
                if dataset_config_name:
                    unified_task_name += f'_{dataset_config_name}'
                unified_task_name += f'_{template_name}_{template_id}'
                # dev数据要分别保存
                dev_dataset_dict[unified_task_name] = single_dev_processed_dataset

    # 开始准备训练
    all_train_dataset = concatenate_datasets(train_dataset_list)
    dev_dataset_dict = dev_dataset_dict  # dev不要concat在一起，因为不同任务选项个数不一样
    print(f'all_train_dataset: {all_train_dataset}')
    print(f'all_dev_dataset: {dev_dataset_dict}')
    do_train(args, model, tokenizer, all_train_dataset, dev_dataset_dict, accelerator)


def do_train(args, model, tokenizer, train_dataset, dev_dataset_dict, accelerator):
    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_train_collator = DataCollatorForMultipleChoiceTraining(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )
        data_eval_collator = DataCollatorForMultipleChoice(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    train_dataset = train_dataset.shuffle(42)
    train_dataloader = DataLoader(train_dataset, collate_fn=data_train_collator, batch_size=args.per_device_train_batch_size)
    train_dataloader = accelerator.prepare(train_dataloader)

    total_train_batch_size = args.per_device_train_batch_size * accelerator.num_processes
    total_dev_batch_size = args.per_device_eval_batch_size * accelerator.num_processes

    # optimizer_grouped_parameters = []
    # TODO: 设置一下
    # optimizer_grouped_parameters.append(
    #     {'params': [soft_prompt_embed.weight], 'weight_decay': 0.00001})

    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": 0.00001,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer = Adafactor(optimizer_grouped_parameters,
                          lr=args.lr,
                          scale_parameter=False,
                          relative_step=False,
                          warmup_init=False,
                          weight_decay=0.00001)
    model.train()

    checkpoint_dir = os.path.join(args.output_dir, 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info("***** Running evaluation *****")
    logger.info(f"  Num train examples = {len(train_dataloader)}")
    # logger.info(f"  Num dev examples = {len(dev_dataloader)}")
    logger.info(f"  Instantaneous train batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Instantaneous dev batch size per device = {args.per_device_eval_batch_size}")
    logger.info(f"  Total eval batch size (w. parallel, distributed) = {total_train_batch_size}")
    logger.info(f"  Total eval batch size (w. parallel, distributed) = {total_dev_batch_size}")

    progress_bar = tqdm(range(args.num_training_steps), disable=not accelerator.is_local_main_process)

    global_step = 0
    train_losses = []
    best_eval_metric = 0
    for _ in range(args.num_training_steps):
        for batch in train_dataloader:
            global_step += 1

            model_inputs = {
                k: batch[k]
                for k in ["input_ids", "attention_mask", "labels"]
            }

            model_output = model(**model_inputs)

            logits = model_output.logits
            logits = torch.log_softmax(logits, dim=-1)
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), model_inputs['labels'].view(-1))

            # loss = model_output.loss
            train_losses.append(loss.detach().cpu())

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            model.zero_grad()

            if global_step % 50 == 0:
                logger.info(
                    f'Time: {datetime.datetime.now()}, global step {global_step}\ttrain loss {np.mean(train_losses)}')
                train_losses = []

            if global_step % args.eval_period == 0:
                # print(f'soft_prompt_embed.weight: {soft_prompt_embed.weight}')
                model.eval()
                eval_metric, task_summary = do_eval(args, model, tokenizer, dev_dataset_dict, accelerator, data_eval_collator)
                logger.info(
                    f'Eval at step: {global_step}, eval_metric: {eval_metric}, previous best: {best_eval_metric}')
                # 处理每个任务：
                # for task_name, acc in task_summary.items():
                #     if task_name not in best_task_summary:
                #         best_task_summary[task_name] = acc
                #     if acc > best_task_summary[task_name]:
                #         best_task_summary[task_name] = acc
                #         # 为单个任务保存最好的ckpt
                #         torch.save(model.state_dict(),
                #                    os.path.join(checkpoint_dir, f'{task_name}.bin'))

                if eval_metric > best_eval_metric:
                    # 如果平均acc大于best，保存一个ckpt
                    best_eval_metric = eval_metric
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_{global_step}.bin'))
                model.train()
            progress_bar.update(1)
            if global_step == args.num_training_steps:
                break
        if global_step == args.num_training_steps:
            break


def do_eval(args, model, tokenizer, dev_dataset_dict, accelerator, data_eval_collator):
    # Metrics
    metric = load_metric("accuracy")
    template_result_summary = {}
    accuracy_sum = 0
    for uniq_task_name, dataset in dev_dataset_dict.items():
        if uniq_task_name.startswith('hellaswag'):
            task_name = 'hellaswag'
        elif uniq_task_name.startswith('super_glue_copa'):
            task_name = 'super_glue_copa'
        elif uniq_task_name.startswith('anli_r1'):
            task_name = 'anli_r1'
        elif uniq_task_name.startswith('anli_r2'):
            task_name = 'anli_r2'
        elif uniq_task_name.startswith('anli_r3'):
            task_name = 'anli_r3'
        elif uniq_task_name.startswith('super_glue_rte'):
            task_name = 'super_glue_rte'
        elif uniq_task_name.startswith('super_glue_cb'):
            task_name = 'super_glue_cb'
        elif uniq_task_name.startswith('super_glue_wsc.fixed'):
            task_name = 'super_glue_wsc.fixed'
        elif uniq_task_name.startswith('super_glue_wic'):
            task_name = 'super_glue_wic'
        elif uniq_task_name.startswith('winogrande_winogrande_xl'):
            task_name = 'winogrande_winogrande_xl'

        # 只评测一次，加快计算
        if args.only_eval_single_template and task_name in template_result_summary:
            continue

        eval_dataloader = DataLoader(dataset, collate_fn=data_eval_collator, batch_size=args.per_device_eval_batch_size)
        eval_dataloader = accelerator.prepare(eval_dataloader)
        print(f'evaluating task: {uniq_task_name}')
        for batch in eval_dataloader:
            model_inputs = {
                k: batch[k]
                for k in ["input_ids", "attention_mask", "labels"]
            }

            with torch.no_grad():
                # [batch_size, seq_len, vocab]
                logits = model(**model_inputs).logits

            # [batch_size, seq_len, vocab]
            masked_log_probs = batch["labels_attention_mask"].unsqueeze(-1) * torch.log_softmax(logits, dim=-1)
            # [batch_size, seq_len]
            seq_token_log_probs = torch.gather(masked_log_probs, -1, batch["labels"].unsqueeze(-1))
            # [batch_size, ]
            seq_log_prob = seq_token_log_probs.squeeze(dim=-1).sum(dim=-1)
            seq_log_prob = seq_log_prob.view(batch["targets"].size(0), -1)
            # TODO(Victor): this reshapes works based on the assumption that all examples have the same number of choices. the pre-processing doesn't make this assumption.

            # [batch_size, choice_num]
            predictions = seq_log_prob.argmax(dim=-1)

            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["targets"]),
            )

        eval_metric = metric.compute()
        accelerator.print(f"Result for {uniq_task_name}: {eval_metric}")
        accuracy_sum += eval_metric['accuracy']
        if task_name not in template_result_summary:
            template_result_summary[task_name] = []
        template_result_summary[task_name].append(eval_metric['accuracy'])

    result_summary = dict()
    for task_name, results in template_result_summary.items():
        result_summary[task_name] = sum(results)   # TODO: 为啥是sum

    result_num = 0   # 测了多少个task
    for k, v in template_result_summary.items():
        result_num += len(v)

    # 返回整体平均的acc， 以及每个数据集各个任务的sum acc
    return accuracy_sum / result_num, result_summary


if __name__ == "__main__":
    main()
