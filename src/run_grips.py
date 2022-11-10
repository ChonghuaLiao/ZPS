import argparse
import copy
import logging
import math
import os
import random
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
import json
import codecs

import datasets
import torch
from datasets import load_dataset, load_metric, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import Counter

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
from templates import DatasetTemplates

from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import word_tokenize, sent_tokenize
# from supar import Parser   # TODO: 这个需要安装，不知道麻烦不
import nltk
import string
import numpy as np
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from templates import Template
import yaml
import uuid
import re


logger = logging.getLogger(__name__)


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

    def __call__(self, features):
        num_choices = len(features[0]["input_ids"])
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
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Pad the labels because it's not padded automatically
        max_label_length = max([len(elem["labels"]) for elem in flattened_features])
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


def detokenize(tokens):
    return TreebankWordDetokenizer().detokenize(tokens)


def get_description_from_template(task_name, template):
    """从template obj中提取description，需要为每个template编写规则, 对于不使用的template返回None"""
    if not template.metadata.original_task:
        return None

    template_text = template.jinja
    template_name = template.name

    description = None

    if template_name in ['MNLI crowdsource', 'based on the previous passage', 'justified in saying']:
        description = template_text.split('|||')[0]
        description = description.replace('{{premise}}', '')
        description = description.replace('"{{hypothesis}}"', 'hypothesis')
    elif template_name in ['should assume', 'must be true', 'can we infer',
                           'guaranteed true']:
        description = template_text.split('|||')[0]
        description = description.replace('{{premise}}', 'premise')
        description = description.replace('"{{hypothesis}}"', 'hypothesis')
    elif template_name in ['does it follow that']:
        description = template_text.split('|||')[0]
        description = description.replace('{{premise}}', 'premise')
        description = description.replace('{{hypothesis}}', 'hypothesis')
    elif template_name in ['does this imply']:
        description = template_text.split('|||')[0]
        description = description.split('\n\n')[1]
        description = description.replace('"{{hypothesis}}"', 'hypothesis')
    elif template_name in ['claim true/false/inconclusive']:
        description = template_text.split(':')[0]
        description = description.replace('{{premise}}', '')
        description = description.strip()
        description += ': hypothesis true or false?'
    # elif template_name in ['GPT-3 style']:
    #     pass  # 无法augment, 没有prompt
    elif template_name in ['take the following as truth', 'guaranteed/possible/impossible',
                           'always/sometimes/never',
                           'consider always/sometimes/never']:
        pass  # 无法augment, 不好处理

    # hellaswage
    elif template_name in ['complete_first_then']:
        description = template_text.split(':')[0]
    elif template_name in ['Randomized prompts template']:
        pass  # 无法augment, 不好处理
    elif template_name in ['Predict ending with hint']:
        description = template_text.split('\n')[0]
    elif template_name in ['if_begins_how_continues']:
        description = template_text.split('\n\n')[0]
        description = description.replace(': {{ ctx }}...', '.')

    # copa
    elif template_name in ['exercise']:
        description = template_text.split('\n')[0]
    elif template_name in ['\u2026What could happen next, C1 or C2?']:
        description = template_text.split('{{ premise }}')[1]
        description = description.split('"{{ answer_choices[0] }}')[0]
    elif template_name in ['i_am_hesitating']:
        description = template_text.split('\n\n')[1]
        description = description.split('{% if question == "cause" %}')[0]
        description += ' option.'
    elif template_name in ['plausible_alternatives']:
        description = template_text.split('\n')[1]
    elif template_name in ['C1 or C2? premise, so/because\u2026', "\u2026why? C1 or C2"]:
        pass  # 无法augment, 没有prompt
    elif template_name in ['choose']:
        description = template_text.split('\n')[1]
    elif template_name in ['\u2026As a result, C1 or C2?', '\u2026which may be caused by']:
        description = template_text.split('{{ premise }}')[1]
        description = description.split('"{{ answer_choices[0] }}"')[0]
    elif template_name in ['best_option']:
        pass  # 无法augment, 不好处理
    elif template_name in ['more likely']:
        description = template_text.split('\n')[0]
    elif template_name in ['cause_effect']:
        description = template_text.split('\n\n')[1]
        description = description.split('{% if question == "cause" %}')[0]
        description = description.strip()
        description += ' option.'

    # rte/CB与anli完全相同
    # wsc.fixed
    elif template_name in ['does the pronoun refer to']:
        description = template_text.split('|||')[0]
        description = description.replace('{{ text }} ', '')
        description = description.replace('"{{ span2_text.lower() }}"', '')
        description = description.replace('{{ span1_text }}', 'the reference')
    elif template_name in ['does p stand for']:
        description = template_text.split('|||')[0]
        description = description.replace('{{ text }} ', '')
        description = description.replace('"{{ span2_text.lower() }}"', 'the pronoun')
        description = description.replace('{{ span1_text }}', 'the reference')
    elif template_name in ['replaced with']:
        description = template_text.split('|||')[0]
        description = description.replace('{{ text }} ', '')
        description = description.replace('"{{ span2_text }}"', '')
        description = description.replace('"{{ span1_text }}"', 'the reference')
    elif template_name in ['the pronoun refers to']:
        description = template_text.split('|||')[0]
        description = description.split('\n')[1]
        description = description.replace('"{{ span2_text }}"', '')
        description = description.replace('{{ span1_text }}', 'the reference')
    elif template_name in ['in other words', 'I think they mean', 'p is/are r',
                           'Who or what is/are', 'by p they mean']:
        pass  # 无法处理
    elif template_name in ['GPT-3 Style', 'GPT-3 style']:
        if task_name.find('wsc') == -1:
            return None
        description = template_text.split('\n\n')[1]
        description = description.replace('\"{{ span2_text }}\"', '')
        description = description.replace('{{ span1_text }}', 'the reference')

    # wic
    elif template_name in ['question-context-meaning-with-label', 'question-context-meaning']:
        description = template_text.split('\n')[0]
        description = description.replace('"{{word}}"', '')
    elif template_name in ['grammar_homework']:
        description = template_text.split('\n\n')[1]
        description = description.split('\n')[0]
        description = description.replace('"{{word}}"', '')
    elif template_name in ['affirmation_true_or_false']:
        description = template_text.split('\n\n')[1]
        description = description.split('\n')[0]
        description = description.replace('"{{word}}"', 'The word')
    elif template_name in ['same_sense']:
        description = template_text.split('\n\n')[1]
        description = description.split('\n')[0]
        description = description.replace('"{{word}}"', '')
    elif template_name in ['GPT-3-prompt', 'GPT-3-prompt-with-label']:
        description = template_text.split('\n')[2]
        description = description.replace('"{{word}}"', '')
        description = description.replace('\'{{word}}\'', '')
        description = description.replace('{{word}}', '')
    elif template_name in ['question-context']:
        description = template_text.split('\n')[0]
        description = description.replace('\'{{word}}\'', '')
    elif template_name in ['polysemous']:
        description = template_text.split('\n\n')[0]
        description = description.replace('"{{word}}"', '')
    elif template_name in ['similar-sense']:
        description = template_text.split('\n')[2]
        description = description.replace('{{word}}', 'the word')
        description = description.strip()

    # winogrande
    elif template_name in ['does underscore refer to']:
        description = template_text.split('{{ option1 }}')[0]
        description = description.replace('{{ sentence }} ', '')
        description = description.replace('_', 'the pronoun')
    elif template_name in ['stand for']:
        description = template_text.split('{{answer_choices[0]}}')[0]
        description = description.replace('_', 'pronoun')
        description += 'the reference?'
    elif template_name in ['underscore refer to']:
        description = template_text.split('\n')[1]
        description = description.split('{{ option1 }}')[0]
        description = description.strip()
        description = description.replace('_', 'pronoun')
    elif template_name in ['fill in the blank']:
        description = template_text.split('\n')[0]
        description = description.replace('_', 'blank')
    elif template_name in ['True or False']:
        description = template_text.split('\n')[0]
        description = description.replace('_', 'pronoun')
        description = description.replace('{{option1}}', 'the reference')
    elif template_name in ['Replace']:
        description = template_text.split('\n')[1]
        description = description.replace('_', 'pronoun')
    # storycloze
    # 0
    elif template_name in ['Answer Given options']:
        description = template_text.split('-')[0]
        description = description.replace('{{input_sentence_1}} {{input_sentence_2}} {{input_sentence_3}} {{input_sentence_4}}', '')
    # 1
    elif template_name in ['Choose Story Ending']:
        description = template_text.split('-')[0]
        description = description.split(' :\n\n{{input_sentence_1}}\n{{input_sentence_2}}\n{{input_sentence_3}}\n{{input_sentence_4}}\n')[1]
        # TODO
    # 3
    elif template_name in ['Story Continuation and Options']:
        description = template_text.split('-')[0]
        description = description.split('\n\n{{input_sentence_1}}\n{{input_sentence_2}}\n{{input_sentence_3}}\n{{input_sentence_4}}\n\n')[0]
        # TODO
    # 2
    elif template_name in ['Movie What Happens Next']:
        description = template_text.split('-')[0]
        description = description.split(' {{input_sentence_1}} {{input_sentence_2}} {{input_sentence_3}} {{input_sentence_4}}')[0]
    # 5
    elif template_name in ['Novel Correct Ending']:
        description = template_text.split('-')[0]
        description = description.split('{{input_sentence_1}} {{input_sentence_2}} {{input_sentence_3}} {{input_sentence_4}} ')[1]
        # TODO
    else:
        print(f'无法识别的template: {template_name}')

    # debug
    # if description is not None:
    #     print(f'template_name: {template_name}\ntemplate_text: {template_text}\ndescription: {description}')
    if description is not None:
        description = description.replace('  ', ' ')
        description = description.strip()

    return description


def get_valid_invalid_option(option_list):
    """返回不合法的option单词，加入黑名单中"""
    option_list = [option.lower().strip() for option in option_list]
    black_set = set()
    if option_list.count('yes') > 0:  # yes/no/(may)
        black_set.add('false')
    elif option_list.count('correct') > 0:  # correct/incorrect/inconlusive
        black_set.add('false')
    elif option_list.count('false') > 0:  # True/false
        pass

    # 不能直接加no，不然find函数会把带no的单词都算命中

    return black_set


def will_keep_with_clean_rule(task_name, template_name, description, option_list):
    """判断生成的description是否合法"""

    description = description.strip()
    description = description.replace('  ', ' ')

    # 长度控制
    if len(description) < 2:
        return False
    if len(description.split(' ')) < 2:
        return False

    if description.count('{') != description.count('}'):
        return False

    MUST_WORD = set()
    BLACK_WORD = {'=', '#', '__', '[', ']', '\\u', '%', '{{{', '}}}', 'God', '\\x', 'ie.', 'i.e.',
                  '(2)', '(1)', '(?', 'ENSLAVED', 'DevOps', 'U.S.', '**', '...', '+', '--', '(a)',
                  '(1)'}
    # 黑名单，过滤有特殊符号的prompt

    VALID_PLACEHOLDER = {'hypothesis', 'premise', 'ctx', 'pronoun', 'reference', 'word', 'option' 'option1', 'option2'}

    # 必须是合法placeholder
    re_pattern = re.compile(r'{{(.*?)}}', re.S)  # 最小匹配
    match_placeholder = re.findall(re_pattern, description)  # 必须是合法的这几个placeholder
    for placeholder in match_placeholder:
        # 每个placeholder的个数不能超过1
        if match_placeholder.count(placeholder) > 1:
            return False

        placeholder = placeholder.strip()
        if placeholder not in VALID_PLACEHOLDER:
            return False
    if len(match_placeholder) > 2:
        return False

    if template_name in ['should assume', 'does it follow that',
                         'must be true', 'can we infer', 'guaranteed true']:
        if description.find('premise') == -1 or description.find('hypothesis') == -1:
            return False
        if description.count('premise') != 1 or description.count('hypothesis') != 1:
            return False
        BLACK_WORD.update(get_valid_invalid_option(option_list))
    elif template_name in ['MNLI crowdsource', 'based on the previous passage', 'justified in saying']:
        if description.find('hypothesis') == -1:
            return False
        if description.count('hypothesis') != 1:
            return False
        BLACK_WORD.update(get_valid_invalid_option(option_list))
        BLACK_WORD.update({'Is there'})
    elif template_name in ['does this imply']:
        if description.find('hypothesis') == -1:
            return False
        if description.count('hypothesis') != 1:
            return False
        BLACK_WORD.update(get_valid_invalid_option(option_list))
    elif template_name in ['claim true/false/inconclusive']:
        if description.find(':') == -1:
            return False
        if description.find('hypothesis') == -1:
            return False
        BLACK_WORD.update(get_valid_invalid_option(option_list))
        BLACK_WORD.update({'not true'})

    # hellaswag
    elif template_name in ['complete_first_then']:
        MUST_WORD.add('end')
    elif template_name in ['Predict ending with hint']:
        MUST_WORD.add('end')
        BLACK_WORD.update({'What are', 'Whats', 'Why'})
    elif template_name in ['if_begins_how_continues']:
        if description.find('this.') == -1 and description.find('this,') == -1:
            return False
        BLACK_WORD.update({'What', 'different', 'Imagine'})

    # copa
    elif template_name in ['exercise']:
        MUST_WORD.add('alternative')
        BLACK_WORD.update({'three', 'four', 'five', 'six', 'How many', 'subject', 'least'})
    elif template_name in ['\u2026What could happen next, C1 or C2?']:
        MUST_WORD.add('happen')
        BLACK_WORD.update({'three', 'four', 'five', 'six', 'How many', 'subject', 'want',
                           'lucky', 'possibility', 'odds'})
    elif template_name in ['i_am_hesitating']:
        if not description[:-1].endswith('option'):
            return False
        BLACK_WORD.update({'three', 'four', 'five', 'six', 'How many', 'subject'})
    elif template_name in ['plausible_alternatives']:
        # MUST_WORD.add('option')  # t5生成质量很好，不需要这个
        BLACK_WORD.update({'three', 'four', 'five', 'six', 'How many', 'subject', 'less'})
    elif template_name in ['\u2026As a result, C1 or C2?']:
        if not description.endswith(','):
            return False
        BLACK_WORD.update({'three', 'four', 'five', 'six', 'How many', 'subject', 'teaching'})
    elif template_name in ['\u2026which may be caused by']:
        if description[-1] in string.punctuation:
            return False  # 不能以标点符号结尾
        BLACK_WORD.update({'three', 'four', 'five', 'six', 'How many', 'subject', 'NOT'})
    elif template_name in ['choose']:
        if not description.endswith(':'):
            return False
    elif template_name in ['more likely']:
        MUST_WORD.add('continuation')
        BLACK_WORD.update({'three', 'four', 'five', 'six', 'How many', 'subject', 'less', 'least'})
    elif template_name in ['cause_effect']:
        if not description[:-1].endswith('option'):
            return False
        BLACK_WORD.update({'three', 'four', 'five', 'six', 'How many', 'subject', 'Do you'})

    # wsc.fixed
    elif template_name in ['does the pronoun refer to', 'does p stand for']:
        if description.find('pronoun') == -1 or description.find('reference') == -1:
            return False
        if description.count('pronoun') != 1 or description.count('reference') != 1:
            return False
    elif template_name in ['replaced with', 'the pronoun refers to']:
        if description.find('pronoun') == -1 or description.find('reference') == -1:
            return False
        if description.count('pronoun') != 1 or description.count('reference') != 1:
            return False
    elif template_name in ['GPT-3 Style', 'GPT-3 style']:
        if task_name.find('wsc') == -1:
            return None  # 只适用于wsc
        if description.find('reference') == -1 or description.find('pronoun') == -1:
            return False
        if description.count('pronoun') != 1 or description.count('reference') != 1:
            return False
    elif template_name in ['by p they mean']:
        pass

    # wic
    elif template_name in ['question-context-meaning-with-label', 'question-context-meaning']:
        if description.find('word') == -1:
            return False
        if description.count('word') != 1:
            return False
        BLACK_WORD.update({'friend', 'watch', 'verb', 'different', 'die', 'which'})
    elif template_name in ['grammar_homework']:
        if description.find('word') == -1:
            return False
        if description.count('word') != 1:
            return False
        BLACK_WORD.update({'friend', 'watch', 'verb', 'different', 'die', 'which', 'How'})
    elif template_name in ['affirmation_true_or_false']:
        if description.find('word') == -1:
            return False
        if description.count('word') != 1:
            return False
        BLACK_WORD.update({'friend', 'watch', 'verb', 'different', 'die', 'which', 'What'})
    elif template_name in ['same_sense']:
        if description.find('word') == -1:
            return False
        if description.count('word') != 1:
            return False
        BLACK_WORD.update({'friend', 'watch', 'verb', 'different', 'die', 'which', 'American'})
    elif template_name in ['GPT-3-prompt', 'GPT-3-prompt-with-label']:
        if description.find('word') == -1:
            return False
        if description.count('word') != 1:
            return False
        BLACK_WORD.update({'friend', 'watch', 'verb', 'different', 'die', 'which', 'American'})
    elif template_name in ['question-context']:
        if description.find('word') == -1:
            return False
        BLACK_WORD.update({'friend', 'watch', 'verb', 'different', 'die', 'which', 'American'})
    elif template_name in ['polysemous']:
        if description.find('word') == -1:
            return False
        if description.count('word') != 1:
            return False
        BLACK_WORD.update({'friend', 'watch', 'verb', 'different', 'die', 'which', 'American'})
    elif template_name in ['similar-sense']:
        if description.find('word') == -1:
            return False
        if description.count('word') != 1:
            return False
        BLACK_WORD.update({'friend', 'watch', 'verb', 'different', 'die', 'which', 'American'})

    # winogrande
    elif template_name in ['does underscore refer to']:
        if description.find('pronoun') == -1:
            return False
        if description.count('pronoun') != 1:
            return False
        if description[-1] in string.punctuation:
            return False  # 不能以标点符号结尾
        BLACK_WORD.update({'\'', 'Why'})
    elif template_name in ['stand for']:
        if description.find('pronoun') == -1 or description.find('reference') == -1:
            return False
        if description.count('pronoun') != 1 or description.count('reference') != 1:
            return False
        if not description[:-1].endswith('reference'):
            return False
        BLACK_WORD.update({'is', 'which', '\'', 'Why'})
    elif template_name in ['underscore refer to']:
        if description.find('pronoun') == -1:
            return False
        if description.count('pronoun') != 1:
            return False
        BLACK_WORD.update({'If', 'subject', 'object', 'function', '\'',  'Why'})
    elif template_name in ['fill in the blank']:
        if description.find('blank') == -1:
            return False
        if description.count('blank') != 1:
            return False
        BLACK_WORD.update({'\'', 'Why'})
    elif template_name in ['True or False']:
        if description.find('pronoun') == -1 or description.find('reference') == -1:
            return False
        if description.count('pronoun') != 1 or description.count('reference') != 1:
            return False
        BLACK_WORD.update({'\'', 'Why'})
    elif template_name in ['Replace']:
        if description.find('pronoun') == -1:
            return False
        if description.count('pronoun') != 1:
            return False
        BLACK_WORD.update({'\'', 'Why'})
    else:
        pass

    # 必须包含这些词
    for word in MUST_WORD:
        if description.find(word) == -1:
            return False

    # 不能包含黑名单里的词
    for word in BLACK_WORD:
        if description.find(word) != -1:
            return False

    return True


def score(args, accelerator, template, dataset, model, tokenizer, metric):
    column_names = dataset.column_names
    logger.info(f'column name: {column_names}')

    def preprocess_function(examples):
        bs = len(examples[column_names[0]])

        input_texts = []
        target_texts = []
        answer_choices_texts = []
        for i in range(bs):
            ex = {
                k: examples[k][i]
                for k in column_names
            }

            input, target = template.apply(ex)

            ex_answer_choices = template.get_answer_choices_list(ex)
            assert target in ex_answer_choices
            input_texts.append(input)
            target_texts.append(target)
            answer_choices_texts.append(ex_answer_choices)

        tokenized_inputs = tokenizer(
            input_texts,
            padding=False,
            max_length=1024,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = [
            tokenizer(
                ans_choi,
                padding=False,
                max_length=256,
                truncation=True,
            )
            for ans_choi in answer_choices_texts
        ]

        features = {
            k: [
                [elem for _ in range(len(tokenized_targets[idx]["input_ids"]))]
                for idx, elem in enumerate(v)
            ]
            for k, v in tokenized_inputs.items()
        }

        features["labels"] = [
            tokenized_targets[idx]["input_ids"]
            for idx in range(bs)
        ]
        features["labels_attention_mask"] = [
            tokenized_targets[idx]["attention_mask"]
            for idx in range(bs)
        ]
        features["targets"] = [
            answer_choices_texts[idx].index(t)
            for idx, t in enumerate(target_texts)
        ]

        return features

    eval_dataset = dataset.map(
        preprocess_function, batched=True, remove_columns=column_names,
        load_from_cache_file=False)

    data_collator = DataCollatorForMultipleChoice(
        tokenizer, pad_to_multiple_of=None
    )

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator,
                                 batch_size=args.per_device_eval_batch_size)

    # Prepare everything with our `accelerator`.
    eval_dataloader = accelerator.prepare(eval_dataloader)

    # Eval!
    progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)

    model.eval()
    for batch in eval_dataloader:
        # batch: attention_mask, input_ids, labels(答案的token ids), labels_attention_mask
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

        progress_bar.update(1)

    eval_metric = metric.compute()

    return float(eval_metric['accuracy'])


def check_child(tree):
    check = False
    count = 0
    total_count = 0
    for subtree in tree:
        total_count += 1
        if type(subtree) == nltk.tree.Tree:
            if subtree.label() == '_':
                count += 1
    if count >= total_count - count: check = True

    return check


def collect_leaves(parsed_tree):
    leaves = []
    for tree in parsed_tree:
        if type(parsed_tree) != nltk.tree.Tree: continue
        if tree.label() == '_':
            leaves.append(detokenize(tree.leaves()))
            continue
        if check_child(tree):
            leaves.append(detokenize(tree.leaves()))
        else:
            leaves.extend(collect_leaves(tree))
    return leaves


def get_phrases(instruction, parser):  # one possible way of obtaining disjoint phrases
    phrases = []
    for sentence in sent_tokenize(instruction):
        parsed_tree = parser.predict(word_tokenize(sentence), verbose=False).sentences[0].trees[0]
        leaves = collect_leaves(parsed_tree)
        phrases.extend(leaves)
    phrases = [detokenize(word_tokenize(phrase)) for phrase in phrases if
               phrase not in string.punctuation or phrase == '']
    return phrases


def get_phrase_lookup(base_candidate, parser, level='word'):
    """因为我们prompt很短，长度就是1-2个句子，所以用phrase还是word?"""
    if level == 'phrase':
        phrase_lookup = {p: phrase for p, phrase in enumerate(get_phrases(base_candidate, parser))}
    elif level == 'word':
        # 我们prompt很短，感觉用word比较合适
        words = word_tokenize(base_candidate)
        words = [w for w in words if w not in string.punctuation or w != '']
        phrase_lookup = {p: phrase for p, phrase in enumerate(words)}
    else:
        raise NotImplementedError(level)
    return phrase_lookup


def delete_phrase(candidate, phrase):
    if candidate.find(' ' + phrase) > 0:
        answer = candidate.replace(' ' + phrase, ' ')
    elif candidate.find(phrase + ' ') > 0:
        answer = candidate.replace(phrase + ' ', ' ')
    else:
        answer = candidate.replace(phrase, '')
    return answer


def add_phrase(candidate, phrase, after):
    if after == '':
        answer = phrase + ' ' + candidate
    else:
        if candidate.find(' ' + after) > 0:
            answer = candidate.replace(' ' + after, ' ' + after + ' ' + phrase)
        elif candidate.find(after + ' ') > 0:
            answer = candidate.replace(after + ' ', after + ' ' + phrase + ' ')
        else:
            answer = candidate.replace(after, after + phrase)
    return answer


def swap_phrases(candidate, phrase_1, phrase_2):
    if candidate.find(' ' + phrase_1 + ' ') >= 0:
        answer = candidate.replace(' ' + phrase_1 + ' ', ' <1> ')
    else:
        answer = candidate.replace(phrase_1, '<1>')
    # 这里原作者实现错了，我进行了修正
    if candidate.find(' ' + phrase_2 + ' ') >= 0:
        # answer = candidate.replace(' ' + phrase_2 + ' ', ' <2> ')
        answer = answer.replace(' ' + phrase_2 + ' ', ' <2> ')
    else:
        # answer = candidate.replace(phrase_2, '<2>')
        answer = answer.replace(phrase_2, '<2>')
    answer = answer.replace('<1>', phrase_2)
    answer = answer.replace('<2>', phrase_1)
    return answer


def get_response(input_text, num_return_sequences, num_beams, para_model, para_tokenizer):
    batch = para_tokenizer([input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt")
    batch = {k: v.cuda() for k, v in batch.items()}
    translated = para_model.generate(**batch, max_length=60, num_beams=num_beams,
                                     num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = para_tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text


def substitute_phrase(candidate, phrase, para_model, para_tokenizer):
    num_beams = 10
    num_return_sequences = 10
    paraphrases = get_response(phrase, num_return_sequences, num_beams, para_model, para_tokenizer)
    paraphrase = np.random.choice(paraphrases, 1)[0]
    paraphrase = paraphrase.strip('.')
    if candidate.find(' ' + phrase) > 0:
        answer = candidate.replace(' ' + phrase, ' ' + paraphrase)
    elif candidate.find(phrase + ' ') > 0:
        answer = candidate.replace(phrase + ' ', paraphrase + ' ')
    else:
        answer = candidate.replace(phrase, paraphrase)
    return answer


def perform_edit(edit, base, phrase_lookup, delete_tracker, para_model, para_tokenizer):
    """执行编辑操作"""
    if edit == 'del':
        # 从phrase_lookup中选一个del的位置
        [i] = np.random.choice(list(phrase_lookup.keys()), 1)
        # 返回删除后的instruction，还有删除的位置
        return delete_phrase(base, phrase_lookup[i]), [i]
    elif edit == 'swap':
        try:
            [i, j] = np.random.choice(list(phrase_lookup.keys()), 2, replace=False)
        except:
            [i, j] = np.random.choice(list(phrase_lookup.keys()), 2, replace=True)
        # 返回交换后的instruction和位置
        return swap_phrases(base, phrase_lookup[i], phrase_lookup[j]), [i, j]
    elif edit == 'sub':
        [i] = np.random.choice(list(phrase_lookup.keys()), 1)
        return substitute_phrase(base, phrase_lookup[i], para_model, para_tokenizer), [i]
    elif edit == 'add':
        keys = list(phrase_lookup.keys())
        keys.append(-1)
        [i] = np.random.choice(keys, 1)  # 采样得到要插入的位置
        if i >= 0:
            after = phrase_lookup[i]  # 得到add位置的后继元素
        else:
            after = ''
        if len(delete_tracker) == 0: return base, []   # TODO: 如果delete_tracker为空，则不执行add操作，为什么?
        phrase = np.random.choice(delete_tracker, 1)[0]
        return add_phrase(base, phrase, after), [phrase]


def build_template(task_name: str, template_name, augment_description):
    """根据description和template_name生成template"""
    augment_template = None
    if template_name in ['must be true', 'can we infer', 'guaranteed true']:
        augment_template = augment_description.replace('premise', ' {{premise}} ')
        augment_template = augment_template.replace('hypothesis', ' "{{hypothesis}}" ')
        if task_name.startswith('anli'):
            augment_template = augment_template + ' ||| {{ answer_choices[label] }}'
        elif task_name.startswith('super_glue'):
            augment_template = augment_template + ' ||| {% if label != -1 %}{{ answer_choices[label] }}{% endif %}'
    elif template_name in ['should assume']:
        augment_template = augment_description.replace('premise', ' {{premise}} ')
        augment_template = augment_template.replace('hypothesis', ' "{{hypothesis}}" ')
        if task_name.startswith('anli'):
            augment_template = augment_template + ' ||| {{ answer_choices[label] }}'
        elif task_name.startswith('super_glue'):
            augment_template = augment_template + ' ||| {% if label != -1 %}{{ answer_choices[label] }}{% endif %}'
    elif template_name in ['does it follow that']:
        augment_template = augment_description.replace('premise', ' {{premise}} ')
        augment_template = augment_template.replace('hypothesis', ' {{hypothesis}} ')
        if task_name.startswith('anli'):
            augment_template = augment_template + ' ||| {{ answer_choices[label] }}'
        elif task_name.startswith('super_glue'):
            augment_template = augment_template + ' ||| {% if label !=-1 %}{{ answer_choices[label] }}{% endif %}'
    elif template_name in ['MNLI crowdsource', 'based on the previous passage', 'justified in saying']:
        augment_template = '{{premise}} ' + augment_description
        augment_template = augment_template.replace('hypothesis', '"{{hypothesis}}"')
        if task_name.startswith('anli'):
            augment_template = augment_template + ' ||| {{ answer_choices[label] }}'
        elif task_name.startswith('super_glue'):
            augment_template = augment_template + ' ||| {% if label !=-1 %}{{ answer_choices[label] }}{% endif %}'
    elif template_name in ['does this imply']:
        augment_template = augment_description.replace('hypothesis', '"{{hypothesis}}"')
        if task_name.startswith('anli'):
            augment_template = '{{premise}} \n\n' + augment_template + ' ||| {{ answer_choices[label] }}'
        elif task_name.startswith('super_glue'):
            augment_template = '{{premise}} \n\n' + augment_template + ' ||| {% if label != -1 %}{{answer_choices[label]}}{% endif %}'
    elif template_name in ['claim true/false/inconclusive']:
        augment_template = augment_description.split(':')[0]
        augment_template = '{{premise}} ' + augment_template + \
                           ': "{{hypothesis}}" {{"true"}}, {{"false"}}, or {{"inconclusive"}}?'
        if task_name.startswith('anli'):
            augment_template = augment_template + ' ||| {{ answer_choices[label] }}'
        elif task_name.startswith('super_glue'):
            augment_template = augment_template + ' ||| {% if label !=-1 %}{{ answer_choices[label] }}{% endif %}'

    # hellaswage
    elif template_name in ['complete_first_then']:
        if augment_description[-1] in string.punctuation:
            augment_description = augment_description[:-1]
        augment_template = augment_description + ':\n'
        augment_template = augment_template + 'First, {{ ctx_a.lower() }} Then, {{ ctx_b.lower() }} ...\n\n(a) {{ answer_choices[0] }}\n\n(b) {{ answer_choices[1] }}\n\n(c) {{ answer_choices[2] }}\n\n(d) {{ answer_choices[3] }}\n|||\n{{ answer_choices[label | int()] }}'
    elif template_name in ['Predict ending with hint']:
        augment_template = augment_description + '\n{{ctx}}\n\n(a)  {{answer_choices[0]}}\n\n(b)  {{answer_choices[1]}}\n\n(c)  {{answer_choices[2]}}\n\n(d)  {{answer_choices[3]}}\n\nHint: the topic of the sentence is {{activity_label}}\n|||\n{{answer_choices [label | int()]}}'
    elif template_name in ['if_begins_how_continues']:
        if augment_description.find('this.') != -1:
            augment_template = augment_description.replace('this.', 'this: {{ ctx }}...')
        else:
            augment_template = augment_description.replace('this,', 'this: {{ ctx }}...')
        augment_template = augment_template + ' \n\nEnding 1: {{ endings[0] }}\n\nEnding 2: {{ endings[1] }}\n\nEnding 3: {{ endings[2] }}\n\nEnding 4: {{ endings[3] }}\n|||{{answer_choices[label | int()] }}'

    # copa
    elif template_name in ['exercise']:
        if not augment_description.startswith('Exercise'):
            augment_description = 'Exercise: ' + augment_description
        augment_template = augment_description + '\n\n{{ premise }} {% if question == "cause" %} because... {% else %} so... {% endif%}\n- {{choice1}}\n- {{choice2}} ||| {% if label != -1 %}{{ answer_choices[label] }}{%endif%}'
    elif template_name in ['\u2026What could happen next, C1 or C2?']:
        augment_template = '{% if question == \"effect\" %} \n{{ premise }} ' + augment_description + \
            ' \"{{ answer_choices[0] }}\" or \"{{ answer_choices[1] }}\"? ||| {% if label != -1 %}{{ answer_choices[label] }}{%endif%}\n{% endif %}'
    elif template_name in ['i_am_hesitating']:
        augment_template = '{{ premise }} \n\n' + augment_description
        augment_template = augment_template[:-(len('option.'))]
        augment_template = augment_template + '{% if question == "cause" %} cause: {% else %} effect: {% endif %}\n- {{choice1}}\n- {{choice2}} ||| {% if label != -1 %}{{ answer_choices[label] }}{%endif%}'
    elif template_name in ['plausible_alternatives']:
        augment_template = '{{ premise }} {% if question == "cause" %} This happened because... {% else %} As a consequence... {% endif %}\n' + augment_description
        augment_template = augment_template + '\n- {{choice1}}\n- {{choice2}} ||| {% if label != -1 %}{{ answer_choices[label] }}{%endif%}'
    elif template_name in ['\u2026which may be caused by']:
        augment_template = '{% if question == \"cause\" %} \n{{ premise }} ' + augment_description
        augment_template = augment_template + ' \"{{ answer_choices[0] }}\" or \"{{ answer_choices[1] }}\"? ||| {% if label != -1 %}{{ answer_choices[label] }}{%endif%}\n{% endif %}'
    elif template_name in ['\u2026As a result, C1 or C2?']:
        augment_template = '{% if question == \"effect\" %} \n{{ premise }} ' + augment_description
        augment_template = augment_template + ' \"{{ answer_choices[0] }}\" or \"{{ answer_choices[1] }}\"? ||| {% if label != -1 %}{{ answer_choices[label] }}{%endif%}\n{% endif %}'
    elif template_name in ['choose']:
        augment_template = '{{ premise }} {% if question == "cause" %} because... {% else %} so... {% endif %}\n' + augment_description
        augment_template = augment_template + '\n- {{choice1}}\n- {{choice2}} ||| {% if label != -1 %}{{ answer_choices[label] }}{%endif%}'
    elif template_name in ['more likely']:
        augment_template = augment_description + '\n{{ premise }} {% if question == "cause" %} as a result of: {% else %} as a consequence:{% endif %}\n- {{choice1}}\n- {{choice2}} ||| {% if label != -1 %}{{ answer_choices[label] }}{%endif%}'
    elif template_name in ['cause_effect']:
        augment_template = '{{ premise }}\n\n' + augment_description
        augment_template = augment_template[:-len('option.')]
        augment_template = augment_template + ' {% if question == "cause" %} cause: {% else %} effect:{% endif %}\n- {{choice1}}\n- {{choice2}} ||| {% if label != -1 %}{{ answer_choices[label] }}{%endif%}'

    # wsc.fixed
    elif template_name in ['does the pronoun refer to']:
        augment_template = '{{ text }} ' + augment_description
        augment_template = augment_template.replace('pronoun', ' "{{ span2_text.lower() }}" ')
        if augment_template.find('the reference') != -1:
            augment_template = augment_template.replace('the reference', ' {{ span1_text }} ')
        else:
            augment_template = augment_template.replace('reference', ' {{ span1_text }} ')
        augment_template = augment_template + '||| {% if label != -1 %}{{ answer_choices[label] }}{% endif %}'
    elif template_name in ['does p stand for']:
        augment_template = '{{ text }} ' + augment_description
        if augment_template.find('the pronoun') != -1:
            augment_template = augment_template.replace('the pronoun', ' "{{ span2_text.lower() }}" ')
        else:
            augment_template = augment_template.replace('pronoun', ' "{{ span2_text.lower() }}" ')
        if augment_template.find('the reference') != -1:
            augment_template = augment_template.replace('the reference', ' {{ span1_text }} ')
        else:
            augment_template = augment_template.replace('reference', ' {{ span1_text }} ')
        augment_template = augment_template + ' ||| {% if label != -1 %}{{ answer_choices[label] }}{% endif %}'
    elif template_name in ['replaced with']:
        augment_template = '{{ text }} ' + augment_description
        augment_template = augment_template.replace('pronoun', 'pronoun "{{ span2_text }}" ')
        if augment_template.find('the reference') != -1:
            augment_template = augment_template.replace('the reference', ' "{{ span1_text }}" ')
        else:
            augment_template = augment_template.replace('reference', ' "{{ span1_text }}" ')
        augment_template = augment_template + ' ||| {% if label != -1 %}{{ answer_choices[label] }}{% endif %}'
    elif template_name in ['the pronoun refers to']:
        augment_template = '{{ text }} \n' + augment_description
        augment_template = augment_template.replace('pronoun', 'pronoun {{ span2_text }} ')
        augment_template = augment_template.replace('{{reference}}', ' {{ span1_text }} ')
        if augment_template.find('the reference') != -1:
            augment_template = augment_template.replace('the reference', ' {{ span1_text }} ')
        else:
            augment_template = augment_template.replace('reference', ' {{ span1_text }} ')
        augment_template = augment_template + ' ||| {% if label != -1 %}{{ answer_choices[label] }}{% endif %}'
    elif template_name in ['GPT-3 Style', 'GPT-3 style']:
        if task_name.find('wsc') == -1:
            return None
        augment_template = 'Passage: {{ text }} \n\n' + augment_description + '\n\nAnswer: ||| {% if label != -1 %}{{ answer_choices[label] }}{% endif %}'
        augment_template = augment_template.replace('pronoun', 'pronoun \"{{ span2_text }}\" ')
        if augment_template.find('the reference') != -1:
            augment_template = augment_template.replace('the reference', ' {{ span1_text }} ')
        else:
            augment_template = augment_template.replace('reference', ' {{ span1_text }} ')
    elif template_name in ['by p they mean']:
        pass

    # wic
    elif template_name in ['question-context-meaning-with-label', 'question-context-meaning']:
        augment_description = augment_description.replace('words', 'word')
        augment_template = augment_description.replace('word', 'word "{{word}}"')
        augment_template = augment_template + '\n{{sentence1}}\n{{sentence2}}\n||| {% if label != -1%}\n{{answer_choices[label]}}\n{% endif %}'
    elif template_name in ['grammar_homework']:
        augment_template = 'Homework\n\n' + augment_description
        augment_template = augment_template.replace('word', 'word "{{word}}"')
        augment_template = augment_template + '\n{{sentence1}}\n{{sentence2}}\n||| {% if label != -1%}\n{{answer_choices[label]}}\n{% endif %}'
    elif template_name in ['affirmation_true_or_false']:
        augment_template = 'Sentence A: {{sentence1}}\nSentence B: {{sentence2}}\n\n' + augment_description
        if augment_template.find('the word') != -1:
            augment_template = augment_template.replace('the word', '"{{word}}"')
        elif augment_template.find('The word') != -1:
            augment_template = augment_template.replace('The word', '"{{word}}"')
        else:
            augment_template = augment_template.replace('word', '"{{word}}"')
        augment_template = augment_template + '\n||| {% if label != -1%}\n{{answer_choices[label]}}\n{% endif %}'
    elif template_name in ['same_sense']:
        augment_template = augment_description.replace('word', 'word "{{word}}"')
        augment_template = 'Sentence A: {{sentence1}}\nSentence B: {{sentence2}}\n\n' + augment_template
        augment_template = augment_template + '\n||| {% if label != -1 %}\n{{answer_choices[label]}}\n{% endif %}'
    elif template_name in ['GPT-3-prompt', 'GPT-3-prompt-with-label']:
        augment_template = augment_description.replace('word', 'word "{{word}}"')
        augment_template = '{{sentence1}}\n{{sentence2}}\n' + augment_template
        augment_template = augment_template + '\n||| {% if label != -1%}\n{{answer_choices[label]}}\n{% endif %}'
    elif template_name in ['question-context']:
        augment_template = augment_description.replace('word', 'word \'{{word}}\'')
        augment_template = augment_template + '\n{{sentence1}}\n{{sentence2}}\n||| {% if label != -1%}\n{{answer_choices[label]}}\n{% endif %}'
    elif template_name in ['polysemous']:
        augment_template = augment_description.replace('word', 'word "{{word}}"')
        augment_template = augment_template + '\n\nSentence 1: {{sentence1}}\nSentence 2: {{sentence2}}\n||| {% if label != -1%}\n{{answer_choices[label]}}\n{% endif %}'
    elif template_name in ['similar-sense']:
        if augment_description.find('the word') != -1:
            augment_template = augment_description.replace('the word', '{{word}}')
        else:
            augment_template = augment_description.replace('word', '{{word}}')
        augment_template = '{{sentence1}}\n{{sentence2}}\n' + augment_template + '\n||| {% if label != -1%}\n{{answer_choices[label]}}\n{% endif %}'

    # winogrande
    elif template_name in ['does underscore refer to']:
        if augment_description.find('the pronoun') != -1:
            augment_template = augment_description.replace('the pronoun', '_')
        else:
            augment_template = augment_description.replace('pronoun', '_')
        augment_template = augment_template + ' {{ option1 }} or {{ option2 }}? ||| {% if answer ==  "1" %} {{option1}} {% else %} {{ option2 }} {% endif %}'
        augment_template = '{{ sentence }} ' + augment_template
    elif template_name in ['stand for']:
        augment_template = augment_description.replace('pronoun', '_')
        if augment_template.find('the reference') != -1:
            augment_template = augment_template[:-len('the reference?')]
        else:
            augment_template = augment_template[:-len('reference?')]
        augment_template = augment_template + ' {{answer_choices[0]}} or {{answer_choices[1]}}?\n{{sentence}}|||\n{{answer_choices[answer | int - 1]}}'
    elif template_name in ['underscore refer to']:
        augment_template = augment_description.replace('pronoun', '_')
        augment_template = '{{sentence}}\n' + augment_template + ' {{ option1 }} or {{ option2 }}? ||| {% if answer == "1" %} {{option1}} {% else %} {{ option2 }} {% endif %}'
    elif template_name in ['fill in the blank']:
        if augment_description.find('blanks') != -1:
            augment_template = augment_description.replace('blanks', '_')
        else:
            augment_template = augment_description.replace('blank', '_')
        augment_template = augment_template + '\n{{sentence}}\n\nChoices:\n- {{ option1 }}\n- {{ option2 }}\n\nAnswer: ||| {% if answer == "1" %} {{option1}} {% else %} {{ option2 }} {% endif %}'
    elif template_name in ['True or False']:
        augment_template = augment_description.replace('pronoun', '_')
        if augment_template.find('the reference') != -1:
            augment_template = augment_template.replace('the reference', '{{option1}}')
        else:
            augment_template = augment_template.replace('reference', '{{option1}}')
        augment_template = augment_template + '\n{{sentence}}|||\n{{answer_choices[answer|int - 1]}}'
    elif template_name in ['Replace']:
        augment_template = augment_description.replace('pronoun', '_')
        augment_template = '{{sentence}}\n' + augment_template + '\n- {{option1}}\n- {{option2}}\n|||\n{% if answer == "1" %} {{option1}} {% else %} {{ option2 }} {% endif %}'
    # storycloze
    elif template_name in ['Answer Given options']:
        augment_template = '{{input_sentence_1}} {{input_sentence_2}} {{input_sentence_3}} {{input_sentence_4}} ' + augment_description + ' - {{answer_choices | join("\\n- ")}} ||| {{answer_choices[answer_right_ending -1]}}'
    elif template_name in ['Choose Story Ending']:
        augment_template = 'Read the following story :\n\n{{input_sentence_1}}\n{{input_sentence_2}}\n{{input_sentence_3}}\n{{input_sentence_4}}\n\n' + augment_description + ' \n- {{answer_choices | join("\\n- ")}}\n|||\n\n{{answer_choices[answer_right_ending -1]}}'
    elif template_name in ['Story Continuation and Options']:
        augment_template = augment_description + '\n\n{{input_sentence_1}}\n{{input_sentence_2}}\n{{input_sentence_3}}\n{{input_sentence_4}}\n\nChoose from the following options:\n- {{answer_choices | join("\\n- ")}}\n|||\n\n{{answer_choices[answer_right_ending -1]}}'
    elif template_name in ['Movie What Happens Next']:
        augment_template = augment_description + ' {{input_sentence_1}} {{input_sentence_2}} {{input_sentence_3}} {{input_sentence_4}} What happens next? - {{answer_choices | join("\\n- ")}} ||| {{answer_choices[answer_right_ending -1]}}'
    elif template_name in ['Novel Correct Ending']:
        augment_template = 'I read the following novel: {{input_sentence_1}} {{input_sentence_2}} {{input_sentence_3}} {{input_sentence_4}} ' + augment_description + ' - {{answer_choices | join("\\n- ")}} ||| {{answer_choices[answer_right_ending -1]}}'
        # TODO
    else:
        print(f'无法识别的template_name: {template_name}')


    if augment_template is not None:
        augment_template = augment_template.strip()
        augment_template = augment_template.replace('  ', ' ')
        augment_template = augment_template.replace('  ', ' ')

    return augment_template


def build_new_template_from_description(task_name, template, description):
    new_template = copy.deepcopy(template)
    template_name = template.name
    augment_template_text = build_template(task_name, template_name, description)
    new_template.jinja = augment_template_text
    new_template.id = str(uuid.uuid4())
    return new_template


def manual_detokenize(token_list):
    s = ''
    for token in token_list:
        if token in string.punctuation:
            s += token
        else:
            s += f' {token}'
    return s.strip()


def run_grips(args, accelerator, task_name, template, model, tokenizer, para_model, para_tokenizer, dataset, metric, phrase_parser, num_steps=10):
    """在这里跑grips"""

    edit_operations = ['del', 'swap', 'sub', 'add']
    num_candidates = 5

    # 得到要augment的句子
    logger.info(f'task_name: {task_name}, template name: {template.name}, jinja: {template.jinja}')
    description = get_description_from_template(task_name, template)
    if description is None:
        return None
    logger.info(f'description for template {task_name}: {description}')

    instruction = description
    operations_tracker = []
    base_candidate = detokenize(word_tokenize(instruction))  # 初始的instruction
    assert word_tokenize(base_candidate) == word_tokenize(instruction)
    original_candidate = base_candidate
    logger.info("Base Candidate:\t " + original_candidate + '\n')
    base_template = build_new_template_from_description(task_name, template, manual_detokenize(word_tokenize(instruction)))
    base_score = score(args, accelerator, base_template, dataset, model, tokenizer, metric)
    logger.info("Base Score:\t " + str(base_score) + '\n')
    delete_tracker = []
    patience_counter = 1

    use_add = True

    for i in range(num_steps):
        logger.info("Running step:\t " + str(i) + '\n')
        deleted = {}
        added = {}
        phrase_lookup = get_phrase_lookup(base_candidate, phrase_parser)  # offset_idx -> phrase/word/sentence/span
        if base_candidate == original_candidate:
            logger.info(phrase_lookup.values())   # 打印下每个phrase
        # edit_operations是所有可用的操作
        if use_add:
            if len(delete_tracker):
                if 'add' not in edit_operations: edit_operations.append('add')
            else:
                if 'add' in edit_operations: edit_operations.remove('add')
        # 从所有可选的操作中采样5个出来
        edits = np.random.choice(edit_operations, num_candidates)
        logger.info(edits)

        # generate candidates
        candidates = []
        # 遍历所有采样出来的操作
        for edit in edits:
            if isinstance(edit, str):
                logger.info("Performing edit:\t " + edit + '\n')
                # 返回操作生成的新instruction，以及操作的位置
                logger.info(f'Before {edit} operation: {base_candidate}')
                candidate, indices = perform_edit(edit, base_candidate, phrase_lookup, delete_tracker, para_model, para_tokenizer)
                logger.info("Generated candidate:\t " + candidate + '\n')
                candidates.append(candidate)
                if edit == 'del':
                    # 如果是删除操作，那么需要保存新instruction对应的删除内容
                    deleted[candidate] = [phrase_lookup[indices[0]]]
                if edit == 'add':
                    if len(indices):  # 如果是add操作并且成功了，那么保存对应的位置
                        added[candidate] = indices

        logger.info(base_score)
        scores = []
        # 遍历当前step所有生成的candidate
        valid_candidate = []
        for c, candidate in enumerate(candidates):
            option_list = template.answer_choices
            option_list = option_list.split('|||')
            template_name = template.name

            if not will_keep_with_clean_rule(task_name, template_name, manual_detokenize(word_tokenize(candidate)), option_list):
                logger.info(f'invalid candidate: {candidate}')
                continue
            valid_candidate.append(candidate)
            candidate_template = build_new_template_from_description(task_name, template, manual_detokenize(word_tokenize(candidate)))
            scores.append(score(args, accelerator, candidate_template, dataset, model, tokenizer, metric))  # 每个candidate的得分
            logger.info(scores[-1])
            logger.info("Score for Candidate " + str(c) + ":\t " + str(scores[-1]) + '\n')

        candidates = valid_candidate
        logger.info(f'valida candidate: {candidates}')
        logger.info(f'scores: {scores}')
        if len(candidates) == 0:
            continue

        best_idx = np.argmax(scores)  # 当前step得分最高的score
        best_score = scores[best_idx]
        if best_score > base_score:  # 如果得分大于base score
            patience_counter = 1
            base_candidate = candidates[best_idx]  # base_candidate变为当前step最好的candidate
            base_score = best_score
            operations_tracker.append(edits[best_idx])  # 最好candidate使用的操作
            logger.info("New Candidate Found" + '\n')
            logger.info("New Candidate Index:\t " + str(best_idx) + '\n')
            logger.info("New Candidate:\t " + base_candidate + '\n')
            logger.info("New Candidate Score:\t " + str(base_score) + '\n')
            try:
                logger.info("New Candidate Edit:\t " + edits[best_idx] + '\n')
            except:
                logger.info("New Candidate Edit:\t " + ' '.join(edits[best_idx]) + '\n')
            logger.info(f'New Base Candidate: {base_candidate}')
            if base_candidate in added.keys():
                print(f'Notice! Prev tracker: {delete_tracker}')
                for chunk in added[base_candidate]:
                    try:
                        delete_tracker.remove(chunk)
                    except:
                        pass
                print(f'Notice! New tracker: {delete_tracker}')
            if base_candidate in deleted.keys():  # 如果base_candidate是delete操作的结果
                delete_tracker.extend(deleted[base_candidate])
            base_candidate = detokenize(word_tokenize(base_candidate))
        else:
            # 如果当前step没有得到更好的结果
            patience_counter += 1

            if patience_counter > args.patience:
                print('Ran out of patience')
                logger.info('Ran out of patience \n')
                break
            else:
                continue

    if base_candidate == original_candidate:
        print('No viable candidate found!')
        logger.info('No viable candidate found!\n')
    base_template = build_new_template_from_description(task_name, template, manual_detokenize(word_tokenize(base_candidate)))
    return base_template


def build_ga_dataset(dataset_name, dataset_config_name, raw_datasets, distribution='uniform'):
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

    filtered_dataset = concatenate_datasets(ga_dataset_list)

    return filtered_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce main evaluation in T0.")

    parser.add_argument("--max_length", type=int, default=1024,
                        help=(
                            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
                            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."),
                        )
    parser.add_argument("--target_max_length", type=int, default=256,
                        help="Target max length. Sequences longer than this will be truncated.")
    parser.add_argument("--pad_to_max_length", action="store_true",
                        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
                        )
    parser.add_argument("--model_name_or_path", type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models. The list of T0 variants can be found on `https://huggingface.co/bigscience/T0_3B`",
                        required=True, )
    parser.add_argument("--config_name", type=str, default=None,
                        help="Pretrained config name or path if not the same as model_name", )
    parser.add_argument("--template_dir", type=str, default='../templates_test',
                        help="模版文件的位置", )
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Pretrained tokenizer name or path if not the same as model_name", )
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).", )
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Batch size (per device) for the evaluation dataloader.", )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--debug", action="store_true",
                        help="Activate debug mode and run training only with a subset of data.", )
    parser.add_argument("--ga_dev_distribution", type=str, choices=['uniform', 'ratio'], default='uniform',
                        help="ga_dev的分布", )
    parser.add_argument("--parallelize", action="store_true",
                        help=(
                            "If passed, will call `model.parallelize` which splits the model on all GPUs available when applicable (model parallelism). "
                            "Note that this feature is still experimental in HF Transformers."),
                        )

    # grips的参数
    parser.add_argument('--patience', default=2, type=int, help='Type in the max patience P (counter)')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    accelerator = Accelerator()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO, )
    logger.info(accelerator.state)

    # Metrics
    metric = load_metric("accuracy")

    test_task_list = ['hellaswag', 'super_glue/copa', 'anli/r1', 'anli/r2', 'anli/r3', 'super_glue/rte',
                      'super_glue/cb', 'super_glue/wsc.fixed', 'super_glue/wic', 'winogrande/winogrande_xl']
    # debug
    # test_task_list = ['super_glue/rte']

    temp = []
    for task in test_task_list:
        task_tuple = task.split('/')
        if len(task_tuple) == 2:
            temp.append(task_tuple)
        else:
            temp.append((task_tuple[0], None))
    test_task_list = temp

    # 创建output目录
    os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        # T0 model
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError("Either `args.config_name` or `args.model_name_or_path` should be provided.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name.")

    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,)
    else:
        raise NotImplementedError

    # T0必须parallel
    device = accelerator.device
    assert torch.cuda.is_available(), "You need at least 1 GPU to call `parallelize` (even though if there is only 1 GPU, there won't be any model parallelism)."
    model.parallelize()

    para_model_name = '../pegasus_paraphrase'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    para_tokenizer = PegasusTokenizer.from_pretrained(para_model_name)
    para_model = PegasusForConditionalGeneration.from_pretrained(para_model_name)
    para_model = para_model.cuda().eval()

    # phrase_parser = Parser.load('crf-con-en')
    phrase_parser = None  # level 是word就不需要这个

    for dataset_name, dataset_config_name in test_task_list:
        # 直接读json文件
        train_file_path = '../T0_dataset'
        if dataset_config_name:
            train_file_path = os.path.join(train_file_path, f'{dataset_name}_{dataset_config_name}')
        else:
            train_file_path = os.path.join(train_file_path, dataset_name)
        data_files = {
            'train': os.path.join(train_file_path, 'train.json'),
            'validation': os.path.join(train_file_path, 'validation.json')
        }
        raw_datasets = load_dataset('json', data_files=data_files)['train']
        # raw_datasets_train = raw_datasets['train']  # 用于抽取样本做in context learning
        uniq_dataset_name = f'{dataset_name}_{dataset_config_name}' if dataset_config_name else dataset_name

        logger.info(f'raw dataset after select for {dataset_name}_{dataset_config_name}: {raw_datasets}')
        column_names = raw_datasets.column_names
        logger.info(f'column name: {column_names}')

        logger.info(f'use template_dir: {args.template_dir}')

        # prompts = DatasetTemplates(
        #     f"{dataset_name}" if dataset_config_name is None else f"{dataset_name}/{dataset_config_name}",
        #     template_dir=args.template_dir)

        config_file_path = os.path.join(args.template_dir, dataset_name)
        if dataset_config_name:
            config_file_path = os.path.join(config_file_path, dataset_config_name)
        config_file_path = os.path.join(config_file_path, 'templates.yaml')
        task_templates = yaml.load(open(config_file_path, "r"), Loader=yaml.FullLoader)
        result_task_templates = copy.deepcopy(task_templates)
        result_task_templates['templates'] = {}

        template_list = task_templates['templates'].keys()
        logger.info(f'{dataset_name}的模板列表：{template_list}')

        for template_id in template_list:
            template = task_templates['templates'][template_id]
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

            if not filtered_dataset:
                filtered_dataset = raw_datasets
            filtered_dataset = build_ga_dataset(dataset_name, dataset_config_name, filtered_dataset,
                                                args.ga_dev_distribution)
            label_key = 'label'
            if dataset_name in ['winogrande']:
                label_key = 'answer'
            print(f'filtered_dataset: {filtered_dataset}')
            print(f'label distribution: {Counter(filtered_dataset[label_key])}')

            result_template = run_grips(args, accelerator, uniq_dataset_name, template, model,
                                        tokenizer, para_model, para_tokenizer, filtered_dataset, metric, phrase_parser,
                                        num_steps=10)
            if result_template is None:
                continue  # 有些template没法处理，不要了
            result_task_templates['templates'][str(uuid.uuid4())] = result_template

        config_file_path = os.path.join(args.output_dir, dataset_name)
        if dataset_config_name:
            config_file_path = os.path.join(config_file_path, dataset_config_name)
        os.makedirs(config_file_path, exist_ok=True)
        config_file_path = os.path.join(config_file_path, 'templates.yaml')
        yaml.dump(result_task_templates, open(config_file_path, "w"))


if __name__ == "__main__":
    main()
