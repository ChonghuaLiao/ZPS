import argparse
import copy
import json
import os
import re
import ujson
import codecs
from datetime import datetime
from typing import Any, Dict

from modeling import GPT2Wrapper, DinoGenerator, PLACEHOLDER_STR
from utils import set_seed, read_inputs, DatasetEntry

from my_utils import translate_a_line, will_keep_with_more_clean_rule


def norm_punc(text: str):
    """转化英文标点为中文标点"""
    text = text.replace(':', '：')
    text = text.replace(',', '，')
    text = text.replace('.', '。')
    text = text.replace('""', '“”')
    text = text.replace('(', '（')
    text = text.replace(')', '）')
    text = text.replace(';', '；')
    text = text.replace('?', '？')
    return text


def will_keep_with_clean_rule(task_name, task_type, description):
    if not will_keep_with_more_clean_rule(task_name, task_type, description):
        return False

    # 长度控制:
    if len(description) < 2:
        return False

    # 过滤英文
    for char in description:
        if char.encode().isalpha():
            if char != '?' and char != '.':
                return False

    # 通用黑名单
    BLACK_WORD = ['不属于', '：', '杀']

    if task_type == 'textclassification':
        pass
    elif task_type == 'sentencepair':
        pass
    elif task_type == 'ner':
        BLACK_WORD += ['没有', '不', '排除']
        description = description.replace('具体', '特定')

        # 必须包含这个词
        if description.find('特定') == -1:
            return False

    elif task_type == 'mrc_qa':
        BLACK_WORD += ['没有答案', '采取行动']

    elif task_type == 'summarize':
        BLACK_WORD += ['问题', '作者', '文件', '算法']

    elif task_type == "sentencetriple":
        pass
    else:
        # raise NotImplementedError(f'没有这个任务: {task_name}, {description}')
        pass

    for word in BLACK_WORD:
        if description.find(word) != -1:
            return False
    return True


def get_description_from_pattern(task_name, task_type, pattern):
    """"根据不同的任务类型提取其description"""
    if task_type == 'textclassification':
        # pattern_str = ''.join(pattern['pattern'])
        pattern_str = ''
        for part in pattern['pattern']:
            if part.endswith('。新闻：') or part.endswith('，新闻：') or part.endswith('？新闻：'):
                part = part.replace('。新闻：', '')
                part = part.replace('，新闻：', '')
                part = part.replace('？新闻：', '')
                pattern_str += part
            elif part.endswith('。句子：'):
                part = part.replace('。句子：', '')
                pattern_str += part
            elif part.endswith('：'):
                pattern_str += part[:-1]
            else:
                pattern_str += part

        pattern_str = pattern_str.replace('“”', '')

        pattern_str = pattern_str.replace('[extra_id_0]。', '')
        pattern_str = pattern_str.replace('[extra_id_0]的。', '')
        pattern_str = pattern_str.replace('[extra_id_0]', '')
        pattern_str = pattern_str.replace('回答：', '')

        description = pattern_str
    elif task_type == 'sentencepair':
        # 注意：tnews_public是新闻分类任务
        # 需要找到最长的part
        print(f'task_name: {task_name}')

        if task_name in ['tnews_public', 'Ifeng']:
            max_len = 0
            pattern_str = ''
            for part in pattern['pattern']:
                if len(part) > max_len:
                    pattern_str = part
                    max_len = len(part)
            if pattern_str.endswith('。新闻：') or pattern_str.endswith('，新闻：') \
                    or pattern_str.endswith('？新闻：'):
                pattern_str = pattern_str.replace('。新闻：', '')
                pattern_str = pattern_str.replace('，新闻：', '')
                pattern_str = pattern_str.replace('？新闻：', '')
        else:
            pattern_str = ''.join(pattern['pattern'])

        pattern_str = pattern_str.replace('第一句话：“”第二句话：“”的', '这两句话')
        pattern_str = pattern_str.replace('“”', '')
        pattern_str = pattern_str.replace('“”', '')
        pattern_str = pattern_str.replace('第一句话：', '')
        pattern_str = pattern_str.replace('第二句话：', '')
        pattern_str = pattern_str.replace('。。', '。')
        pattern_str = pattern_str.replace('[extra_id_0]。', '')
        pattern_str = pattern_str.replace('回答：', '')

        if pattern_str.startswith('。'):
            pattern_str = pattern_str[1:]
        description = pattern_str

        # debug
        # print(f'task_name: {task_name}, description: {description}')

    elif task_type == 'ner':
        pattern_str = ''
        first_part: str = pattern['pattern'][0]
        if first_part.endswith('：'):
            first_part = first_part[:-1]
        pattern_str += first_part

        second_part: str = pattern['pattern'][1]
        if not second_part.startswith('中'):
            second_part = '中' + second_part
        pattern_str += second_part
        pattern_str += '特定类别的实体'
        pattern_str += pattern['pattern'][2]
        pattern_str = pattern_str.replace('“”', '')
        pattern_str = pattern_str.replace('[extra_id_0]。', '')
        pattern_str = pattern_str.replace('回答：', '')
        description = pattern_str

    elif task_type == 'mrc_qa':
        pattern_str = ''.join(pattern['pattern'])
        pattern_str = pattern_str.replace('“”', '')
        pattern_str = pattern_str.replace('[extra_id_0]。', '')
        pattern_str = pattern_str.replace('文章：', '')
        pattern_str = pattern_str.replace('问题：', '')
        pattern_str = pattern_str.replace('回答：', '')
        description = pattern_str
    elif task_type == 'summarize':
        pattern_str = ''.join(pattern['pattern'])
        pattern_str = pattern_str.replace('“”', '')
        pattern_str = pattern_str.replace('[extra_id_0]。', '')
        pattern_str = pattern_str.replace('[extra_id_0]', '')
        description = pattern_str
    elif task_type == "sentencetriple":
        # 指代消岐任务, pattern太复杂了，很难augment
        description = ''
        first_part = pattern['pattern'][0]
        if first_part.endswith('：'):
            description += first_part[:-1]
        else:
            description += first_part

    elif task_type == 'portrait_classification':
        description = pattern['pattern'][2]
        end_pos = description.find('回答：')
        if end_pos == -1:
            end_pos = description.find('回答:')

        assert end_pos != -1, f'错误：pattern: {pattern["pattern"]}, description: {description}'

        description = description[:end_pos]

    elif task_type == 'portrait_sentencepair':
        description = pattern['pattern'][0]
        # 定位description的开始和结束位置
        start_pos = description.find('【')
        end_pos = description.find('】')
        description = description[start_pos + 1: end_pos]

    else:
        # raise NotImplementedError(f'没有这个任务: {task_type}, {pattern}')
        print(f'WARNING: 没有这个任务: {task_type}, {pattern}')

    print(f'原始pattern: {pattern}')
    print(f'提取到的description: {description}')
    description = description.replace('[extra_id_0]', '')
    return description


def build_pattern(task_name, task_type, augment_description):
    """根据augment的结果来生成pattern, 返回的是列表"""
    # 注意一个
    augment_pattern_list = []
    if task_type == 'textclassification':
        # 答案在前和答案在后
        augment_pattern_list.append(["“", "”" + augment_description + '回答：[extra_id_0]。'])

        # 答案在前
        augment_pattern_list.append([augment_description + '回答：[extra_id_0]。' + '“', '”'])

    elif task_type == 'sentencepair':
        # TODO: nlpcc2016-dbqa结果可能会有问题
        if task_name == ['tnews_public', 'Ifeng']:
            # 答案在前
            augment_pattern_list.append([augment_description + '[extra_id_0]。新闻：', '关键词：', ""])
            augment_pattern_list.append(["新闻内容：", "关键词：", augment_description + "[extra_id_0]"])
            augment_pattern_list.append(["根据这段新闻：", "和关键词：", augment_description + "？回答：[extra_id_0]。"])

        else:
            # 答案在后
            augment_pattern_list.append([augment_description + "第一句话：“", "”第二句话：“", "”回答：[extra_id_0]。"])
            # 答案在前
            augment_pattern_list.append([augment_description + "回答：[extra_id_0]。第一句话：“", "”第二句话：“", "”"])
            # 答案在中间
            augment_pattern_list.append(["第一句话：“", "”。" + augment_description + "[extra_id_0]。第二句话：“", "”。"])

    elif task_type == 'ner':
        augment_description = augment_description.replace('具体', '特定')

        split_description = augment_description.split('特定')

        assert len(split_description) >= 2, f'{split_description}'

        augment_pattern_list.append(['文本：', split_description[0], split_description[1] + '回答：[extra_id_0]。'])
        augment_pattern_list.append(['回答：[extra_id_0]。文本', split_description[0], split_description[1]])

    elif task_type == 'mrc_qa':
        augment_pattern_list.append([augment_description + "文章：", "问题：", "回答：[extra_id_0]。"])
        augment_pattern_list.append(["回答：[extra_id_0]。" + augment_description, "问题：", ""])

    elif task_type == 'summarize':
        augment_pattern_list.append(["", augment_description + "[extra_id_0]"])
        augment_pattern_list.append([augment_description, "[extra_id_0]"])

    elif task_type == "sentencetriple":
        # 指代消岐
        augment_pattern_list.append([augment_description, "其中：", "指的[extra_id_0]", ""])
        augment_pattern_list.append([augment_description, "问：", "是指代", "吗？回答：[extra_id_0]"])
        augment_pattern_list.append([augment_description, "代词：", "指代的是：", "吗？回答：[extra_id_0]。"])
    elif task_type == 'portrait_classification':
        augment_pattern_list.append(["这句话：", "。", augment_description + "回答：[extra_id_0]。"])
    elif task_type == 'portrait_sentencepair':
        augment_pattern_list.append(["在【" + augment_description + "】主题下，“", "”和“", "”是相似的意思吗？回答：[extra_id_0]。"])
    else:
        raise NotImplementedError(f'没有这个任务: {task_type}, {augment_description}')

    return augment_pattern_list


def read_input_config(args, input_dir):
    """读取input目录下的config文件, 返回：
        1. config json dict
        2. 中文description list
        3. 英文description list
    """
    config_file_list = os.listdir(input_dir)

    config_file_list = [file_name for file_name in config_file_list if file_name.endswith('json')]

    config_dict = dict()
    description_list = []
    for file_name in config_file_list:
        config = ujson.load(codecs.open(os.path.join(input_dir, file_name), 'r', encoding='utf-8'))
        print(f'Reading config file: {file_name}')

        task_name = config['task_name']
        task_type = config['task_type']

        # 不使用画像:
        # if task_type in ['portrait_sentencepair', 'portrait_classification']:
        #     continue

        # 对画像任务做特别处理
        if task_type in ['portrait_sentencepair', 'portrait_classification']:
            task_name = file_name.replace('.json', '')

        patterns_list = config['patterns']
        patterns_list = [pattern for pattern_id, pattern in patterns_list.items()]

        # 只使用前3个pattern
        patterns_list = patterns_list[:args.Top_N_patterns]

        config_dict[task_name] = config
        description_list.extend([get_description_from_pattern(task_name, task_type, pattern)
                                 for pattern in patterns_list])

    en_description_list = []
    for description in description_list:

        description_en = translate_a_line(description, src='zh-CHS', tgt='en')
        if len(description_en) == 0:
            print(f'警告：返回一个空的翻译结果：原文：{description_en}')
        en_description_list.append(description_en)
        print(f'中文：{description}, 英文：{description_en}')

    assert len(description_list) == len(en_description_list)

    return config_dict, description_list, en_description_list


def build_output_config(args, origin_config_dict, all_description_list, all_en_description_list,
                        output_entry_list):
    """根据dino生成的结果生成config, 这里主要是些"""

    en_augment_dict = {}
    # 防止有个别description生成的结果都一样导致被去重去掉，出现keyerror
    for en_description in en_description_list:
        en_augment_dict[en_description] = []

    for entry in output_entry_list:
        text_a = entry.text_a
        text_b = entry.text_b

        if text_a not in en_augment_dict:
            en_augment_dict[text_a] = [text_b]
        else:
            en_augment_dict[text_a].append(text_b)

    augment_dict = dict()
    # 得到中文augment结果
    for description, en_description in zip(all_description_list, all_en_description_list):
        augment_result = en_augment_dict[en_description]

        # 去重
        augment_result = list(set(augment_result))

        augment_dict[description] = []
        for result in augment_result:
            if len(result) < 2:
                continue
            result_zh = translate_a_line(result, src='en', tgt='zh-CHS')
            # response没有translation字段时，会返回空字符串
            if len(result_zh) < 1:
                continue
            result_zh = norm_punc(result_zh)
            augment_dict[description].append(result_zh)

    # 生成新的config文件
    for task_name, config in origin_config_dict.items():

        task_type = config['task_type']

        patterns_list = config['patterns']
        patterns_list = [pattern for pattern_id, pattern in patterns_list.items()]

        # 只用前3个
        patterns_list = patterns_list[:args.Top_N_patterns]

        # new_pattern_list = copy.deepcopy(patterns_list)
        new_pattern_list = []

        # 用于去重
        exist_description = set()

        for pattern in patterns_list:
            origin_description = get_description_from_pattern(task_name, task_type, pattern)

            # origin早就加进new pattern list了
            # if origin_description not in exist_description:
            #     augment_pattern_list = build_pattern(task_name, task_type, origin_description)
            #     for augment_pattern in augment_pattern_list:
            #         new_pattern = copy.deepcopy(pattern)
            #         new_pattern['patter'] = augment_pattern
            #         new_pattern_list.append(new_pattern)

            # exist_description.add(origin_description)

            # debug:
            # print(f'augment_dict: {augment_dict}')

            augment_result = augment_dict[origin_description]

            # debug
            # print(f'augment_result: {augment_result}')

            # 规则过滤
            augment_result = [augment_description for augment_description in
                              augment_result if
                              will_keep_with_clean_rule(task_name, task_type, augment_description)]

            for augment_description in augment_result:
                if augment_description in exist_description:
                    continue
                augment_pattern_list = build_pattern(task_name, task_type, augment_description)

                # debug
                # print(f'augment_pattern_list: ')
                # print(augment_pattern_list)

                for augment_pattern in augment_pattern_list:
                    new_pattern = copy.deepcopy(pattern)
                    new_pattern['pattern'] = augment_pattern
                    new_pattern_list.append(new_pattern)

                exist_description.add(augment_description)

        # 生成新的config，并输出
        new_config = copy.deepcopy(config)
        new_config['patterns'] = dict()
        idx = 0
        for pattern in new_pattern_list:
            new_config['patterns'][str(idx)] = pattern
            idx += 1

        if task_type in []:
            pass

        output_file = codecs.open(os.path.join(args.output_dir, f'{task_name}.json'), 'w',
                                  encoding='utf-8')

        ujson.dump(new_config, output_file, ensure_ascii=False, indent=4)
        output_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_dir", type=str, required=True,
                        help="需要augment的config文件")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory to which the generated dataset is saved")
    parser.add_argument("--task_file", type=str,
                        default='/mfs/shaonan/moonshot/dino-main/task_specs/generate_task_description_en_v2.json',
                        help="A json file providing the instructions and other information required for dataset generation. "
                             "See the 'task_specs' directory for examples and 'README.md' for more details on how to create this file.")

    parser.add_argument("--model_name", type=str, default="/data1/ygwang/pretrained_model/gpt2-xl",
                        help="The pretrained model to use for dataset generation. Currently, only variants of GPT2 are supported.")

    parser.add_argument("--max_output_length", type=int, default=40,
                        help="The maximum output length for each generated text.")
    parser.add_argument("--decay_constant", type=float, default=100,
                        help="The decay constant for self-debiasing")
    parser.add_argument("--top_p", type=float, default=0.8,
                        help="p value for top-p sampling (set to 0 to perform no top-p sampling)")
    parser.add_argument("--top_k", type=int, default=0,
                        help="k value for top-k sampling (set to 0 to perform no top-k sampling)")

    parser.add_argument("--num_entries_per_input_and_label", type=int, default=30,
                        help="The number of entries to generate for each pair of input text and label (only if --input_file is set)")
    parser.add_argument("--num_entries_per_label", type=int, default=None,
                        help="The number of entries to generate for each label (only if --input_file is not set)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="The batch size for generation (only if --input_file is not set)")
    parser.add_argument("--remove_duplicates", action='store_true',
                        help="Whether duplicates should be removed from the generated dataset")
    parser.add_argument("--remove_identical_pairs", action='store_true',
                        help="Whether text pairs with text_a == text_b should be removed from the dataset (only for text pair datasets)")

    parser.add_argument("--keep_outputs_without_eos", action='store_true',
                        help="If set to true, examples where the language model does not output a quotation mark ("
                             "which is interpreted as "
                             "a signal that it has completed its output) are not removed from the dataset.")
    parser.add_argument("--allow_newlines_in_outputs", action='store_true',
                        help="If set to true, model outputs that contain a newline character before the end-of-sequence token (a quotation "
                             "mark) are not removed from the dataset.")
    parser.add_argument("--min_num_words", type=int, default=-1,
                        help="The minimum number of (whitespace-separated) words for each dataset entry. Entries with fewer words are "
                             "removed.")
    parser.add_argument("--min_num_tokens", type=int, default=-1,
                        help="The minimum number of tokens for each dataset entry. Entries with fewer tokens are removed.")

    parser.add_argument("--Top_N_patterns", type=int, default=6,
                        help="使用top n 个pattern进行argument")

    # Miscellaneous further parameters
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)
    args.date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"Parameters: {args}")

    args.remove_identical_pairs = True
    # 没有用的参数，单纯防止报错
    args.input_file_type = 'plain'
    args.openai_api_key = None

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.task_file, 'r', encoding='utf8') as fh:
        # 提供instruction的文件
        task_specification = json.load(fh)
        # validate_task_spec(task_specification, with_inputs=args.input_file is not None)

    # 根据input目录读需要augment的config文件
    print(f'Reading input config file: {args.input_dir}')
    config_dict, description_list, en_description_list = read_input_config(args, args.input_dir)

    inputs = en_description_list
    model = GPT2Wrapper(model_name=args.model_name,
                        use_cuda=not args.no_cuda) if not args.openai_api_key else args.model_name
    generator = DinoGenerator(
        task_spec=task_specification, model=model, openai_api_key=args.openai_api_key,
        max_output_length=args.max_output_length,
        decay_constant=args.decay_constant, top_p=args.top_p, top_k=args.top_k,
        remove_duplicates=args.remove_duplicates,
        remove_identical_pairs=args.remove_identical_pairs, min_num_words=args.min_num_words,
        min_num_tokens=args.min_num_tokens,
        keep_outputs_without_eos=args.keep_outputs_without_eos, allow_newlines_in_outputs=args.allow_newlines_in_outputs
    )

    print("Starting dataset generation with DINO...")
    outputs = generator.generate_dataset(inputs, num_entries_per_input_and_label=args.num_entries_per_input_and_label,
                                         num_entries_per_label=args.num_entries_per_label, batch_size=args.batch_size)

    build_output_config(args, config_dict, description_list, en_description_list, outputs)
