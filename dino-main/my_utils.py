import codecs
import os
import ujson
import requests
import hashlib
import time
from openpyxl import load_workbook
from typing import List
import copy

import time  # 用于翻译出现网络错误时等待一段时间重试

import uuid

# from use_dino_to_generate import build_pattern


TRAIN_TASK = ['yf_amazon', 'JD_full', 'JD_binary', 'waimai_10k',
              'NLPCC2014_LSHT_sample', 'Chinanews', 'CNSS', 'CNSE', 'nlpcc2018_slu',
               'c3_public', 'DuReader_robust', 'DuReader_checklist', 'DuReader_yesno', 'dureader',
              'cmnli_public', 'LCQMC', 'bq_corpus', 'sohu-sts-A-sl',
              'BosonNLP_NER_6C', 'cluener_public', 'RENMIN_NER', 'LCSTS', 'NLPCC2017', 'SHENCE']

TEST_TASK = ['online_shopping_10cats', 'ChnSentiCorp_htl_all', 'nlpcc2014_task2', 'yf_dianping',
             'car_sentiment', 'dmsc', 'weibo_senti_100k', 'simplifyweibo_4', 'NLPCC2014_Weibo_Emotion_classification',
             'nCoV_100k', 'Internet_News', 'BDCI2019', 'SMP2019_ECISA',
             'THUCNews', 'CCFBDCI2020', 'tnews_public', 'Ifeng', 'nlpcc2017_news_headline_categorization',
             'catslu_traindev', 'e2e_dials', 'intent_classification',
             'cmrc2018_public', 'DRCD', 'CCF2020-BDCI-QA', 'CAIL2019-QA', 'CAIL2020-QA',
             'ocnli_public', 'nlpcc2016-dbqa',
             'CBLUE-CHIP-STS', 'CBLUE-KUAKE-QTR', 'CBLUE-KUAKE-QQR', 'afqmc_public', 'phoenix_pair',
             'sohu-sts-A-ll', 'sohu-sts-A-ss', 'sohu-sts-B-ll', 'sohu-sts-B-sl', 'sohu-sts-B-ss',
             'PAWS-X', 'msra_ner', 'weibo_ner', 'nlpcc2020-AutoIE', 'CCF2020-BDCI-NER', 'CMeEE', 'SanWen-ner',
             'NLPCC2015', 'CAIL2020', 'WANFANG', 'CSL_SUMM', 'EDU_SUMM', 'WEIBO', 'COTE-BD', 'COTE-MFW', 'COTE-DP',
             'cluewsc2020_public', 'iflytek_public'
             ]


KEEP_KEYWORD = {'评价', '怎么看', '含义', '意思', '相同', '相似', '情感', '情绪', '主题', '态度',
                '倾向', '还是', '不同', '一样', '意图', '怎么样', '如何', '是好是坏'}
BLACK_WORD = {'不属于', '：', '杀'}


def will_keep_this_description(text: str):
    # 必须以问号或者句号结尾
    if not (text.endswith('。') or text.endswith('.') or
            text.endswith('?') or text.endswith('？')):
        return False

    # 过滤英文
    for char in text:
        if char.encode().isalpha():
            if char != '?' and char != '.':
                return False

    for word in BLACK_WORD:
        if text.find(word) != -1:
            return False

    keyword_flag = False
    for word in KEEP_KEYWORD:
        if text.find(word) != -1:
            return True

    return keyword_flag


def generate_raw_description():
    """读取所有已有的任务config，把其中所有的description按行写入output文件"""
    input_file_path = '/mfs/shaonan/moonshot/megabart/tasks/task_configs'
    description_list = set()
    task_list = os.listdir(input_file_path)
    print(task_list)
    for task in task_list:
        if not task.endswith('json'):
            continue

        config_file = codecs.open(os.path.join(input_file_path, task), 'r', encoding='utf-8')
        config = ujson.load(config_file)
        task_type = config['task_type']
        if task_type not in ['textclassification', 'sentencepair']:
            continue

        patterns = config['patterns']
        for key, value in patterns.items():
            if 'task_description' in value:
                description_list.add(value['task_description'])

    output_file = codecs.open('./data/task_description.txt', 'w', encoding='utf-8')
    for description in description_list:
        output_file.write(description + '\n')


def generate_config_with_augment_description():
    origin_zh_file = codecs.open('data/task_description.txt', 'r', encoding='utf-8')
    origin_list = []
    for line in origin_zh_file.readlines():
        origin_list.append(line.strip())

    augment_file = codecs.open('data/task_description_augment_zh.txt', 'r', encoding='utf-8')
    augment_dict = {}  # key: 原始description， value: list of generated description
    idx = 0
    temp_list = []
    for line in augment_file.readlines():
        if len(line.strip()) == 0:
            augment_dict[origin_list[idx]] = temp_list
            idx += 1
            temp_list = []
        else:
            temp_list.append(line.strip())

    # 读取原始config，然后从新生成
    input_file_path = '/mfs/shaonan/moonshot/megabart/tasks/task_configs'
    task_list = os.listdir(input_file_path)
    # print(task_list)
    for task in task_list:
        # task 的config文件名
        if not task.endswith('json'):
            continue

        config_file = codecs.open(os.path.join(input_file_path, task), 'r', encoding='utf-8')
        config = ujson.load(config_file)
        task_type = config['task_type']
        task_name = config['task_name']
        if task_type not in ['textclassification', 'sentencepair']:
            continue
        patterns = config['patterns']
        if 'task_description' not in patterns['0']:
            print(f'没有task_description的任务：{config["task_name"]}')
            continue
        task_description = patterns['0']['task_description']
        task_pattern = patterns['0']['task_pattern']

        gold_description_list = []
        gold_verbalizer_list = []
        for key, value in patterns.items():
            if 'task_description' in value:
                gold_description_list.append(patterns[key]['task_description'])
                gold_verbalizer_list.append(patterns[key]['verbalizer'])

        # 用于去重的set
        temp_set = set()
        idx = 0
        augment_pattern_list = []
        new_patterns_dict = {}
        # 先把gold加入到augment_pattern_list，然后依次把generated pattern加入
        for description, verbalizer in zip(gold_description_list, gold_verbalizer_list):
            # 现添加gold description
            if description in temp_set:
                # gold_description也可能有重复
                continue
            temp_set.add(description)
            new_pattern = {}
            if task_type == 'textclassification':
                new_pattern['pattern'] = [description, "回答：[extra_id_0]。"]
            else:
                new_pattern['pattern'] = [description + '第一句话:“', '”第二句话：“', '”回答：[extra_id_0]。']
            new_pattern['verbalizer'] = verbalizer
            new_pattern['task_description'] = task_description
            new_pattern['task_pattern'] = task_pattern
            new_patterns_dict[str(idx)] = new_pattern
            idx += 1

            # 先添加各个fake pattern
            generated_description_to_process = set(augment_dict[description])
            for generated_description in generated_description_to_process:
                if generated_description in temp_set:
                    continue
                else:
                    temp_set.add(generated_description)
                if not will_keep_this_description(generated_description):
                    print(f'任务{task_name}不合理的描述: {generated_description}')
                    continue
                new_pattern = {}
                if task_type == 'textclassification':
                    new_pattern['pattern'] = [generated_description, "回答：[extra_id_0]。"]
                else:
                    new_pattern['pattern'] = [generated_description + '第一句话:“', '”第二句话：“', '”回答：[extra_id_0]。']
                new_pattern['verbalizer'] = verbalizer
                new_pattern['task_description'] = task_description
                new_pattern['task_pattern'] = task_pattern
                augment_pattern_list.append(new_pattern)

        # 把自动生成pattern也加入进来
        for augment_pattern in augment_pattern_list:
            new_patterns_dict[str(idx)] = augment_pattern
            idx += 1

        config['patterns'] = new_patterns_dict

        output_file = codecs.open(os.path.join('augment_config', task), 'w', encoding='utf-8')
        ujson.dump(config, output_file, ensure_ascii=False, indent=4)


def generate_raw_augment_description():
    """已经有了模型生成的结果（json文件）转换成文本文件"""
    augment_file = codecs.open('data/result_gpt_p0.9.json', 'r', encoding='utf-8')
    origin_file = codecs.open('data/task_description_en.txt', 'r', encoding='utf-8')

    origin_list = []
    for line in origin_file.readlines():
        origin_list.append(line.strip())

    augment_dict = {}
    for line in augment_file.readlines():
        obj = ujson.loads(line)
        text_a = obj['text_a']
        text_b = obj['text_b']
        if text_a not in augment_dict:
            augment_dict[text_a] = [text_b]
        else:
            augment_dict[text_a].append(text_b)

    output_en_file = codecs.open('data/task_description_augment_en.txt', 'w', encoding='utf-8')
    output_zh_file = codecs.open('data/task_description_augment_zh.txt', 'w', encoding='utf-8')
    for text in origin_list:
        # print(text)
        augment_list = augment_dict[text]
        # print(augment_list)
        # 去重
        augment_list = list(set(augment_list))
        for line in augment_list:
            output_en_file.write(line + '\n')
            translate_line = translate_a_line(line)
            output_zh_file.write(translate_line + '\n')
        output_en_file.write('\n')
        output_zh_file.write('\n')


def translate_a_line(text, src='en', tgt='zh-CHS'):
    YOUDAO_URL = 'https://openapi.youdao.com/api'
    APP_KEY = '5cafc4479239ee48'
    APP_SECRET = 'yO9yRKpuvIEayWswOm62XaZRbiBDYT78'

    def encrypt(signStr):
        hash_algorithm = hashlib.sha256()
        hash_algorithm.update(signStr.encode('utf-8'))
        return hash_algorithm.hexdigest()

    def truncate(q):
        if q is None:
            return None
        size = len(q)
        return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]

    def do_request(data):
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        return requests.post(YOUDAO_URL, data=data, headers=headers)

    # q = "The following two sentences come from the medical field. Do they have similar meanings?"

    error_count = 0
    while True:
        data = {}
        data['from'] = src
        data['to'] = tgt
        # data['from'] = 'zh-CHS'
        # data['to'] = 'en'
        data['signType'] = 'v3'
        curtime = str(int(time.time()))
        data['curtime'] = curtime
        salt = str(uuid.uuid1())
        signStr = APP_KEY + truncate(text) + salt + curtime + APP_SECRET
        sign = encrypt(signStr)
        data['appKey'] = APP_KEY
        data['q'] = text
        data['salt'] = salt
        data['sign'] = sign
        # en 2 zh
        # data['vocabId'] = "98CDE5ABD22A4DA3BA7E9ECB2E8931AE"
        # zh 2 en
        # data['vocabId'] = "F9DD0CD88258412A9EA6C50C577ED6F8"

        try:
            response = do_request(data)
            result = ujson.loads(response.content)
            if 'translation' in result:
                result = result['translation'][0]
                break
            else:
                return ''
        except Exception:
            if error_count == 3:
                raise AssertionError(f'多次尝试依然失败：原文：{text}')
            time.sleep(60)
            print(f'重新尝试请求接口，重试次数：{error_count}')
            error_count += 1
            continue

    return result


def state_training_summary():
    """统计训练集日志的结果"""
    summary_file = codecs.open('/mfs/xuhanwei/projects/eval_summary/train/0906/train_summary_1b_ssl-v48.txt',
                               'r', encoding='utf-8')
    config_input_dir = '/mfs/shaonan/moonshot/dino-main/augment_config'
    config_output_dir = '/mfs/shaonan/moonshot/dino-main/augment_config_v2'

    result_dict = {}

    for line in summary_file.readlines():
        if len(line.strip()) == 0:
            continue
        item = line.strip().split('\t')
        task_type = item[0]
        # 不看画像任务
        if task_type in ['portrait_classification', 'portrait_sentencepair']:
            continue

        task_name = item[1]
        pattern_id = item[3]
        result = float(item[5])

        if task_name not in result_dict:
            result_dict[task_name] = {}

        if pattern_id not in result_dict[task_name]:
            result_dict[task_name][pattern_id] = 0

        if result > result_dict[task_name][pattern_id]:
            result_dict[task_name][pattern_id] = result

    # 手写的效果是否优于自动生成的？
    for task_name in result_dict.keys():
        single_result_dict = result_dict[task_name]
        manuscript_result = 0
        manuscript_count = 0
        generated_result = 0
        generated_count = 0

        best_result = ('0', 0)
        worst_result = ('0', 999.0)
        for pattern_id, result in single_result_dict.items():
            # 统计最好和最坏结果
            if result > best_result[1]:
                best_result = (pattern_id, result)
            if result < worst_result[1]:
                worst_result = (pattern_id, result)

            if pattern_id < '3':
                manuscript_result += result
                manuscript_count += 1
            else:
                generated_result += result
                generated_count += 1

        manuscript_result = manuscript_result / manuscript_count
        generated_result = generated_result / generated_count

        print(f'{task_name}: 手写效果：{(100 * manuscript_result):.4f}, 生成效果: {(100 * generated_result):.4f}, '
              f'diff: {(100 * (manuscript_result - generated_result)):.4f}')
        print(f'{task_name}: 最好效果: {best_result[0]}-{(100 * best_result[1]):.4f}, '
              f'最差效果: {worst_result[0]}-{(100 * worst_result[1]):.4f}, diff: {(100 * (best_result[1] - worst_result[1])):.4f}')

    # 把效果最好的3个pattern挑出来
    # 先读取所有config文件
    config_file_list = os.listdir(config_input_dir)
    config_dict = {}
    for config_file_name in config_file_list:
        if not config_file_name.endswith('json'):
            continue
        config_file = codecs.open(os.path.join(config_input_dir, config_file_name), 'r', encoding='utf-8')
        config = ujson.load(config_file)
        task_name = config['task_name']
        config_dict[task_name] = config

    # 遍历result_dict，把有结果的config修改保存到v2
    for task_name in result_dict.keys():
        if task_name not in config_dict:
            print(f'找不到config文件: {task_name}')
        config = config_dict[task_name]
        all_patterns = config['patterns']
        # [(0, result), (1, result), etc...]
        single_result = list(result_dict[task_name].items())
        single_result.sort(key=lambda x: x[1], reverse=True)
        # 只使用前三个
        single_result = single_result[:3]
        selected_pattern_id = set([int(pattern_id[0]) for pattern_id in single_result])

        # 把效果最好的3个放config里
        config['patterns'] = []
        new_patterns = {}
        count = 0
        for pattern_id, pattern in all_patterns.items():
            if int(pattern_id) in selected_pattern_id:
                new_patterns[str(count)] = pattern
                count += 1

        config['patterns'] = new_patterns
        config_output_file = codecs.open(os.path.join(config_output_dir, f'{task_name}.json'),
                                         'w', encoding='utf-8')
        ujson.dump(config, config_output_file, ensure_ascii=False, indent=4)


def generate_raw_portrait_description():
    """从excel读取每个画像任务的description,并翻译生成英文plain输入文件"""
    portrait_file = load_workbook('/mfs/shaonan/moonshot/dino-main/data/portrait_cla_v2.xlsx')
    zh_output_file = codecs.open('/mfs/shaonan/moonshot/dino-main/data/portrait_description_zh.txt',
                                 'w', encoding='utf-8')
    en_output_file = codecs.open('/mfs/shaonan/moonshot/dino-main/data/portrait_description_en.txt',
                                 'w', encoding='utf-8')

    # 根据train目录下的文件名获得训练集的任务名
    # cla_config_list = os.listdir('/mfs/data/NLP_dataset/all_tasks/portrait_classification/train/hp_hn')
    # sentpair_config_list = os.listdir()

    description_set = set()
    portrait_sheet = portrait_file.active
    for row_id, row in enumerate(portrait_sheet.rows):
        if row_id == 0:
            continue
        patterns = row[3:7]

        augmented_pattern = row[7]

        if len(str(augmented_pattern.value).strip()) == 0 \
                or not augmented_pattern \
                or str(augmented_pattern.value) == 'None':
            # print(f'company: {row[1].value}, task: {row[2].value}')
            continue

        for pattern in patterns:
            # print(pattern.value)
            description_set.add(pattern.value)

    for description in description_set:
        zh_output_file.write(description + '\n')
        en_description = translate_a_line(description)
        print(f'中文：{description}, 英文：{en_description}')
        en_output_file.write(en_description + '\n')


def generate_other_raw_description():
    """给mrc_qa, summ, ner任务的description 生成相应的plain text，给dino生成"""
    base_dir = '/mfs/shaonan/moonshot/megabart/tasks/task_configs'

    # 只处理
    candidate_task = ['c3_public', 'DuReader_robust', 'DuReader_checklist', 'DuReader_yesno',
                      'dureader', 'BosonNLP_NER_6C', 'cluener_public', 'RENMIN_NER', 'LCSTS',
                      'NLPCC2017', 'SHENCE', 'cmrc2018_public', 'DRCD', 'CCF2020-BDCI-QA',
                      'CAIL2019-QA', 'CAIL2020-QA', 'msra_ner', 'weibo_ner', 'nlpcc2020-AutoIE',
                      'CCF2020-BDCI-NER', 'CMeEE', 'SanWen-ner', 'NLPCC2015', 'CAIL2020',
                      'WANFANG', 'CSL_SUMM', 'EDU_SUMM', 'WEIBO', 'COTE-BD', 'COTE-MFW', 'COTE-DP',
                      'cluewsc2020_public']

    config_file_list = os.listdir(base_dir)

    description_set = set()
    for config_file_name in config_file_list:
        config_file = codecs.open(os.path.join(base_dir, config_file_name), 'r', encoding='utf-8')
        config = ujson.load(config_file)
        task_name = config['task_name']
        if task_name not in candidate_task:
            continue
        print(f'processing {task_name}')

        pattern_list = config['patterns']
        pattern_list = [pattern for idx, pattern in pattern_list.items()]
        for pattern in pattern_list:
            if 'task_description' in pattern:
                description_set.add(pattern['task_description'])

    # description数据写到磁盘中
    raw_file = codecs.open('/mfs/shaonan/moonshot/dino-main/data/other_description.txt',
                           'w', encoding='utf-8')
    raw_en_file = codecs.open('/mfs/shaonan/moonshot/dino-main/data/other_description_en.txt',
                              'w', encoding='utf-8')
    for description in description_set:
        description_en = translate_a_line(description)
        raw_file.write(description + '\n')
        raw_en_file.write(description_en + '\n')


def translate_other_augment_result():
    augment_file = codecs.open('/mfs/shaonan/moonshot/dino-main/data/other_result_gpt_p0.8_k30.json', 'r',
                               encoding='utf-8')
    output_file = codecs.open('/mfs/shaonan/moonshot/dino-main/data/other_result_gpt_p0.8_k30_zh.json', 'w',
                              encoding='utf-8')

    for line in augment_file:
        data_obj = ujson.loads(line)
        text_b = data_obj['text_b']
        zh_result = translate_a_line(text_b)
        data_obj['text_b'] = zh_result
        output_file.write(ujson.dumps(data_obj, ensure_ascii=False) + '\n')


def will_keep_this_pattern_for_other(task_name, task_type, text):
    # 用于COTE-MFW数据集
    if task_name in ['COTE-BD', 'COTE-DP', 'COTE-MFW']:
        if text.find('描述') != -1 and text.find('对象'):
            return True
        if text.find('主体') != -1:
            return True

    if task_type == 'summarize':
        black_word = ['问题', '作者', '文件', '算法']
        for word in black_word:
            if text.find(word) != -1:
                return False

        # 三选二
        if text.find('摘要') != -1 and text.find('生成') != -1:
            return True
        if text.find('摘要') != -1 and text.find('文本') != -1:
            return True
        # if text.find('生成') != -1 and text.find('文本') != -1:
        #     return True
        return False

    if task_type == 'mrc_qa':
        black_word = ['没有答案', '采取行动']
        for word in black_word:
            if text.find(word) != -1:
                return False

        white_word = ['问题', '回答', '问答']
        for word in white_word:
            if text.find(word) != -1:
                return True
        return False

    if task_type == 'ner':
        black_word = ['没有', '不', '排除']
        for word in black_word:
            if text.find(word) != -1:
                return False

        if text.find('实体') != -1 and text.find('哪些') != -1:
            if text.find('具体') != -1 or text.find('特定') != -1:
                return True

        return False

    if task_type == 'sentencetriple':
        if text.find('指代') != -1 or text.find('消歧') != -1:
            return True
        return False

    raise NotImplementedError(f'错误的任务类型: {task_type}')


def build_pattern_for_other(task_type: str, description: str) -> List:
    """根据任务类型和description来生成pattern"""
    if task_type == 'summarize':
        if description.endswith('文本为'):
            return [description + '：', '摘要：[extra_id_0]']
        else:
            if description.endswith('。') or description.endswith('?') or description.endswith('：'):
                return ['', description + '[extra_id_0]']
            return ['', description + '：[extra_id_0]']

    if task_type == 'mrc_qa':
        return [description + '文章：', '问题：', '回答：[extra_id_0]']

    if task_type == 'ner':
        if description.find('具体') != -1:
            description_split = description.split('具体')
            return ['文本：', description_split[0], description_split[1] + "回答：[extra_id_0]。"]
        elif description.find('特定') != -1:
            description_split = description.split('特定')
            return ['文本：', description_split[0], description_split[1] + "回答：[extra_id_0]。"]
        else:
            return ["文本：", "中有哪些属于", "？回答：[extra_id_0]。"]

    raise NotImplementedError(f'错误的任务类型: {task_type}')


def generate_other_config():
    """得到了dino augment的结果（英文）生成config文件,mrc_qa, sum, ner任务"""
    augment_result = codecs.open('/mfs/shaonan/moonshot/dino-main/data/other_result_gpt_p0.8_k30_zh.json',
                                 'r', encoding='utf-8')
    en2result = dict()
    for line in augment_result:
        data_obj = ujson.loads(line)
        text_a = data_obj['text_a']
        text_b = data_obj['text_b']
        if text_a not in en2result:
            en2result[text_a.strip()] = []
        en2result[text_a.strip()].append(text_b.strip())
    # print(en2result)

    # 同时读中文和英文的description文件，方便对其
    zh_description_file = codecs.open('/mfs/shaonan/moonshot/dino-main/data/other_description.txt',
                                      'r', encoding='utf-8')
    en_description_file = codecs.open('/mfs/shaonan/moonshot/dino-main/data/other_description_en.txt',
                                      'r', encoding='utf-8')

    zh2result = dict()
    for line_zh, line_en in zip(zh_description_file, en_description_file):
        # print(f'zh: {line_zh}, en: {line_en}')
        zh2result[line_zh.strip()] = en2result[line_en.strip()]

    # 读取mrc_qa, sum, ner任务的config，然后把augment的内容拼进去。
    # 注意specific里很多都把description给删了
    base_dir = '/mfs/shaonan/moonshot/megabart/tasks/task_configs'
    config_file_list = os.listdir(base_dir)
    config_file_list = [config_file for config_file in config_file_list if config_file.endswith('json')]

    candidate_task = ['c3_public', 'DuReader_robust', 'DuReader_checklist', 'DuReader_yesno',
                      'dureader', 'BosonNLP_NER_6C', 'cluener_public', 'RENMIN_NER', 'LCSTS',
                      'NLPCC2017', 'SHENCE', 'cmrc2018_public', 'DRCD', 'CCF2020-BDCI-QA',
                      'CAIL2019-QA', 'CAIL2020-QA', 'msra_ner', 'weibo_ner', 'nlpcc2020-AutoIE',
                      'CCF2020-BDCI-NER', 'CMeEE', 'SanWen-ner', 'NLPCC2015', 'CAIL2020',
                      'WANFANG', 'CSL_SUMM', 'EDU_SUMM', 'WEIBO', 'COTE-BD', 'COTE-MFW', 'COTE-DP',
                      'cluewsc2020_public']

    for file_name in config_file_list:
        config_file = codecs.open(os.path.join(base_dir, file_name), 'r', encoding='utf-8')
        config = ujson.load(config_file)
        task_name = config['task_name']
        task_type = config['task_type']
        if task_name not in candidate_task:
            continue
        print(f'processing task: {task_name}')

        origin_pattern = config['patterns']
        origin_pattern = [pattern for idx, pattern in origin_pattern.items()]
        origin_description_set = set()
        origin_pattern_dict = dict()
        for pattern in origin_pattern:
            if 'task_description' in pattern:
                origin_description_set.add(pattern['task_description'])
                if pattern['task_description'] not in origin_pattern_dict:
                    origin_pattern_dict[pattern['task_description']] = pattern

        new_pattern_list = []
        # 先把原始的加进来(原始的specific跟description不一样，所以后面一起拼上就行)
        for origin_description in origin_description_set:
            old_pattern = origin_pattern_dict[origin_description]
            new_pattern = copy.deepcopy(old_pattern)
            if task_type == 'sentencetriple':
                new_pattern['pattern'][0] = origin_description + new_pattern['pattern'][0]
            else:
                new_pattern['pattern'] = build_pattern_for_other(task_type, origin_description)
            new_pattern_list.append(new_pattern)

        # 再把生成的加进来
        for origin_description in origin_description_set:
            generated_description = set(zh2result[origin_description])
            for description in generated_description:
                if not will_keep_this_pattern_for_other(task_name, task_type, description):
                    print(f'抛弃质量低的description: {description}')
                    continue
                # 去重
                if description in origin_description_set:
                    continue

                old_pattern = origin_pattern_dict[origin_description]
                new_pattern = copy.deepcopy(old_pattern)
                if task_type == 'sentencetriple':
                    new_pattern['pattern'][0] = description + new_pattern['pattern'][0]
                else:
                    new_pattern['pattern'] = build_pattern_for_other(task_type, description)
                new_pattern_list.append(new_pattern)

        # 新的pattern_list放入config
        output_dir = '/mfs/shaonan/moonshot/dino-main/augment_config_v2'
        config['patterns'] = dict()
        count = 0
        for new_pattern in new_pattern_list:
            config['patterns'][str(count)] = new_pattern
            count += 1

        output_file = codecs.open(os.path.join(output_dir, file_name), 'w', encoding='utf-8')
        ujson.dump(config, output_file, ensure_ascii=False, indent=4)


def append_specific_pattern():
    """把specific的几个pattern拼到augment v2每个config前面"""
    specific_base_dir = '/mfs/shaonan/moonshot/megabart/tasks/task_configs_specific'
    augment_base_dir = '/mfs/shaonan/moonshot/dino-main/augment_config_v2'

    specific_config_list = os.listdir(specific_base_dir)
    augment_config_list = os.listdir(augment_base_dir)
    specific_config_list = [config_file for config_file in specific_config_list if config_file.endswith('json')]
    augment_config_list = [config_file for config_file in augment_config_list if config_file.endswith('json')]

    for file_name in augment_config_list:
        if file_name not in specific_config_list:
            continue

        augment_config = ujson.load(codecs.open(os.path.join(augment_base_dir, file_name), 'r', encoding='utf-8'))
        specific_config = ujson.load(codecs.open(os.path.join(specific_base_dir, file_name), 'r', encoding='utf-8'))

        specific_pattern = specific_config['patterns']
        specific_pattern = [pattern for idx, pattern in specific_pattern.items()]

        specific_pattern = specific_pattern[:3]

        augment_pattern = augment_config['patterns']
        augment_pattern = [pattern for idx, pattern in augment_pattern.items()]

        new_pattern_list = specific_pattern + augment_pattern

        augment_config['patterns'] = dict()

        count = 0
        for new_pattern in new_pattern_list:
            augment_config['patterns'][str(count)] = new_pattern
            count += 1

        # 先用temp文件夹
        output_dir = '/mfs/shaonan/moonshot/dino-main/v2_temp'
        with codecs.open(os.path.join(output_dir, file_name), 'w', encoding='utf-8') as output_file:
            ujson.dump(augment_config, output_file, ensure_ascii=False, indent=4)


def _unify_verbalizer():
    """把v2的vervalizer统一一下, 目前所有config已经在开头加入specific"""
    base_dir = '/mfs/shaonan/moonshot/dino-main/augment_config_v2'

    config_file_list = os.listdir(base_dir)
    config_file_list = [config_file for config_file in config_file_list if config_file.endswith('json')]

    for file_name in config_file_list:
        config = ujson.load(codecs.open(os.path.join(base_dir, file_name), 'r', encoding='utf-8'))

        patterns = config['patterns']
        if 'verbalizer' not in patterns['0']:
            continue
        verbalizer = patterns['0']['verbalizer']
        for idx, pattern in patterns.items():
            patterns[idx]['verbalizer'] = verbalizer

        with codecs.open(os.path.join(base_dir, file_name), 'w', encoding='utf-8') as output_file:
            ujson.dump(config, output_file, ensure_ascii=False, indent=4)


def _fix_qa_config():
    """qa config 有点问题，应该是3段的变成4段了"""
    base_dir = '/mfs/shaonan/moonshot/dino-main/augment_config_v2'

    config_file_list = os.listdir(base_dir)
    config_file_list = [config_file for config_file in config_file_list if config_file.endswith('json')]

    for file_name in config_file_list:
        config = ujson.load(codecs.open(os.path.join(base_dir, file_name), 'r', encoding='utf-8'))
        task_type = config['task_type']
        if task_type != 'mrc_qa':
            continue

        patterns = config['patterns']
        for idx, pattern in patterns.items():
            pattern_list = pattern['pattern']
            if len(pattern_list) == 4:
                new_pattern_list = [pattern_list[0] + pattern_list[1], pattern_list[2], pattern_list[3]]
                pattern['pattern'] = new_pattern_list
            elif len(pattern_list) > 4:
                raise NotImplementedError(f'pattern: {pattern_list}')

        with codecs.open(os.path.join(base_dir, file_name), 'w', encoding='utf-8') as output_file:
            ujson.dump(config, output_file, ensure_ascii=False, indent=4)


def _fix_ner_config():
    """qa config 有点问题，应该是3段的变成4段了"""
    base_dir = '/mfs/shaonan/moonshot/dino-main/augment_config_v2'

    config_file_list = os.listdir(base_dir)
    config_file_list = [config_file for config_file in config_file_list if config_file.endswith('json')]

    for file_name in config_file_list:
        config = ujson.load(codecs.open(os.path.join(base_dir, file_name), 'r', encoding='utf-8'))
        task_type = config['task_type']
        if task_type != 'ner':
            continue

        patterns = config['patterns']
        for idx, pattern in patterns.items():
            pattern_list = pattern['pattern']
            if len(pattern_list) == 4:
                new_pattern_list = [pattern_list[0], pattern_list[1], pattern_list[2] + pattern_list[3]]
                pattern['pattern'] = new_pattern_list
            elif len(pattern_list) > 4:
                raise NotImplementedError(f'pattern: {pattern_list}')

        with codecs.open(os.path.join(base_dir, file_name), 'w', encoding='utf-8') as output_file:
            ujson.dump(config, output_file, ensure_ascii=False, indent=4)


def _fix_phoenix_config():
    base_dir = '/mfs/shaonan/moonshot/dino-main/augment_config_v2'
    config = ujson.load(codecs.open(os.path.join(base_dir, 'phoenix_pair.json'), 'r', encoding='utf-8'))
    patterns = config['patterns']
    for idx, pattern in patterns.items():
        pattern['verbalizer'] = {"相关": "相似", "不相关": "不同"}

    with codecs.open(os.path.join(base_dir, 'phoenix_pair.json'), 'w', encoding='utf-8') as output_file:
        ujson.dump(config, output_file, ensure_ascii=False, indent=4)


def will_keep_with_more_clean_rule(task_name, task_type, description):
    SENTIMENT = ['yf_amazon', 'JD_full', 'JD_binary', 'waimai_10k', 'online_shopping_10cats',
                 'ChnSentiCorp_htl_all', 'nlpcc2014_task2', 'yf_dianping',
                 'car_sentiment', 'dmsc', 'weibo_senti_100k', 'simplifyweibo_4',
                 'NLPCC2014_Weibo_Emotion_classification',
                 'nCoV_100k', 'Internet_News', 'BDCI2019', 'SMP2019_ECISA']
    NEWS = ['NLPCC2014_LSHT_sample', 'Chinanews', 'CNSS',
            'CNSE', 'THUCNews', 'CCFBDCI2020', 'tnews_public', 'Ifeng', 'nlpcc2017_news_headline_categorization']
    INTENT = ['nlpcc2018_slu', 'catslu_traindev', 'e2e_dials', 'intent_classification', ]
    PAIR = ['cmnli_public', 'LCQMC', 'bq_corpus', 'sohu-sts-A-ll',
            'sohu-sts-A-sl', 'sohu-sts-A-ss',
            'ocnli_public', 'nlpcc2016-dbqa', 'CBLUE-CHIP-STS', 'CBLUE-KUAKE-QTR', 'CBLUE-KUAKE-QQR',
            'afqmc_public', 'phoenix_pair', 'sohu-sts-A-ll', 'sohu-sts-A-ss', 'sohu-sts-B-ll',
            'sohu-sts-B-sl', 'sohu-sts-B-ss', 'PAWS-X']
    QA = ['c3_public', 'DuReader_robust', 'DuReader_checklist', 'DuReader_yesno', 'dureader',
          'cmrc2018_public', 'DRCD', 'CCF2020-BDCI-QA', 'CAIL2019-QA', 'CAIL2020-QA']
    NER = ['BosonNLP_NER_6C', 'cluener_public', 'RENMIN_NER',
           'msra_ner', 'weibo_ner', 'nlpcc2020-AutoIE', 'CCF2020-BDCI-NER', 'CMeEE', 'SanWen-ner']
    SUMMARY = ['LCSTS', 'NLPCC2017', 'SHENCE',
               'NLPCC2015', 'CAIL2020', 'WANFANG', 'CSL_SUMM', 'EDU_SUMM', 'WEIBO'
               ]
    KEYWORD_EXTRACT = ['COTE-BD', 'COTE-MFW', 'COTE-DP']
    RESOLUTION = ['cluewsc2020_public']

    # 情感分类
    if task_name in SENTIMENT:
        BLOCK_WORD = ['不同', '最重要', '偏向产品', '两者都有', '三个特点', '不同的意思',
                      '审查', '没有帮助', '伤害', '好主意', '更适合', '产品的关系', '考虑的因素']
        for word in BLOCK_WORD:
            if description.find(word) != -1:
                return False
        return True
    if task_name in NEWS:
        BLOCK_WORD = ['是什么意思', '例子']
        for word in BLOCK_WORD:
            if description.find(word) != -1:
                return False
        return True

    if task_name in INTENT:
        BLOCK_WORD = ['撞倒行人', '迎面而来', '是什么意思']
        for word in BLOCK_WORD:
            if description.find(word) != -1:
                return False
        return True
    if task_name in PAIR:
        # TODO：训练的时候没见过QApair
        pass
    if task_name in QA:
        BLOCK_WORD = ['相同的风格', '域名', '最佳方法', '目的是什么', '英语', '死亡原因',
                      '最常见的原因', '北极星', '基督徒', '同样的问题', '写一个问题', '有什么不同',
                      '邮件地址', '区别']
        for word in BLOCK_WORD:
            if description.find(word) != -1:
                return False
        return True
    if task_name in NER:
        pass
    if task_name in SUMMARY:
        BLOCK_WORD = ['程序生成', '有什么区别', '文章列表']
        for word in BLOCK_WORD:
            if description.find(word) != -1:
                return False
        return True

    return True


def get_description_from_pattern(task_type, pattern):
    """"根据不同的任务类型提取其description"""
    description = None
    if task_type == 'textclassification':
        description = pattern['pattern'][0]
    elif task_type == 'sentencepair':
        description = pattern['pattern'][0]
        description = description.replace('第一句话:“', '')
    elif task_type == 'ner':
        description = pattern['pattern'][1] + '特定' + pattern['pattern'][2]
        description = description.replace('回答：[extra_id_0]。', '')
    elif task_type == 'mrc_qa':
        description = pattern['pattern'][0]
        description = description.replace('文章：', '')
    elif task_type == 'summarize':
        # print(pattern)
        description = pattern['pattern'][1]
        description = description.replace('[extra_id_0]', '')
        description = description.replace(':：', '：')
    elif task_type == "sentencetriple":
        description = pattern['pattern'][0]
        description = description.replace('对于句子：', '')
        description = description.replace('句子：', '')
    else:
        raise NotImplementedError(f'没有这个任务: {task_type}, {pattern}')

    return description


def more_clean_rule_for_v2():
    base_dir = '/mfs/shaonan/moonshot/dino-main/augment_config_v2'
    output_dir = '/mfs/shaonan/moonshot/dino-main/augment_config_v2_clean'
    config_file_list = os.listdir(base_dir)

    for file_name in config_file_list:
        config = ujson.load(codecs.open(os.path.join(base_dir, file_name), 'r', encoding='utf-8'))
        task_name = config['task_name']
        task_type = config['task_type']

        # if task_name not in TRAIN_TASK and task_name not in TEST_TASK:
        #     continue

        # 根据不同的细分任务类型进行过滤:
        patterns = list(config['patterns'].values())
        new_pattern_list = patterns[:3]
        for pattern in patterns[3:]:
            description = get_description_from_pattern(task_type, pattern)
            if not will_keep_with_more_clean_rule(task_name, task_type, description):
                print(f'丢弃{task_name}任务中的pattern: {description}')
                continue
            else:
                new_pattern_list.append(pattern)

        config['patterns'] = dict()
        idx = 0
        for pattern in new_pattern_list:
            config['patterns'][str(idx)] = pattern
            idx += 1

        with codecs.open(os.path.join(output_dir, file_name), 'w', encoding='utf-8') as output_file:
            ujson.dump(config, output_file, ensure_ascii=False, indent=4)


# def generate_random_prompt():
#     """生成随机pattern的prompt，用来遗传算法总可以得到一个不错的结果"""
#     input_dir = ''
#     output_dir = ''
#
#     config_file_list = os.listdir(input_dir)
#     config_file_list = [file_name for file_name in config_file_list if file_name.endswith('json')]
#
#     for file_name in config_file_list:
#         config = ujson.load(codecs.open(os.path.join(input_dir, file_name), 'r', encoding='utf-8'))
#         print(f'Reading config file: {file_name}')
#
#         task_name = config['task_name']
#         task_type = config['task_type']
#
#         if task_name not in TEST_TASK:
#             continue
#
#         # 不修改ner任务，因为限制的比较死，必须有"特定"
#         if task_type == 'ner':
#             # TODO： 输出
#             continue
#
#         patterns_list = config['patterns']
#         patterns_list = [pattern for pattern_id, pattern in patterns_list.items()]
#         patterns_list = patterns_list[:3]
#
#         new_pattern_list = []
#         for pattern in patterns_list:
#             # origin_description = get_description_from_pattern(task_name, task_type, pattern)
#
#             # 获取一个随机的句子作为description， 不修改NER
#             # random_description = get_a_random_description()
#             random_description = ''
#
#             augment_pattern_list = build_pattern(task_name, task_type, random_description)
#
#             new_pattern = copy.deepcopy(pattern)
#             pattern['pattern']



def count_chinese_ga_dev_token_num():
    """统计ga算法使用dev集的token数和ga_dev的token数，用于估算算法的计算量"""

    token_count = 0

    dev_file = 'dev'

    base_dir = '/mfs/data/NLP_dataset/all_tasks'

    for task in TEST_TASK:
        task_dir = os.path.join(base_dir, task)
        input_file = codecs.open(os.path.join(task_dir, f'{dev_file}.json'), 'r', encoding='utf-8')



    pass






if __name__ == '__main__':
    # generate_raw_description()
    # generate_raw_augment_description()
    # generate_config_with_augment_description()
    # state_training_summary()
    # generate_raw_portrait_description()
    # generate_other_raw_description()
    # translate_other_augment_result()
    # generate_other_config()
    # append_specific_pattern()
    # _unify_verbalizer()
    # _fix_qa_config()
    # _fix_ner_config()
    # _fix_phoenix_config()
    more_clean_rule_for_v2()
