import torch
from transformers import TextGenerationPipeline, GPT2LMHeadModel
# from cpm.tokenization_cpm import CpmTokenizer
# from transformers import TextGenerationPipeline
from transformers import GPT2LMHeadModel, PreTrainedTokenizer, PreTrainedModel, GPT2Tokenizer


def test_cpm():
    tokenizer = CpmTokenizer.from_pretrained("/mfs/shaonan/pretrained_model/CPM")
    model = GPT2LMHeadModel.from_pretrained("/mfs/shaonan/pretrained_model/CPM")
    model.parallelize()
    # text_generator = TextGenerationPipeline(model, tokenizer)
    # result = text_generator('清华大学', max_length=50, do_sample=True, top_p=0.9)
    # print(result)
    # result = text_generator('任务:请写下两个意思完全相反的句子。\n第一个句子:"最近我的加班强度很大。"\n第二个句子:"', max_length=100, do_sample=True, top_p=0.9, top_k=5)
    # print(result)
    # result = text_generator('任务:请写下两个意思完全相同的句子。\n第一个句子:"我对在上海的生活非常满意。"\n第二个句子:"', max_length=100, do_sample=True, top_p=0.9,
    #                         top_k=5)
    # print(result)
    # result = text_generator('任务:请写下两个意思相似的问题。\n第一个问题:"请判断这句话属于新闻领域中的哪个主题？"\n第二个问题:"', max_length=100, do_sample=True,
    #                         top_p=0.9,
    #                         top_k=5)
    # print(result)
    # result = text_generator('任务:请写下两个意思相似的问题。\n第一个问题:"以下文本来自新闻领域，这句话属于哪个主题？"\n第二个问题:"', max_length=100, do_sample=True,
    #                         top_p=0.9,
    #                         top_k=5)
    #
    # print(result)
    # result = text_generator('生成古诗：锄禾日当午,', max_length=50, do_sample=True, top_p=0.9, top_k=5)
    # print(result)

    text = '任务:请写下两个意思相似的问题。\n第一个问题:"请判断这句话属于新闻领域中的哪个主题？"\n第二个问题:"'
    encoded_prompt = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    print(f'encoded_prompt: {type(encoded_prompt)}, {encoded_prompt}')
    encoded_prompt.cuda()
    output = model(encoded_prompt)


def test_gpt():
    tokenizer = GPT2Tokenizer.from_pretrained('/mfs/shaonan/pretrained_model/gpt2-xl')
    model = GPT2LMHeadModel.from_pretrained('/mfs/shaonan/pretrained_model/gpt2-xl')
    # model.cuda()
    text = "{{premise}} Using only the above description and what you know about theworld, {{hypothesis}} is definitely correct, incorrect, or inconclusive?"
    text_generator = TextGenerationPipeline(model, tokenizer)
    result = text_generator(f'Task: Write two sentences that are on completely different topics.\nSentence 1: "{text}"\nSentence 2:"',
                            max_length=100, do_sample=True, top_p=0.9, top_k=5)
    print(result)
    result = text_generator(
        f'Task: Write two questions that mean the same thing.\nQuestion 1: "{text}"\nQuestion 2:"',
        max_length=100, do_sample=True, top_p=0.9, top_k=5)
    print(result)
    result = text_generator(
        f'Task: Write two questions that mean the same thing.\nQuestion 1: "{text}"\nQuestion 2:"',
        max_length=100, do_sample=True, top_p=0.9, top_k=5)

    print(result)


if __name__ == '__main__':
    # test_cpm()
    test_gpt()
