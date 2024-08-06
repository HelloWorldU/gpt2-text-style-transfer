import re

def preprocess_text(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # 去除每个句子前的数字和空格
    text = re.sub(r'\d+\s*', '', text)
    
    # 去除特殊字符（保留中文、英文、数字和常用标点符号）
    text = re.sub(r'[^\w\s\u4e00-\u9fff。，！？“”‘’：……]', '', text)
    
    # 去除多余的空行
    text = re.sub(r'\n+', '\n', text)

    # length=len(text)
    # for i in range(length):
    #     if i%2!=0:
    #         text = re.sub(text[i], '', text)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)

# # 处理语料库
# input_path = r'C:\Users\Administrator\Desktop\大模型\落笔风雨\backend\model\corpus\corpus.txt'
# output_path = r'C:\Users\Administrator\Desktop\大模型\落笔风雨\backend\model\corpus\clean_corpus.txt'
# preprocess_text(input_path, output_path)

# 处理示例样本
input_path = r'C:\Users\Administrator\Desktop\大模型\落笔风雨\backend\model\corpus\reference.txt'
output_path = r'C:\Users\Administrator\Desktop\大模型\落笔风雨\backend\model\corpus\clean_reference.txt'
# input_path = r'C:\Users\Administrator\Desktop\大模型\落笔风雨\backend\model\corpus\valid_corpus.txt'
# output_path = r'C:\Users\Administrator\Desktop\大模型\落笔风雨\backend\model\corpus\clean_valid_corpus.txt'
preprocess_text(input_path, output_path)
