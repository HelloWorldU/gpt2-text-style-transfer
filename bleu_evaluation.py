import sacrebleu

# 加载参考文本和生成文本
def load_texts(reference_path, generated_path):
    with open(reference_path, 'r', encoding='utf-8') as ref_file:
        reference_texts = ref_file.readlines()
    with open(generated_path, 'r', encoding='utf-8') as gen_file:
        generated_texts = gen_file.readlines()
    return reference_texts, generated_texts

# 计算BLEU分数
def calculate_bleu(reference_texts, generated_texts):
    # 参考文本是一个列表的列表
    references = [[ref.strip()] for ref in reference_texts]
    hypotheses = [gen.strip() for gen in generated_texts]
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    return bleu.score

if __name__ == "__main__":
    reference_path = r'C:\Users\Administrator\Desktop\大模型\落笔风雨\backend\model\corpus\clean_reference.txt'  # 参考文本文件路径
    generated_path = r'C:\Users\Administrator\Desktop\大模型\落笔风雨\backend\model\corpus\generated.txt'  # 生成文本文件路径
    reference_texts, generated_texts = load_texts(reference_path, generated_path)
    bleu_score = calculate_bleu(reference_texts, generated_texts)
    print(f"BLEU score: {bleu_score}")