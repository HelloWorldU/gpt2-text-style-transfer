from transformers import BertTokenizer, TFGPT2LMHeadModel, DataCollatorForLanguageModeling
import tensorflow as tf
import os
import re
import matplotlib.pyplot as plt
import random 

i=1
# 绘制困惑度曲线
def perplexity_curve(perplexities, save_dir):
    global i
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(perplexities) + 1), perplexities, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Perplexity Curve')
    plt.savefig(os.path.join(save_dir,f'perplexity_curve{i+1}.png'))
    i+=1
    plt.close()

# 绘制损失值曲线
def loss_curve(train_loss,valid_loss,save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss) + 1), train_loss, marker='o', label='Train Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, marker='o', label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir,f'loss_curve{i}.png'))
    plt.close()

# 绘制准确率曲线
def accuracy_curve(train_accuracy,valid_accuracy,save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, marker='o', label='Train Accuracy')
    plt.plot(range(1, len(valid_accuracy) + 1), valid_accuracy, marker='o', label='Valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir,f'accuracy_curve{i}.png'))
    plt.close()

# 绘制学习率曲线
def learning_rate_curve(learning_rates,save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(learning_rates) + 1),learning_rates,marker='o',label='Learning Rate')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir,f'learning_rate_curve{i}.png'))
    plt.close()

# 自定义学习率调度器
class WarmUpDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_schedule_fn, warmup_steps):
        super(WarmUpDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_schedule_fn = decay_schedule_fn
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        def warmup():
            return self.initial_learning_rate * (step / self.warmup_steps)
        def decay():
            return self.decay_schedule_fn(step - self.warmup_steps)
        
        # 图模式下使用 tf.cond 进行条件判断
        return tf.cond(step < self.warmup_steps, warmup, decay)
    
# 生成文本并保存到文件
def generate_text(model, tokenizer, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="tf")
    outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 删除中文字符之间的空格
    generated_text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', generated_text)
    
    # 删除标点符号前后的空格
    generated_text = re.sub(r'\s+([。，！？：；\'\"）】》])', r'\1', generated_text)
    generated_text = re.sub(r'([\'\"（【《])\s+', r'\1', generated_text)
    
    # 删除多余的空格，但保留英文单词之间的单个空格
    generated_text = re.sub(r'\s+', ' ', generated_text).strip()
    
    # 删除非中文、英文、数字和指定标点之外的字符
    generated_text = re.sub(r'[^\w\u4e00-\u9fff。，！？""''：……]', '', generated_text)
    
    return generated_text

# 准备数据集
def load_dataset(file_path_X, file_path_Y, test_file_path, tokenizer, split_ratio=0.9, seed=None):
    if seed is not None:
        random.seed(seed)

    with open(file_path_X, 'r', encoding='utf-8') as fx:
        lines_X = fx.read().splitlines()
    random.shuffle(lines_X)
    split = int(len(lines_X) * split_ratio)
    train_lines_X = [(line, 0) for line in lines_X[:split]]  # 风格 0
    valid_lines_X = [(line, 0) for line in lines_X[split:]]

    with open(file_path_Y, 'r', encoding='utf-8') as fy:
        lines_Y = fy.read().splitlines()
    random.shuffle(lines_Y)
    split = int(len(lines_Y) * split_ratio)
    train_lines_Y = [(line, 1) for line in lines_Y[:split]]  # 风格 1
    valid_lines_Y = [(line, 1) for line in lines_Y[split:]]

    with open(test_file_path, 'r', encoding='utf-8') as f_test:
        test_lines = f_test.read().splitlines()
    test_dataset = tokenizer(test_lines, return_tensors='tf', truncation=True, padding=True, pad_to_multiple_of=8)

    train_dataset_X = tokenizer([line for line, _ in train_lines_X], return_tensors='tf', truncation=True, padding=True, pad_to_multiple_of=8)
    valid_dataset_X = tokenizer([line for line, _ in valid_lines_X], return_tensors='tf', truncation=True, padding=True, pad_to_multiple_of=8)
    train_dataset_Y = tokenizer([line for line, _ in train_lines_Y], return_tensors='tf', truncation=True, padding=True, pad_to_multiple_of=8)
    valid_dataset_Y = tokenizer([line for line, _ in valid_lines_Y], return_tensors='tf', truncation=True, padding=True, pad_to_multiple_of=8)

    train_styles_X = tf.constant([style for _, style in train_lines_X], dtype=tf.int32)
    valid_styles_X = tf.constant([style for _, style in valid_lines_X], dtype=tf.int32)
    train_styles_Y = tf.constant([style for _, style in train_lines_Y], dtype=tf.int32)
    valid_styles_Y = tf.constant([style for _, style in valid_lines_Y], dtype=tf.int32)

    return (train_dataset_X, train_styles_X), (train_dataset_Y, train_styles_Y), (
        valid_dataset_X, valid_styles_X), (valid_dataset_Y, valid_styles_Y), test_dataset


# 教师强制
def prepare_input_data(dataset, styles):
    input_ids = dataset['input_ids']
    attention_mask = dataset['attention_mask']
    labels = tf.roll(input_ids, shift=-1, axis=1)
    labels = tf.where(attention_mask == 0, -100, labels)  # 使用-100来忽略填充标记
    return input_ids, attention_mask, labels, styles

# 测试集评估函数
def test_evalution(test_step,dataset):
    total_test_loss=tf.constant(0.0,dtype=tf.float32)
    total_test_accuracy=tf.constant(0.0,dtype=tf.float32)
    num_test_batches=0
    for batch_input_ids,batch_attention_mask,batch_labels in dataset:
        loss,accuracy=test_step(batch_input_ids,batch_attention_mask,batch_labels)
        total_test_loss+=loss
        total_test_accuracy+=accuracy
        num_test_batches+=1
    avg_test_loss=total_test_loss/tf.cast(num_test_batches,tf.float32)
    avg_test_accuracy=total_test_accuracy/tf.cast(num_test_batches,tf.float32)
    test_perplexity=tf.exp(avg_test_loss).numpy()
    print(f"Test Loss: {avg_test_loss}, Test Accuracy: {avg_test_accuracy}")
    print(f"Test Perplexity: {test_perplexity}")

# KL散度正则化
def kl_divergence(p, q):
    return tf.reduce_sum(p * tf.math.log(p / q))

import tensorflow as tf
from transformers import TFBertModel, BertTokenizer

# 初始化编码器模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoder_model = TFBertModel.from_pretrained('bert-base-uncased')

# 定义计算内容分布的函数
def compute_distribution(input_ids):
    outputs = encoder_model(input_ids)
    hidden_states = outputs.last_hidden_state
    content_vector = tf.reduce_mean(hidden_states, axis=1)
    return content_vector

# 绘制生成器鉴别器的各个损失
def plot_losses(rec_losses, lm_losses, adv_losses, kl_losses, disc_losses, disc_z_losses, save_dir):
    plt.figure(figsize=(12, 8))
    plt.plot(rec_losses, label='Reconstruction Loss')
    plt.plot(lm_losses, label='Language Model Loss')
    plt.plot(adv_losses, label='Adversarial Loss')
    plt.plot(kl_losses, label='KL Divergence Loss')
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.plot(disc_z_losses, label='Discriminator Z Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'losses.png'))
    plt.close()
