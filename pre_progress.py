import re
import random
import numpy as np
import tensorflow as tf
import json
import os
from dataclasses import dataclass, field

# 文本预处理
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

# input_path = r'C:\Users\Administrator\Desktop\大模型\落笔风雨\backend\model\corpus\fanghuang_luxun.txt'
# output_path = r'C:\Users\Administrator\Desktop\大模型\落笔风雨\backend\model\corpus\Y_corpus.txt'
# input_path = r'C:\Users\Administrator\Desktop\大模型\落笔风雨\backend\model\corpus\valid_corpus.txt'
# output_path = r'C:\Users\Administrator\Desktop\大模型\落笔风雨\backend\model\corpus\clean_valid_corpus.txt'
# preprocess_text(input_path, output_path)

# 随机种子
def set_seed(seed=42) -> tuple:
    random.seed(seed)


# 训练配置
@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 8
    max_len: int = 0
    accumulation_steps: int = 2
    lambda_rec: float = 1.0
    lambda_lm: float = 1.0
    lambda_adv: float = 1.0
    lambda_kl: float = 1.0
    gamma: float = 1.0
    label_smoothing: float = 0.1
    train_losses: list = field(default_factory=list)
    train_accuracies: list = field(default_factory=list)
    rec_losses: list = field(default_factory=list)
    lm_losses: list = field(default_factory=list)
    adv_losses: list = field(default_factory=list)
    kl_losses: list = field(default_factory=list)
    disc_y_losses: list = field(default_factory=list)
    disc_z_losses: list = field(default_factory=list)
    reinforce_losses: list = field(default_factory=list)
    valid_losses: list = field(default_factory=list)
    valid_accuracies: list = field(default_factory=list)
    perplexities: list = field(default_factory=list)
    learning_rates: list = field(default_factory=list)
    lr_step: int = 0


# 读取数据集
def load_dataset(file_path_X, file_path_Y, test_file_path, tokenizer, split_ratio=0.9, seed=None):
    if seed is not None:
        random.seed(seed)

    def read_and_tokenize(file_path, style):
        with tf.io.gfile.GFile(file_path, 'r') as f:
            lines = f.read().splitlines()
        random.shuffle(lines)
        split = int(len(lines) * split_ratio)
        train_lines = lines[:split]
        valid_lines = lines[split:]
        
        train_encoded = tokenizer(train_lines, truncation=True, padding=False)
        valid_encoded = tokenizer(valid_lines, truncation=True, padding=False)

        train_dataset = (
            train_encoded['input_ids'],
            train_encoded['attention_mask'],
            [style] * len(train_lines)
        )
        valid_dataset = (
            valid_encoded['input_ids'],
            valid_encoded['attention_mask'],
            [style] * len(valid_lines)
        )
        
        return train_dataset, valid_dataset

    train_dataset_X, valid_dataset_X = read_and_tokenize(file_path_X, 0)
    train_dataset_Y, valid_dataset_Y = read_and_tokenize(file_path_Y, 1)

    with tf.io.gfile.GFile(test_file_path, 'r') as f_test:
        test_lines = f_test.read().splitlines()
    test_encoded = tokenizer(test_lines, truncation=True, padding=False)
    test_dataset = (
        test_encoded['input_ids'], 
        test_encoded['attention_mask'], 
        [0] * len(test_lines)
    )

    return (train_dataset_X, train_dataset_Y, valid_dataset_X, valid_dataset_Y, test_dataset)

def create_labels(input_ids, attention_mask):
    max_len = tf.reduce_max(tf.map_fn(lambda x: tf.shape(x)[0], input_ids, dtype=tf.int32))

    input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, maxlen=max_len, padding='post')
    attention_mask = tf.keras.preprocessing.sequence.pad_sequences(attention_mask, maxlen=max_len, padding='post')
    
    labels = tf.roll(input_ids, shift=-1, axis=1)
    labels = tf.where(attention_mask == 0, -100, labels)
    return labels

# 创建数据集
def create_tf_dataset(dataset, batch_size, drop_remainder=True, shuffle=True, shuffle_buffer_size=10000):
    input_ids, attention_mask, styles = dataset
    
    def gen():
        for i in range(len(input_ids)):
            yield {
                'input_ids': input_ids[i], 
                'attention_mask': attention_mask[i], 
                'style': styles[i]
            }    

    tf_dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature={
            'input_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'attention_mask': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'style': tf.TensorSpec(shape=(), dtype=tf.int32)
        }
    )

    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=shuffle_buffer_size)
    
    padded_shapes = {
        'input_ids': tf.TensorShape([None]),
        'attention_mask': tf.TensorShape([None]),
        'style': tf.TensorShape([])
    }
    
    tf_dataset = tf_dataset.padded_batch(
        batch_size, 
        padded_shapes=padded_shapes, 
        drop_remainder=drop_remainder
    )


    """
    Setting up an auto-sharding policy to automatically slice the dataset during distributed training.
    """
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    tf_dataset = tf_dataset.with_options(options)    

    return tf_dataset


# 张量转换
def conv_tensor_to_int(**kwargs):
    for key, value in kwargs.items():
        if isinstance(value, tf.Tensor):
            kwargs[key] = tf.cast(value, tf.float32)
    return kwargs
