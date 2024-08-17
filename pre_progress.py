import re
import random
import numpy as np
import tensorflow as tf
import json
import os
from dataclasses import dataclass

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
    batch_size: int = 4
    max_len: int = 0
    accumulation_steps: int = 2
    lambda_rec: float = 1.0
    lambda_lm: float = 1.0
    lambda_adv: float = 1.0
    lambda_kl: float = 1.0
    gamma: float = 1.0
    label_smoothing: float = 0.1
    train_losses: list = None
    train_accuracies: list = None
    rec_losses: list = None
    lm_losses: list = None
    adv_losses: list = None
    kl_losses: list = None
    disc_losses: list = None
    disc_z_losses: list = None
    reinforce_losses: list = None
    valid_losses: list = None
    valid_accuracies: list = None
    perplexities: list = None
    learning_rates: list = None
    lr_step: int = 0


# 数据并行
mirrored_strategy = tf.distributed.MirroredStrategy()
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        "chief": ["host1:2222"],
        "worker": ["host2:2222", "host3:2222", "host4:2222"]
    },
    "task": {"type": "worker", "index": 1}
})

def data_parallelism(train_step, dist_inputs):
    per_replica_losses = mirrored_strategy.run(train_step, args=(dist_inputs,))
    return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

