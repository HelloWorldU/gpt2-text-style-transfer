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
def set_seed(seed=42) -> None:
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
    rec_losses: list = None
    lm_losses: list = None
    adv_losses: list = None
    kl_losses: list = None
    disc_losses: list = None
    disc_z_losses: list = None
    train_accuracies: list = None
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

with mirrored_strategy.scope():
    gen = gen
    dis_Y = dis_Y
    dis_Z = dis_Z
    optimizer_gen = 



def data_parallelism(train_step, dist_inputs):
    per_replica_losses = mirrored_strategy.run(train_step, args=(dist_inputs,))
    return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

for dist_inputs in dataset:
    print(data_parallelism(train_step, dist_inputs))


for epoch in range(epochs):
    print(f"Epoch {epoch + 1} started")

    total_gen_loss = tf.constant(0.0, dtype=tf.float32)
    total_rec_loss = tf.constant(0.0, dtype=tf.float32)
    total_lm_loss = tf.constant(0.0, dtype=tf.float32)
    total_adv_loss = tf.constant(0.0, dtype=tf.float32)
    total_kl_loss = tf.constant(0.0, dtype=tf.float32)
    total_disc_loss = tf.constant(0.0, dtype=tf.float32)
    total_disc_z_loss = tf.constant(0.0, dtype=tf.float32)
    total_accuracy = tf.constant(0.0, dtype=tf.float32)
    total_reinforce_loss = tf.constant(0.0, dtype=tf.float32)
    num_batches = 0

    # 生成器鉴别器交替训练
    for batch_X, batch_Y in zip(train_tf_dataset_X, train_tf_dataset_Y):
        # 对于 X 数据
        batch_input_ids_X = batch_X['input_ids']
        batch_attention_mask_X = batch_X['attention_mask']
        batch_labels_X = ex.create_labels(batch_input_ids_X, batch_attention_mask_X)
        batch_styles_X = batch_X['style']

        # 对于 Y 数据
        batch_input_ids_Y = batch_Y['input_ids']
        batch_attention_mask_Y = batch_Y['attention_mask']
        batch_labels_Y = ex.create_labels(batch_input_ids_Y, batch_attention_mask_Y)
        batch_styles_Y = batch_Y['style']
        print("Processing batch")
        # print("batch_labels_X:", batch_labels_X)

        # 动态 Padding
        max_len_X, batch_input_ids_X, batch_attention_mask_X = dis.dynamic_padding(batch_input_ids_X, batch_attention_mask_X)
        max_len_Y, batch_input_ids_Y, batch_attention_mask_Y = dis.dynamic_padding(batch_input_ids_Y, batch_attention_mask_Y)
        max_len = max(max_len_X, max_len_Y) + 101
        
        batch_attention_mask_X = tf.convert_to_tensor(batch_attention_mask_X, dtype=tf.int32)
        batch_attention_mask_Y = tf.convert_to_tensor(batch_attention_mask_Y, dtype=tf.int32)

        print("batch_input_ids_X shape:", batch_input_ids_X.shape)
        print("batch_attention_mask_X shape:", batch_attention_mask_X.shape)
        print("batch_input_ids_Y shape:", batch_input_ids_Y.shape)
        print("batch_attention_mask_Y.shape:", batch_attention_mask_Y.shape)

        # 生成器
        print("Training gen")
        gen_loss, rec_loss, lm_loss, adv_loss, kl_loss, current_lr, accuracy = train_gen_step(gen, dis_Y, 
                                                                                batch_input_ids_X, batch_attention_mask_X, batch_labels_X, 
                                                                                batch_styles_X, max_len, tf.cast(lr_step, tf.float32), 
                                                                                accumulation_steps, lambda_rec, lambda_lm, lambda_adv, 
                                                                                lambda_kl, gamma)
        print("Training REINFORCE")
        reinforce_loss = reinforce_step(batch_input_ids_X, batch_attention_mask_X, batch_labels_X, batch_styles_X, max_len, gen, 
                                                                            dis_Y, gen_optimizer)

        # 鉴别器
        generated_ids_Y = gen.generate(batch_input_ids_X, attention_mask=batch_attention_mask_X, max_length=max_len)
        print("Training discriminator Y")
        disc_loss_Y, real_loss_Y, fake_loss_Y = train_dis_Y_step(batch_input_ids_Y, batch_attention_mask_Y, generated_ids_Y, 
                                                                           dis_Y, dis_Y_optimizer)
        print("Training discriminator Z")
        disc_z_loss_X, disc_z_loss_Y = train_gen_with_dis_Z(gen, dis_Z, batch_input_ids_X, batch_input_ids_Y, 
                                                                            batch_attention_mask_X, batch_attention_mask_Y, 
                                                                            batch_styles_X, batch_styles_Y, max_len, gen_optimizer, 
                                                                            dis_Z_optimizer, label_smoothing)
        print("Epoch completed")
        
    total_gen_loss += tf.cast(gen_loss, tf.float32)
    print("gen loss: ", gen_loss)
    total_rec_loss += tf.cast(rec_loss, tf.float32)
    total_lm_loss += tf.cast(lm_loss, tf.float32)
    total_adv_loss += tf.cast(adv_loss, tf.float32)
    total_kl_loss += tf.cast(kl_loss, tf.float32)
    total_disc_loss += tf.cast(disc_loss_Y, tf.float32)
    total_disc_z_loss += tf.cast((disc_z_loss_X + disc_z_loss_Y) / 2, tf.float32)
    total_accuracy += tf.cast(accuracy, tf.float32)
    total_reinforce_loss += tf.cast(reinforce_loss, tf.float32)
    num_batches += 1
    lr_step += 1
    learning_rates.append(current_lr.numpy())

    # 验证循环
    total_valid_loss = tf.constant(0.0, dtype=tf.float32)
    total_valid_accuracy = tf.constant(0.0, dtype=tf.float32)
    num_valid_batches = 0
    for batch_valid_X in valid_tf_dataset_X:
        batch_valid_ids = batch_valid_X['input_ids']
        batch_valid_attention_mask = batch_valid_X['attention_mask']
        batch_valid_labels = ex.create_labels(batch_valid_ids, batch_valid_attention_mask)
        batch_valid_styles = batch_valid_X['style']

        batch_valid_attention_mask = tf.convert_to_tensor(batch_valid_attention_mask, dtype=tf.int32)
        loss, accuracy = ex.valid_step(gen, batch_valid_ids, batch_valid_attention_mask, batch_valid_labels, batch_valid_styles)
        total_valid_loss += loss
        total_valid_accuracy += accuracy
        num_valid_batches += 1
    avg_train_loss = total_gen_loss / tf.cast(num_batches, tf.float32)
    avg_rec_loss = total_rec_loss / tf.cast(num_batches, tf.float32)
    avg_lm_loss = total_lm_loss / tf.cast(num_batches, tf.float32)
    avg_adv_loss = total_adv_loss / tf.cast(num_batches, tf.float32)
    avg_kl_loss = total_kl_loss / tf.cast(num_batches, tf.float32)
    avg_disc_loss = total_disc_loss / tf.cast(num_batches, tf.float32)
    avg_disc_z_loss = total_disc_z_loss / tf.cast(num_batches, tf.float32)
    avg_train_accuracy = total_accuracy / tf.cast(num_batches, tf.float32)
    avg_reinforce_loss = total_reinforce_loss / tf.cast(num_batches, tf.float32)
    avg_valid_loss = total_valid_loss / tf.cast(num_valid_batches, tf.float32)
    avg_valid_accuracy = total_valid_accuracy / tf.cast(num_valid_batches, tf.float32)
    train_losses.append(avg_train_loss.numpy())
    rec_losses.append(avg_rec_loss.numpy())
    lm_losses.append(avg_lm_loss.numpy())
    adv_losses.append(avg_adv_loss.numpy())
    kl_losses.append(avg_kl_loss.numpy())
    disc_losses.append(avg_disc_loss.numpy())
    disc_z_losses.append(avg_disc_z_loss.numpy())
    train_accuracies.append(avg_train_accuracy.numpy())
    valid_losses.append(avg_valid_loss.numpy())
    valid_accuracies.append(avg_valid_accuracy.numpy())
    train_perplexity = tf.exp(avg_train_loss).numpy()
    valid_perplexity = tf.exp(avg_valid_loss).numpy()
    perplexities.append(valid_perplexity)
    print(f"Epoch {epoch + 1} ended")
    print(f"Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")
    print(f"Train Accuracy: {avg_train_accuracy:.4f}, Valid Accuracy: {avg_valid_accuracy:.4f}")
    print(f"Train Perplexity: {train_perplexity:.4f}, Valid Perplexity: {valid_perplexity:.4f}")
    print(f"Discriminator Loss: {avg_disc_loss:.4f}")
    print(f"REINFORCE Loss: {avg_reinforce_loss:.4f}")
    print("---")
