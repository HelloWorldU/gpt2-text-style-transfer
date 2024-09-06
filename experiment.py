import tensorflow as tf
import os
import re
import matplotlib.pyplot as plt
import random 
import pre_progress as pr

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
def loss_curve(train_loss, valid_loss, save_dir):
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
def accuracy_curve(train_accuracy, valid_accuracy, save_dir):
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

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 测试集评估函数
def test_evalution(gen, test_step, dataset):
    test_perplexity = 0.0
    total_test_loss = tf.constant(0.0,dtype=tf.float32)
    total_test_accuracy = tf.constant(0.0,dtype=tf.float32)
    num_test_batches = 0
    for batch_test_X in dataset:
        batch_input_ids = batch_test_X['input_ids']
        batch_attention_mask = batch_test_X['attention_mask']
        batch_labels = create_labels(batch_input_ids, batch_attention_mask)
        styles = batch_test_X['style']
        batch_attention_mask = tf.convert_to_tensor(batch_attention_mask, tf.int32)
        loss,accuracy = test_step(gen, batch_input_ids, batch_attention_mask, batch_labels, styles)
        total_test_loss += loss
        total_test_accuracy += accuracy
        num_test_batches += 1
    avg_test_loss = total_test_loss / tf.cast(num_test_batches, tf.float32)
    avg_test_accuracy = total_test_accuracy / tf.cast(num_test_batches, tf.float32)
    test_perplexity = tf.exp(avg_test_loss).numpy()
    print(f"Test Loss: {avg_test_loss}, Test Accuracy: {avg_test_accuracy}")
    print(f"Test Perplexity: {test_perplexity}")
    return avg_test_accuracy,avg_test_loss,test_perplexity

# 测试集标签
def create_labels(input_ids, attention_mask):
    max_len = tf.reduce_max(tf.map_fn(lambda x: tf.shape(x)[0], input_ids, fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.int32)))
    input_ids = tf.pad(input_ids, [[0, 0], [0, max_len - tf.shape(input_ids)[1]]])
    attention_mask = tf.pad(attention_mask, [[0, 0], [0, max_len - tf.shape(attention_mask)[1]]])
    labels = tf.roll(input_ids, shift=-1, axis=1)
    labels = tf.where(attention_mask == 0, -100, labels)
    return labels

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

# 测试集相关
def get_params(batch_X):
    input_ids = batch_X['input_ids']
    attention_mask = batch_X['attention_mask']
    labels = create_labels(input_ids, attention_mask)
    style = batch_X['style']
    return input_ids, attention_mask, labels, style


# 验证和测试步骤
@tf.function
def valid_step(gen, embedding, input_ids, attention_mask, labels, styles):
    """
    we embed the style ID into the same embedding space as the input IDs
    """
    input_ids = input_ids + tf.expand_dims(styles, axis=1)

    outputs = gen(input_ids=input_ids, attention_mask=attention_mask, training=False)
    logits = outputs.logits
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.cast(labels != -100, tf.float32)
    loss = loss_fn(tf.where(labels == -100, 0, labels), logits)
    loss = tf.reduce_sum(tf.cast(loss, tf.float32) * mask) / tf.reduce_sum(mask)
    predictions = tf.argmax(logits, axis=-1)
    predictions = tf.cast(predictions,tf.int32)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(predictions,labels), tf.float32) * mask) / tf.reduce_sum(mask)
    return tf.cast(loss, tf.float32), accuracy

@tf.function
def test_step(gen, input_ids, attention_mask, labels, styles):
    input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
    attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.int32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    styles = tf.convert_to_tensor(styles, dtype=tf.int32)

    outputs = gen(input_ids=input_ids, attention_mask=attention_mask, training=False)
    logits = outputs.logits
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.cast(labels != -100, tf.float32)
    loss = loss_fn(tf.where(labels == -100, 0, labels), logits)
    loss = tf.reduce_sum(tf.cast(loss, tf.float32) * mask) / tf.reduce_sum(mask)
    predictions = tf.argmax(logits, axis=-1)
    predictions = tf.cast(predictions, tf.int32)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(predictions, labels), tf.float32) * mask) / tf.reduce_sum(mask)
    return tf.cast(loss, tf.float32), accuracy

# 模型初始化时创建层
class Embedding(tf.keras.Model):
    def __init__(self, gen):
        super(Embedding, self).__init__()
        self.gen = gen
        self.style_embeddings_layer = tf.keras.layers.Embedding(input_dim=2, output_dim=gen.config.n_embd)
    def call(self, styles, *args, **kwargs):
        style_embeddings = self.style_embeddings_layer(styles)
        return style_embeddings
    
